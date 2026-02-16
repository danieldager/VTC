#!/usr/bin/env python3
"""
VTC inference with adaptive per-file thresholding.

For each audio file in the manifest shard:
  1. Forward pass through segma model → raw logits
  2. If VAD segments are available: sweep sigmoid thresholds (high → low)
     and pick the highest threshold where VTC–VAD IoU ≥ ``--target_iou``
  3. Otherwise: apply default threshold (0.5)
  4. Convert thresholded logits to labelled speech segments

Paths are derived from the dataset name:
    manifests/{dataset}.parquet        input manifest
    output/{dataset}/vtc_raw/          raw VTC segments   (parquet shards)
    output/{dataset}/vtc_merged/       merged VTC segments (parquet shards)
    output/{dataset}/vtc_meta/         per-file metadata   (parquet shards)
    output/{dataset}/logits/           [optional] saved logits
    output/{dataset}/vad_merged/       VAD reference (read-only)
    metadata/{dataset}/                VAD metadata  (read-only)

Usage:
    python scripts/infer.py chunks30
    python scripts/infer.py chunks30 --target_iou 0.85

SLURM array:
    python scripts/infer.py chunks30 \\
        --array_id $SLURM_ARRAY_TASK_ID \\
        --array_count $SLURM_ARRAY_TASK_COUNT
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Literal

import polars as pl
import torch

from segma.config import load_config
from segma.config.base import Config
from segma.inference import apply_model_on_audio, apply_thresholds, create_intervals
from segma.models import Models
from segma.utils.conversions import frames_to_seconds
from segma.utils.encoders import MultiLabelEncoder

from scripts.utils import (
    atomic_write_parquet,
    get_dataset_paths,
    get_log_interval,
    load_manifest,
    log_progress,
    merge_segments_df,
    shard_list,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] - %(message)s",
    datefmt="%Y.%m.%d %H:%M:%S",
)
logger = logging.getLogger("vtc")


# ---------------------------------------------------------------------------
# Interval / IoU helpers
# ---------------------------------------------------------------------------


def _merge_pairs(pairs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping (onset, offset) intervals."""
    if not pairs:
        return []
    pairs = sorted(pairs)
    merged = [pairs[0]]
    for onset, offset in pairs[1:]:
        if onset <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], offset))
        else:
            merged.append((onset, offset))
    return merged


def _total_duration(pairs: list[tuple[float, float]]) -> float:
    return sum(b - a for a, b in pairs)


def compute_iou(
    vtc_pairs: list[tuple[float, float]],
    vad_pairs: list[tuple[float, float]],
) -> float:
    """IoU between two sets of merged (onset, offset) intervals."""
    if not vtc_pairs and not vad_pairs:
        return 1.0
    if not vtc_pairs or not vad_pairs:
        return 0.0

    tp = 0.0
    vi, ai = 0, 0
    while vi < len(vtc_pairs) and ai < len(vad_pairs):
        v_on, v_off = vtc_pairs[vi]
        a_on, a_off = vad_pairs[ai]
        overlap = min(v_off, a_off) - max(v_on, a_on)
        if overlap > 0:
            tp += overlap
        if v_off <= a_off:
            vi += 1
        else:
            ai += 1

    union = _total_duration(vtc_pairs) + _total_duration(vad_pairs) - tp
    return tp / union if union > 0 else 0.0


def intervals_to_pairs(
    intervals: list[tuple[int, int, str]],
) -> list[tuple[float, float]]:
    """Convert sample-index intervals to merged (onset, offset) pairs in seconds."""
    pairs = []
    for start_s, end_s, _label in intervals:
        onset = round(float(frames_to_seconds(start_s)), 3)
        offset = round(float(frames_to_seconds(end_s)), 3)
        if offset > onset:
            pairs.append((onset, offset))
    return _merge_pairs(sorted(pairs))


def intervals_to_segments(
    intervals: list[tuple[int, int, str]],
    uid: str,
) -> list[dict]:
    """Convert sample-index intervals to segment dicts."""
    rows = []
    for start_s, end_s, label in intervals:
        onset = round(float(frames_to_seconds(start_s)), 3)
        offset = round(float(frames_to_seconds(end_s)), 3)
        duration = round(offset - onset, 3)
        if duration > 0:
            rows.append(
                {
                    "uid": uid,
                    "onset": onset,
                    "offset": offset,
                    "duration": duration,
                    "label": label,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Adaptive threshold sweep
# ---------------------------------------------------------------------------


def find_best_threshold(
    logits_t: torch.Tensor,
    vad_pairs: list[tuple[float, float]],
    conv_settings,
    l_encoder: MultiLabelEncoder,
    thresholds: list[float],
    target_iou: float,
) -> tuple[float, float, list[tuple[int, int, str]]]:
    """Sweep thresholds high→low.  Return (threshold, iou, intervals).

    Returns the highest threshold meeting *target_iou*, or the one with
    the best IoU if none meet the target.
    """
    best_thresh = thresholds[-1]
    best_iou = 0.0
    best_intervals: list[tuple[int, int, str]] = []

    for thresh in thresholds:
        thresh_dict = {
            label: {"lower_bound": thresh, "upper_bound": 1.0}
            for label in l_encoder._labels
        }
        thresholded = apply_thresholds(logits_t, thresh_dict, "cpu").detach()
        intervals = create_intervals(thresholded, conv_settings, l_encoder)
        vtc_pairs = intervals_to_pairs(intervals)
        iou = compute_iou(vtc_pairs, vad_pairs)

        if iou >= target_iou:
            return thresh, iou, intervals

        if iou > best_iou:
            best_iou = iou
            best_thresh = thresh
            best_intervals = intervals

    return best_thresh, best_iou, best_intervals


def apply_default_threshold(
    logits_t: torch.Tensor,
    threshold: float,
    conv_settings,
    l_encoder: MultiLabelEncoder,
) -> list[tuple[int, int, str]]:
    """Apply a single threshold and return intervals."""
    thresh_dict = {
        label: {"lower_bound": threshold, "upper_bound": 1.0}
        for label in l_encoder._labels
    }
    thresholded = apply_thresholds(logits_t, thresh_dict, "cpu").detach()
    return create_intervals(thresholded, conv_settings, l_encoder)


# ---------------------------------------------------------------------------
# VAD reference
# ---------------------------------------------------------------------------


def load_vad_reference(
    output_dir: Path,
) -> dict[str, list[tuple[float, float]]]:
    """Load merged VAD segments and return {uid: [(onset, offset), ...]}."""
    vad_dir = output_dir / "vad_merged"
    if not vad_dir.exists():
        return {}

    files = sorted(vad_dir.glob("*.parquet"))
    if not files:
        return {}

    vad_df = pl.read_parquet(files)
    vad_by_uid: dict[str, list[tuple[float, float]]] = {}
    for (uid,), group_df in vad_df.group_by("uid"):
        pairs = sorted(
            zip(group_df["onset"].to_list(), group_df["offset"].to_list())
        )
        vad_by_uid[uid] = _merge_pairs(pairs)
    return vad_by_uid


# ---------------------------------------------------------------------------
# Per-file metadata row constructors
# ---------------------------------------------------------------------------

_EMPTY_META = dict(
    vtc_threshold=float("nan"),
    vtc_vad_iou=float("nan"),
    vtc_status="",
    vtc_speech_dur=0.0,
    vtc_n_segments=0,
    vtc_label_counts="{}",
    vtc_max_sigmoid=float("nan"),
    vtc_mean_sigmoid=float("nan"),
    error="",
)


def _error_row(uid: str, error: str) -> dict:
    return {**_EMPTY_META, "uid": uid, "vtc_status": "error", "error": error}


def _meta_row(
    uid: str,
    threshold: float,
    iou: float,
    status: str,
    segments: list[dict],
    max_sigmoid: float,
    mean_sigmoid: float,
) -> dict:
    label_counts: dict[str, int] = {}
    speech_dur = 0.0
    for s in segments:
        speech_dur += s["duration"]
        label_counts[s["label"]] = label_counts.get(s["label"], 0) + 1
    return {
        "uid": uid,
        "vtc_threshold": threshold,
        "vtc_vad_iou": round(iou, 4),
        "vtc_status": status,
        "vtc_speech_dur": round(speech_dur, 3),
        "vtc_n_segments": len(segments),
        "vtc_label_counts": json.dumps(label_counts),
        "vtc_max_sigmoid": max_sigmoid,
        "vtc_mean_sigmoid": mean_sigmoid,
        "error": "",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    dataset: str,
    config: str = "VTC-2.0/model/config.yml",
    checkpoint: str = "VTC-2.0/model/best.ckpt",
    target_iou: float = 0.9,
    threshold_max: float = 0.5,
    threshold_min: float = 0.1,
    threshold_step: float = 0.1,
    min_duration_on_s: float = 0.1,
    min_duration_off_s: float = 0.1,
    batch_size: int = 128,
    save_logits: bool = False,
    device: Literal["cuda", "cpu", "mps"] = "cuda",
    array_id: int | None = None,
    array_count: int | None = None,
):
    paths = get_dataset_paths(dataset)
    logger.info(f"Dataset: {dataset}")
    logger.info(f"  manifest : {paths.manifest}")
    logger.info(f"  output   : {paths.output}")
    logger.info(f"  metadata : {paths.metadata}")

    # ------------------------------------------------------------------
    # Load manifest and shard
    # ------------------------------------------------------------------
    manifest_df = load_manifest(paths.manifest)
    file_paths = manifest_df.select("path").to_series().to_list()
    file_ids = [Path(p).stem for p in file_paths]
    uid_to_path = dict(zip(file_ids, file_paths))

    if array_id is not None and array_count is not None:
        file_ids = shard_list(file_ids, array_id, array_count)
        logger.info(
            f"Shard {array_id}/{array_count - 1}: {len(file_ids)} files"
        )

    shard_id = array_id if array_id is not None else 0

    # ------------------------------------------------------------------
    # Resume: skip files already in metadata shard
    # ------------------------------------------------------------------
    meta_dir = paths.output / "vtc_meta"
    meta_path = meta_dir / f"shard_{shard_id}.parquet"
    prev_meta_df: pl.DataFrame | None = None
    completed_uids: set[str] = set()

    if meta_path.exists():
        prev_meta_df = pl.read_parquet(meta_path)
        completed_uids = set(prev_meta_df["uid"].to_list())
        remaining = [uid for uid in file_ids if uid not in completed_uids]
        if len(remaining) < len(file_ids):
            logger.info(
                f"Resume: {len(file_ids) - len(remaining)} done, "
                f"{len(remaining)} remaining"
            )
        file_ids_to_process = remaining
    else:
        file_ids_to_process = list(file_ids)

    if not file_ids_to_process and not file_ids:
        logger.info("No files to process.")
        return

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    logger.info(f"Loading model: {config}")
    model_config: Config = load_config(config)
    l_encoder = MultiLabelEncoder(labels=model_config.data.classes)
    model = Models[model_config.model.name].load_from_checkpoint(
        checkpoint_path=checkpoint,
        label_encoder=l_encoder,
        config=model_config,
        train=False,
    )
    model.eval()
    model.to(torch.device(device))

    conv_settings = model.conv_settings
    chunk_duration_s = model_config.audio.chunk_duration_s

    # ------------------------------------------------------------------
    # Load VAD reference for adaptive thresholding
    # ------------------------------------------------------------------
    vad_by_uid = load_vad_reference(paths.output)
    adaptive = bool(vad_by_uid)
    if adaptive:
        logger.info(f"Adaptive mode: VAD reference loaded ({len(vad_by_uid)} uids)")
    else:
        logger.info("No VAD reference — using default threshold")

    # Threshold sweep list (high → low)
    thresholds: list[float] = []
    t = threshold_max
    while t >= threshold_min - 1e-9:
        thresholds.append(round(t, 4))
        t -= threshold_step
    logger.info(f"Thresholds: {thresholds}  target IoU: {target_iou}")

    # ------------------------------------------------------------------
    # Logit saving (optional)
    # ------------------------------------------------------------------
    logits_dir = paths.output / "logits"
    if save_logits:
        logits_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Process files
    # ------------------------------------------------------------------
    meta_rows: list[dict] = []
    seg_rows: list[dict] = []
    n_met_target = 0
    n_errors = 0
    total = len(file_ids_to_process)
    t0 = time.time()

    for i, uid in enumerate(file_ids_to_process, 1):
        audio_path = Path(uid_to_path[uid])

        # --- Forward pass ---
        try:
            with torch.no_grad():
                logits_t = apply_model_on_audio(
                    audio_path=audio_path,
                    model=model,
                    conv_settings=conv_settings,
                    device=device,
                    batch_size=batch_size,
                    chunk_duration_s=chunk_duration_s,
                )
        except Exception as e:
            n_errors += 1
            logger.warning(f"{uid}: {e}")
            meta_rows.append(_error_row(uid, str(e)))
            continue

        # --- Optional logit save ---
        if save_logits:
            torch.save(
                {
                    l_encoder.inv_transform(j): logits_t[:, j]
                    for j in range(l_encoder.n_labels)
                },
                logits_dir / f"{uid}-logits_dict_t.pt",
            )

        # --- Sigmoid summary ---
        logits_cpu = logits_t.cpu()
        probs = logits_cpu.sigmoid()
        max_sig = round(float(probs.max().item()), 4)
        mean_sig = round(float(probs.mean().item()), 4)

        # --- Choose threshold ---
        vad_pairs = vad_by_uid.get(uid)

        if adaptive and vad_pairs:
            chosen, iou, intervals = find_best_threshold(
                logits_cpu, vad_pairs, conv_settings, l_encoder,
                thresholds, target_iou,
            )
            met = iou >= target_iou
            status = "met_target" if met else "best_effort"
            if met:
                n_met_target += 1

        elif adaptive and not vad_pairs:
            # VAD found no speech → accept default as perfect agreement
            chosen, iou, status = threshold_max, 1.0, "vad_empty"
            intervals = apply_default_threshold(
                logits_cpu, threshold_max, conv_settings, l_encoder,
            )

        else:
            # No VAD data — plain inference
            chosen, iou, status = threshold_max, float("nan"), "default"
            intervals = apply_default_threshold(
                logits_cpu, threshold_max, conv_settings, l_encoder,
            )

        # --- Convert to segments ---
        file_segs = intervals_to_segments(intervals, uid)
        seg_rows.extend(file_segs)
        meta_rows.append(
            _meta_row(uid, chosen, iou, status, file_segs, max_sig, mean_sig)
        )

        if i % get_log_interval(i) == 0 or i == total:
            log_progress(i, total, t0, "VTC")

    # ------------------------------------------------------------------
    # Merge with previous results (resume case)
    # ------------------------------------------------------------------
    new_meta_df = pl.DataFrame(meta_rows) if meta_rows else None

    meta_parts: list[pl.DataFrame] = []
    if prev_meta_df is not None:
        if new_meta_df is not None:
            new_uids = set(new_meta_df["uid"].to_list())
            kept = prev_meta_df.filter(~pl.col("uid").is_in(list(new_uids)))
            if not kept.is_empty():
                meta_parts.append(kept)
        else:
            meta_parts.append(prev_meta_df)
    if new_meta_df is not None:
        meta_parts.append(new_meta_df)

    meta_df = pl.concat(meta_parts) if meta_parts else pl.DataFrame()

    # Segments: merge new + previously saved
    empty_seg = pl.DataFrame(
        schema={
            "uid": pl.String,
            "onset": pl.Float64,
            "offset": pl.Float64,
            "duration": pl.Float64,
            "label": pl.String,
        }
    )
    new_seg_df = pl.DataFrame(seg_rows) if seg_rows else empty_seg

    prev_seg_path = paths.output / "vtc_raw" / f"shard_{shard_id}.parquet"
    if prev_seg_path.exists() and completed_uids:
        prev_seg_df = pl.read_parquet(prev_seg_path)
        new_uids = set(r["uid"] for r in seg_rows) if seg_rows else set()
        kept = prev_seg_df.filter(~pl.col("uid").is_in(list(new_uids)))
        seg_parts = [p for p in [kept, new_seg_df] if not p.is_empty()]
        seg_df = pl.concat(seg_parts) if seg_parts else empty_seg
    else:
        seg_df = new_seg_df

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    meta_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(meta_df, meta_path)
    logger.info(f"Saved: {meta_path}  ({len(meta_df)} rows)")

    vtc_raw_dir = paths.output / "vtc_raw"
    vtc_raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = vtc_raw_dir / f"shard_{shard_id}.parquet"
    atomic_write_parquet(seg_df, raw_path)
    logger.info(f"Saved: {raw_path}  ({len(seg_df)} rows)")

    merged_df = merge_segments_df(seg_df, min_duration_off_s, min_duration_on_s)
    vtc_merged_dir = paths.output / "vtc_merged"
    vtc_merged_dir.mkdir(parents=True, exist_ok=True)
    merged_path = vtc_merged_dir / f"shard_{shard_id}.parquet"
    atomic_write_parquet(merged_df, merged_path)
    logger.info(f"Saved: {merged_path}  ({len(merged_df)} rows)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    processed = total - n_errors
    logger.info(f"\n{'=' * 60}")
    logger.info(f"VTC inference — shard {shard_id}")
    logger.info(f"  Files processed : {total}  (errors: {n_errors})")
    if adaptive and processed > 0:
        logger.info(f"  Met target IoU  : {n_met_target}/{processed}")
        valid = meta_df.filter(pl.col("vtc_threshold").is_not_nan())
        if not valid.is_empty():
            logger.info(f"  Mean threshold  : {valid['vtc_threshold'].mean():.3f}")
            logger.info(f"  Mean IoU        : {valid['vtc_vad_iou'].mean():.3f}")
            logger.info(f"\n  Threshold distribution:")
            for row in (
                valid.group_by("vtc_threshold")
                .len()
                .sort("vtc_threshold", descending=True)
                .iter_rows(named=True)
            ):
                logger.info(f"    {row['vtc_threshold']:.2f}: {row['len']} files")
    logger.info(f"  Segments        : {len(seg_df)} raw, {len(merged_df)} merged")
    logger.info(f"  Output          : {paths.output}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VTC inference with adaptive per-file thresholding."
    )
    parser.add_argument(
        "dataset",
        help="Dataset name (derives manifests/{name}.parquet, output/{name}/, etc.)",
    )
    parser.add_argument(
        "--config", default="VTC-2.0/model/config.yml",
        help="segma model config (default: VTC-2.0/model/config.yml)",
    )
    parser.add_argument(
        "--checkpoint", default="VTC-2.0/model/best.ckpt",
        help="segma model checkpoint (default: VTC-2.0/model/best.ckpt)",
    )
    parser.add_argument(
        "--target_iou", type=float, default=0.9,
        help="Per-file VTC–VAD IoU target for adaptive thresholding (default: 0.9)",
    )
    parser.add_argument(
        "--threshold_max", type=float, default=0.5,
        help="Starting (highest) threshold (default: 0.5)",
    )
    parser.add_argument(
        "--threshold_min", type=float, default=0.1,
        help="Minimum threshold to sweep to (default: 0.1)",
    )
    parser.add_argument(
        "--threshold_step", type=float, default=0.1,
        help="Threshold step size (default: 0.1)",
    )
    parser.add_argument(
        "--min_duration_on_s", type=float, default=0.1,
        help="Remove segments shorter than this (default: 0.1s)",
    )
    parser.add_argument(
        "--min_duration_off_s", type=float, default=0.1,
        help="Fill gaps shorter than this (default: 0.1s)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128,
        help="Batch size for model forward pass (default: 128)",
    )
    parser.add_argument(
        "--save_logits", action="store_true",
        help="Save per-file logits to output/{dataset}/logits/",
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu", "mps"],
        help="Device for inference (default: cuda)",
    )
    parser.add_argument(
        "--array_id", type=int,
        help="SLURM array task ID",
    )
    parser.add_argument(
        "--array_count", type=int,
        help="Total SLURM array tasks",
    )

    args = parser.parse_args()
    main(**vars(args))
