#!/usr/bin/env python3
"""
VTC inference with adaptive per-file thresholding.

For each audio file in the manifest shard:
  1. Forward pass through segma model → raw logits
  2. If VAD segments are available: sweep sigmoid thresholds (high → low)
     and pick the highest threshold where VTC–VAD IoU ≥ ``--target_iou``
  3. Otherwise: apply default threshold (0.5)
  4. Convert thresholded logits to labelled speech segments

When VAD activity regions cover < 90 % of the file the forward pass is
restricted to those regions only — cutting GPU time on long recordings
with low speech ratios by up to 10×.

Paths are derived from the dataset name:
    manifests/{dataset}.parquet        input manifest
    output/{dataset}/vtc_raw/          raw VTC segments   (parquet shards)
    output/{dataset}/vtc_merged/       merged VTC segments (parquet shards)
    output/{dataset}/vtc_meta/         per-file metadata   (parquet shards)
    output/{dataset}/logits/           [optional] saved logits
    output/{dataset}/vad_merged/       VAD reference (read-only)

Usage:
    python -m src.pipeline.vtc chunks30
    python -m src.pipeline.vtc chunks30 --target_iou 0.85

SLURM array:
    python -m src.pipeline.vtc chunks30 \\
        --array_id $SLURM_ARRAY_TASK_ID \\
        --array_count $SLURM_ARRAY_TASK_COUNT
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Literal

import polars as pl
import torch

from segma.config import load_config
from segma.config.base import Config
from segma.models import Models
from segma.utils.encoders import MultiLabelEncoder
from segma.utils.io import get_audio_info

from src.core.intervals import intervals_to_segments
from src.core.metadata import (
    load_vad_merged,
    vtc_error_row,
    vtc_meta_row,
)
from src.core.regions import (
    activity_region_coverage,
    forward_pass_full_file,
    forward_pass_regions,
    merge_into_activity_regions,
)
from src.core.thresholds import (
    apply_default_threshold_regions,
    find_best_threshold_regions,
)
from src.utils import (
    add_sample_argument,
    atomic_write_parquet,
    get_dataset_paths,
    load_manifest,
    log_benchmark,
    merge_segments_df,
    sample_manifest,
    shard_list,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("vtc")


def _hhmmss(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


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
    threshold_high: float = 0.9,
    threshold_step: float = 0.1,
    min_duration_on_s: float = 0.1,
    min_duration_off_s: float = 0.1,
    batch_size: int = 128,
    save_logits: bool = False,
    no_regions: bool = False,
    device: Literal["cuda", "cpu", "mps"] = "cuda",
    array_id: int | None = None,
    array_count: int | None = None,
    sample: int | float | None = None,
):
    paths = get_dataset_paths(dataset)
    logger.info(f"Dataset: {dataset}")
    logger.info(f"  manifest  : {paths.manifest}")
    logger.info(f"  output    : {paths.output}")

    # ------------------------------------------------------------------
    # Load manifest and shard
    # ------------------------------------------------------------------
    manifest_df = load_manifest(paths.manifest)
    manifest_df = sample_manifest(manifest_df, sample)
    if sample is not None:
        logger.info(f"  sample    : {len(manifest_df)} files")
    resolved_paths = manifest_df["path"].drop_nulls().to_list()
    file_ids = [Path(p).stem for p in resolved_paths]
    uid_to_path = dict(zip(file_ids, resolved_paths))

    if array_id is not None and array_count is not None:
        file_ids = shard_list(file_ids, array_id, array_count)
        logger.info(f"Shard {array_id}/{array_count - 1}: " f"{len(file_ids)} files")

    shard_id = array_id if array_id is not None else 0

    # ------------------------------------------------------------------
    # Resume: skip files already processed by ANY shard
    # ------------------------------------------------------------------
    meta_dir = paths.output / "vtc_meta"
    meta_path = meta_dir / f"shard_{shard_id}.parquet"
    prev_meta_df: pl.DataFrame | None = None
    completed_uids: set[str] = set()

    # Check all shard metas so a manifest reorder doesn't re-process
    all_meta_files = sorted(meta_dir.glob("shard_*.parquet"))
    if all_meta_files:
        all_meta = pl.read_parquet(all_meta_files)
        completed_uids = set(all_meta["uid"].to_list())

    if meta_path.exists():
        prev_meta_df = pl.read_parquet(meta_path)

    remaining = [uid for uid in file_ids if uid not in completed_uids]
    if len(remaining) < len(file_ids):
        skipped = len(file_ids) - len(remaining)
        logger.info(f"Resume: {skipped} done, " f"{len(remaining)} remaining")
    file_ids_to_process = remaining

    if not file_ids_to_process and not file_ids:
        logger.info("No files to process.")
        return

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    logger.info(f"Model: {Path(config).stem}")
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
    vad_by_uid = load_vad_merged(paths.output)
    adaptive = bool(vad_by_uid)
    if adaptive:
        logger.info(
            f"Adaptive: {len(vad_by_uid)} VAD refs, " f"target IoU={target_iou}"
        )
    else:
        logger.info("No VAD reference — default threshold")

    # Threshold sweep list (bidirectional from threshold_max)
    # Upward first (reduces over-detection), then downward.
    thresholds: list[float] = []
    t = threshold_max
    while t <= threshold_high + 1e-9:
        thresholds.append(round(t, 4))
        t += threshold_step
    t = threshold_max - threshold_step
    while t >= threshold_min - 1e-9:
        thresholds.append(round(t, 4))
        t -= threshold_step
    th_str = ", ".join(f"{t:.2f}" for t in thresholds)
    logger.info(f"Thresholds: [{th_str}]")

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
    last_log_t = t0
    log_interval_s = 60  # heartbeat every 60 s
    log_every = max(1, total // 20)  # also every ~5 % of files

    # Pre-stat files for GB-based progress
    file_sizes: dict[str, int] = {}
    for uid in file_ids_to_process:
        try:
            file_sizes[uid] = Path(uid_to_path[uid]).stat().st_size
        except OSError:
            file_sizes[uid] = 0
    total_bytes = sum(file_sizes.values()) or 1
    bytes_done = 0
    print(
        f"Shard {shard_id}: {total} files, " f"{total_bytes / 1e9:.1f} GB",
        flush=True,
    )

    for i, uid in enumerate(file_ids_to_process, 1):
        audio_path = Path(uid_to_path[uid])
        vad_pairs = vad_by_uid.get(uid)

        # --- Decide forward-pass strategy ---
        use_regions = False
        activity_regions: list[tuple[float, float]] = []
        file_dur_s = 0.0

        if not no_regions and adaptive and vad_pairs:
            try:
                file_info = get_audio_info(audio_path)
                file_dur_s = file_info.n_samples / 16_000
                activity_regions = merge_into_activity_regions(
                    vad_pairs,
                    file_dur_s,
                )
                coverage = activity_region_coverage(
                    activity_regions,
                    file_dur_s,
                )
                if coverage < 0.9 and activity_regions:
                    use_regions = True
            except Exception:
                pass  # fall through to full-file processing

        # --- Forward pass ---
        try:
            if use_regions:
                region_data = forward_pass_regions(
                    audio_path,
                    model,
                    conv_settings,
                    device,
                    batch_size,
                    chunk_duration_s,
                    activity_regions,
                )
            else:
                region_data = forward_pass_full_file(
                    audio_path,
                    model,
                    conv_settings,
                    device,
                    batch_size,
                    chunk_duration_s,
                )
        except Exception as e:
            n_errors += 1
            logger.warning(f"{uid}: {e}")
            meta_rows.append(vtc_error_row(uid, str(e)))
            continue

        # Move logits to CPU for thresholding
        region_data_cpu = [(off, lg.cpu()) for off, lg in region_data]

        # --- Optional logit save ---
        if save_logits:
            all_logits = torch.cat([lg for _, lg in region_data_cpu], dim=0)
            torch.save(
                {
                    l_encoder.inv_transform(j): all_logits[:, j]
                    for j in range(l_encoder.n_labels)
                },
                logits_dir / f"{uid}-logits_dict_t.pt",
            )

        # --- Sigmoid summary ---
        if region_data_cpu:
            all_logits_cpu = torch.cat([lg for _, lg in region_data_cpu], dim=0)
            probs = all_logits_cpu.sigmoid()
            max_sig = round(float(probs.max().item()), 4)
            mean_sig = round(float(probs.mean().item()), 4)
        else:
            max_sig = 0.0
            mean_sig = 0.0

        # --- Choose threshold ---
        if adaptive and vad_pairs:
            chosen, iou, intervals = find_best_threshold_regions(
                region_data_cpu,
                vad_pairs,
                conv_settings,
                l_encoder,
                thresholds,
                target_iou,
            )
            met = iou >= target_iou
            status = "met_target" if met else "best_effort"
            if met:
                n_met_target += 1

        elif adaptive and not vad_pairs:
            chosen, iou, status = threshold_max, 1.0, "vad_empty"
            intervals = apply_default_threshold_regions(
                region_data_cpu,
                threshold_max,
                conv_settings,
                l_encoder,
            )

        else:
            chosen, iou, status = threshold_max, float("nan"), "default"
            intervals = apply_default_threshold_regions(
                region_data_cpu,
                threshold_max,
                conv_settings,
                l_encoder,
            )

        # Log activity-region stats for the first few files
        if use_regions and i <= 5:
            cov = activity_region_coverage(
                activity_regions,
                file_dur_s,
            )
            logger.info(f"  {uid}")
            logger.info(
                f"    {len(activity_regions)} regions, "
                f"{cov:.0%} coverage, "
                f"{_hhmmss(file_dur_s)} duration"
            )

        # --- Convert to segments ---
        file_segs = intervals_to_segments(intervals, uid)
        seg_rows.extend(file_segs)
        meta_rows.append(
            vtc_meta_row(uid, chosen, iou, status, file_segs, max_sig, mean_sig)
        )

        bytes_done += file_sizes.get(uid, 0)
        now = time.time()
        elapsed = now - t0
        rate = bytes_done / elapsed if elapsed > 0 else 0
        remaining_bytes = total_bytes - bytes_done
        remaining_s = remaining_bytes / rate if rate > 0 else 0
        eta = (
            f"{remaining_s/60:.0f}m"
            if remaining_s < 3600
            else f"{remaining_s/3600:.1f}h"
        )
        pct = 100.0 * bytes_done / total_bytes
        if i % log_every == 0 or i == total or (now - last_log_t) >= log_interval_s:
            print(
                f"  VTC  {i:>4}/{total}"
                f"  {bytes_done/1e9:.1f}/{total_bytes/1e9:.1f} GB"
                f" ({pct:.0f}%)  ETA {eta}",
                flush=True,
            )
            last_log_t = now

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

    # Guard: deduplicate by uid (race condition when shard restarts)
    if not meta_df.is_empty():
        before = len(meta_df)
        meta_df = meta_df.unique(subset=["uid"], keep="last")
        if len(meta_df) < before:
            logger.warning(
                f"Dedup: removed {before - len(meta_df)} duplicate "
                f"meta rows (shard {shard_id})"
            )

    # Segments
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

    # Guard: deduplicate segments (same race condition)
    if not seg_df.is_empty():
        before = len(seg_df)
        seg_df = seg_df.unique()
        if len(seg_df) < before:
            logger.warning(
                f"Dedup: removed {before - len(seg_df)} duplicate "
                f"segment rows (shard {shard_id})"
            )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    meta_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(meta_df, meta_path)

    vtc_raw_dir = paths.output / "vtc_raw"
    vtc_raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = vtc_raw_dir / f"shard_{shard_id}.parquet"
    atomic_write_parquet(seg_df, raw_path)

    merged_df = merge_segments_df(seg_df, min_duration_off_s, min_duration_on_s)
    vtc_merged_dir = paths.output / "vtc_merged"
    vtc_merged_dir.mkdir(parents=True, exist_ok=True)
    merged_path = vtc_merged_dir / f"shard_{shard_id}.parquet"
    atomic_write_parquet(merged_df, merged_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    processed = total - n_errors
    wall = time.time() - t0
    print(flush=True)
    logger.info(f"{'─' * 50}")
    logger.info(f"Shard {shard_id} complete")
    logger.info(f"  Files     : {processed}/{total}  ({n_errors} errors)")
    if adaptive and processed > 0:
        logger.info(f"  Target IoU: {n_met_target}/{processed} met")
        valid = meta_df.filter(pl.col("vtc_threshold").is_not_nan())
        if not valid.is_empty():
            mean_t = valid["vtc_threshold"].mean()
            mean_i = valid["vtc_vad_iou"].mean()
            logger.info(f"  Threshold : {mean_t:.3f} mean  " f"IoU: {mean_i:.3f} mean")
            # Compact threshold distribution on one line
            dist_parts = []
            for row in (
                valid.group_by("vtc_threshold")
                .len()
                .sort("vtc_threshold", descending=True)
                .iter_rows(named=True)
            ):
                dist_parts.append(f"{row['vtc_threshold']:.2f}={row['len']}")
            logger.info(f"  Dist      : {', '.join(dist_parts)}")
    logger.info(f"  Segments  : {len(seg_df):,} raw, " f"{len(merged_df):,} merged")
    logger.info(f"  Wall time : {_hhmmss(wall)}")
    logger.info(f"{'─' * 50}")

    # ---- Benchmark logging ----
    wall_seconds = time.time() - t0
    total_bytes = sum(
        os.path.getsize(uid_to_path[uid])
        for uid in file_ids_to_process
        if os.path.exists(uid_to_path[uid])
    )
    log_benchmark(
        step="vtc",
        dataset=dataset,
        n_files=total,
        wall_seconds=wall_seconds,
        total_bytes=total_bytes,
        n_workers=1,
        extra={"device": device, "shard_id": shard_id},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VTC inference with adaptive per-file thresholding.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.pipeline.vtc chunks30\n"
            "  python -m src.pipeline.vtc chunks30 --target_iou 0.85\n"
            "  python -m src.pipeline.vtc chunks30 --sample 500\n"
        ),
    )
    parser.add_argument(
        "dataset",
        help="Dataset name — used to derive output/, metadata/, and figures/ directories.",
    )
    parser.add_argument(
        "--config",
        default="VTC-2.0/model/config.yml",
        help="segma model config (default: VTC-2.0/model/config.yml)",
    )
    parser.add_argument(
        "--checkpoint",
        default="VTC-2.0/model/best.ckpt",
        help="segma model checkpoint (default: VTC-2.0/model/best.ckpt)",
    )
    parser.add_argument(
        "--target_iou",
        type=float,
        default=0.9,
        help="Per-file VTC–VAD IoU target for adaptive thresholding (default: 0.9)",
    )
    parser.add_argument(
        "--threshold_max",
        type=float,
        default=0.5,
        help="Starting threshold for bidirectional sweep (default: 0.5)",
    )
    parser.add_argument(
        "--threshold_min",
        type=float,
        default=0.1,
        help="Minimum threshold to sweep down to (default: 0.1)",
    )
    parser.add_argument(
        "--threshold_high",
        type=float,
        default=0.9,
        help="Maximum threshold to sweep up to (default: 0.9)",
    )
    parser.add_argument(
        "--threshold_step",
        type=float,
        default=0.1,
        help="Threshold step size (default: 0.1)",
    )
    parser.add_argument(
        "--min_duration_on_s",
        type=float,
        default=0.1,
        help="Remove segments shorter than this (default: 0.1s)",
    )
    parser.add_argument(
        "--min_duration_off_s",
        type=float,
        default=0.1,
        help="Fill gaps shorter than this (default: 0.1s)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for model forward pass (default: 128)",
    )
    parser.add_argument(
        "--save_logits",
        action="store_true",
        help="Save per-file logits to output/{dataset}/logits/",
    )
    parser.add_argument(
        "--no_regions",
        action="store_true",
        help="Disable activity-region optimization; always run VTC on full files.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device for inference (default: cuda)",
    )
    parser.add_argument(
        "--array_id",
        type=int,
        help="SLURM array task ID",
    )
    parser.add_argument(
        "--array_count",
        type=int,
        help="Total SLURM array tasks",
    )
    add_sample_argument(parser)

    args = parser.parse_args()
    main(**vars(args))
