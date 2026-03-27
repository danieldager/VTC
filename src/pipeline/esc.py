#!/usr/bin/env python3
"""
PANNs ESC (Environmental Sound Classification) pipeline: classify non-speech audio into AudioSet
categories using a pretrained CNN14 model.

For each audio file in the manifest:
  1. Load the full audio (mono, 32 kHz — PANNs native rate)
  2. Run PANNs CNN14 inference in fixed-length windows (default 10 s)
  3. Keep scores for non-speech AudioSet classes only
  4. Average-pool window-level scores into 1 s bins  → compact float16 array
  5. Save pooled class probabilities as compressed .npz per file

Non-speech masking is done *downstream* in the packaging step using the
VAD output — this pipeline saves predictions for the full audio so users
can apply any mask they wish.

Paths derived from the dataset name:
    manifests/{dataset}.<ext>              input manifest
    output/{dataset}/esc/                pooled class arrays (.npz)
    output/{dataset}/esc_meta/           per-file metadata  (.parquet)

Usage:
    python -m src.pipeline.esc chunks30
    python -m src.pipeline.esc chunks30 --pool_window 1.0

SLURM:
    sbatch slurm/esc.slurm chunks30
"""

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from src.utils import set_seeds
from src.utils import (
    add_sample_argument,
    atomic_write_parquet,
    get_dataset_paths,
    hhmmss,
    load_completed_ids,
    load_manifest,
    log_benchmark,
    sample_manifest,
    shard_list,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("esc")

# ---------------------------------------------------------------------------
# AudioSet category mapping
# ---------------------------------------------------------------------------
# We group 527 AudioSet classes into coarser categories relevant for
# daylong recordings.  The mapping is index-based (0..526).

# Speech-related class indices to EXCLUDE from ESC output:
_SPEECH_INDICES: set[int] = set(range(0, 16))  # classes 0-15 are speech/vocal

# Coarse groupings — each is a set of AudioSet *indices* (0-based, matching the
# class_labels_indices.csv ordering embedded in the NPZ audioset_names array).
# Together these 16 categories cover all 511 non-speech AudioSet classes so the
# residual "other" bucket should always be zero.
_CATEGORY_MAP: dict[str, set[int]] = {
    # ── Vocalizations (non-speech) ───────────────────────────────────────
    "laughter":       set(range(16, 22)),     # 16–21  Laughter, giggle, belly laugh …
    "crying":         set(range(22, 27)),     # 22–26  Crying, baby cry, whimper, wail, sigh
    "singing":        set(range(27, 37)),     # 27–36  Singing incl. child singing, rapping
    # ── Human activity / body sounds ─────────────────────────────────────
    "human_activity": set(range(37, 72)),     # 37–71  Humming, breath, cough, footsteps,
                                              #         clapping, cheering, children playing …
    # ── Animals ──────────────────────────────────────────────────────────
    "animal":         set(range(72, 137)),    # 72–136 All animal sounds (pets → whale)
    # ── Music ─────────────────────────────────────────────────────────────
    "music":          set(range(137, 283)),   # 137–282 Music genres + instruments
    # ── Natural / outdoor sounds ──────────────────────────────────────────
    "nature":         set(range(283, 300)),   # 283–299 Wind, rain, thunder, ocean, fire
    # ── Transport ─────────────────────────────────────────────────────────
    "vehicle":        set(range(300, 344)),   # 300–343 Boats, cars, trains, aircraft, engine
    # ── Machinery & tools ─────────────────────────────────────────────────
    "machinery":      set(range(344, 354)) | set(range(404, 426)),
                                              # 344–353 Chainsaw, drill, engine variants
                                              # 404–425 Mechanisms, clock, camera, drill
    # ── Domestic / household ──────────────────────────────────────────────
    "household":      set(range(354, 388)),   # 354–387 Doors, kitchen, taps, vacuum …
    # ── Alarms, signals & electronic beeps ───────────────────────────────
    "alarm_signal":   set(range(388, 404)) | set(range(481, 500)),
                                              # 388–403 Alarm, phone, siren, whistle
                                              # 481–499 Beep, ping, clang, rumble …
    # ── Impacts, crashes, explosions ─────────────────────────────────────
    "impact":         set(range(426, 481)),   # 426–480 Gunshot, boom, wood/glass/liquid
    # ── Silence & synthetic tones ─────────────────────────────────────────
    "silence":        {500} | set(range(501, 506)),
                                              # 500     Silence
                                              # 501–505 Sine wave, chirp tone, pulse …
    # ── Acoustic environment descriptors ─────────────────────────────────
    "environment":    set(range(506, 524)),   # 506–523 Inside/outside, reverb, echo,
                                              #         static, white/pink noise …
    # ── Media / broadcast ─────────────────────────────────────────────────
    "tv_radio":       {524, 525, 526},        # 524 Television, 525 Radio, 526 Field rec.
}

# Reverse map: audioset index → category name.  Unmatched indices become "other".
_INDEX_TO_CATEGORY: dict[int, str] = {}
for _cat, _indices in _CATEGORY_MAP.items():
    for _idx in _indices:
        _INDEX_TO_CATEGORY[_idx] = _cat

CATEGORIES: list[str] = sorted(_CATEGORY_MAP.keys()) + ["other"]


def audioset_labels() -> list[str]:
    """Return ordered list of 527 AudioSet display names."""
    csv_path = Path.home() / "panns_data" / "class_labels_indices.csv"
    if not csv_path.exists():
        return [f"class_{i}" for i in range(527)]
    names: list[str] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row["display_name"].strip('"'))
    return names


def map_to_categories(probs: np.ndarray) -> dict[str, float]:
    """Map a 527-d probability vector to coarse category probabilities.

    For each category, we take the *max* probability among its constituent
    AudioSet classes — this reflects "what sound is most likely present?"
    rather than summing (which inflates categories with many classes).

    Speech-related indices are excluded.

    Returns dict mapping category name → max probability.
    """
    result: dict[str, float] = {}
    for cat, indices in _CATEGORY_MAP.items():
        non_speech = indices - _SPEECH_INDICES
        if non_speech:
            result[cat] = float(np.max(probs[list(non_speech)]))
        else:
            result[cat] = 0.0

    # "other" = max prob among uncategorised, non-speech classes
    all_mapped = set()
    for indices in _CATEGORY_MAP.values():
        all_mapped |= indices
    other_indices = set(range(527)) - all_mapped - _SPEECH_INDICES
    if other_indices:
        result["other"] = float(np.max(probs[list(other_indices)]))
    else:
        result["other"] = 0.0

    return result


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def extract_panns(
    at,
    audio_path: Path,
    window_s: float = 10.0,
    sr: int = 32000,
    batch_size: int = 1,
) -> tuple[np.ndarray, float]:
    """Run PANNs on an audio file in fixed windows.

    Parameters
    ----------
    at : AudioTagging instance (from panns_inference)
    audio_path : Path to audio file
    window_s : inference window size in seconds
    sr : sample rate for PANNs (32000 native)
    batch_size : number of windows to process per forward pass

    Returns
    -------
    probs : shape (n_windows, 527), float32 class probabilities per window
    step_s : window step = window_s
    """
    import librosa

    # Load full audio at PANNs native rate
    waveform, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    n_samples = len(waveform)
    window_samples = int(window_s * sr)

    if n_samples == 0:
        return np.zeros((0, 527), dtype=np.float32), window_s

    # Chunk into windows
    chunks = []
    for start in range(0, n_samples, window_samples):
        chunk = waveform[start : start + window_samples]
        if len(chunk) < window_samples:
            chunk = np.pad(chunk, (0, window_samples - len(chunk)))
        chunks.append(chunk)

    # Process in batches
    all_probs = []
    for b_start in range(0, len(chunks), batch_size):
        batch = np.stack(chunks[b_start : b_start + batch_size])  # (B, samples)
        probs, _ = at.inference(batch)  # (B, 527), (B, 2048)
        all_probs.append(probs)

    return np.concatenate(all_probs, axis=0).astype(np.float32), window_s


def pool_esc(
    raw_probs: np.ndarray,
    step_s: float,
    pool_window_s: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Average-pool window-level class probabilities into finer bins.

    Parameters
    ----------
    raw_probs : (n_windows, 527), from extract_panns
    step_s : raw window step (e.g. 10s)
    pool_window_s : target pooling window

    Returns
    -------
    pooled : (n_bins, 527) float16
    pool_step_s : effective step
    """
    if pool_window_s >= step_s:
        # Already at or below target resolution — just convert
        return raw_probs.astype(np.float16), step_s

    # Upsample via repeat to approximate finer bins
    repeat_factor = max(1, int(round(step_s / pool_window_s)))
    upsampled = np.repeat(raw_probs, repeat_factor, axis=0)
    return upsampled.astype(np.float16), pool_window_s


def pool_to_categories(
    raw_probs: np.ndarray,
    step_s: float,
    pool_window_s: float = 1.0,
) -> tuple[np.ndarray, list[str], float]:
    """Pool raw (n_windows, 527) probs into (n_bins, n_categories) array.

    Returns
    -------
    pooled : (n_bins, n_categories) float16, one column per category
    categories : sorted category names matching columns
    pool_step_s : effective step in seconds
    """
    cats = CATEGORIES  # sorted list

    # Map each window to category max-probs
    n_windows = raw_probs.shape[0]
    cat_probs = np.zeros((n_windows, len(cats)), dtype=np.float32)
    for i in range(n_windows):
        mapped = map_to_categories(raw_probs[i])
        for j, cat in enumerate(cats):
            cat_probs[i, j] = mapped.get(cat, 0.0)

    # Pool to finer resolution if needed
    if pool_window_s >= step_s:
        return cat_probs.astype(np.float16), cats, step_s

    repeat_factor = max(1, int(round(step_s / pool_window_s)))
    upsampled = np.repeat(cat_probs, repeat_factor, axis=0)
    return upsampled.astype(np.float16), cats, pool_window_s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    dataset: str,
    pool_window: float = 1.0,
    inference_window: float = 10.0,
    batch_size: int = 0,
    device: str = "cuda",
    array_id: int | None = None,
    array_count: int | None = None,
    sample: int | float | None = None,
):
    set_seeds(42)

    paths = get_dataset_paths(dataset)
    logger.info(f"Dataset: {dataset}")
    logger.info(f"  manifest  : {paths.manifest}")
    logger.info(f"  output    : {paths.output}")
    logger.info(f"  inference : {inference_window}s windows")
    logger.info(f"  pool      : {pool_window}s bins")

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
        logger.info(
            f"Shard {array_id}/{array_count - 1}: {len(file_ids)} files"
        )

    shard_id = array_id if array_id is not None else 0

    # ------------------------------------------------------------------
    # Resume: skip files already processed
    # ------------------------------------------------------------------
    esc_dir = paths.output / "esc"
    esc_dir.mkdir(parents=True, exist_ok=True)

    completed_uids: set[str] = set()
    meta_dir = paths.output / "esc_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / f"shard_{shard_id}.parquet"
    prev_meta_df: pl.DataFrame | None = None

    completed_uids = load_completed_ids(meta_dir, id_column="uid", pattern="shard_*.parquet")

    if meta_path.exists():
        prev_meta_df = pl.read_parquet(meta_path)

    remaining = [uid for uid in file_ids if uid not in completed_uids]
    if len(remaining) < len(file_ids):
        skipped = len(file_ids) - len(remaining)
        logger.info(f"Resume: {skipped} done, {len(remaining)} remaining")
    file_ids_to_process = remaining

    if not file_ids_to_process and not file_ids:
        logger.info("No files to process.")
        return

    # ------------------------------------------------------------------
    # Load model (lazy imports — safe on login nodes)
    # ------------------------------------------------------------------
    import torch
    from panns_inference import AudioTagging

    dev = device if torch.cuda.is_available() else "cpu"
    at = AudioTagging(checkpoint_path=None, device=dev)
    logger.info(f"PANNs CNN14 loaded on {dev}")

    # Auto-detect batch size from GPU VRAM when batch_size <= 0
    if batch_size <= 0:
        from src.pipeline.resources import query_local_gpu, recommend_esc_batch_size
        local_gpu = query_local_gpu()
        if local_gpu is not None:
            batch_size = recommend_esc_batch_size(local_gpu.vram_gb)
            logger.info(f"Auto batch_size={batch_size} for {local_gpu.name} ({local_gpu.vram_gb} GB)")
        else:
            batch_size = 1
            logger.info(f"No GPU detected — using default batch_size={batch_size}")
    logger.info(f"  batch_size: {batch_size}")

    # ------------------------------------------------------------------
    # Process files
    # ------------------------------------------------------------------
    meta_rows: list[dict] = []
    n_errors = 0
    total = len(file_ids_to_process)
    t0 = time.time()
    log_every = max(1, total // 20)

    file_sizes: dict[str, int] = {}
    for uid in file_ids_to_process:
        try:
            file_sizes[uid] = Path(uid_to_path[uid]).stat().st_size
        except OSError:
            file_sizes[uid] = 0
    total_bytes = sum(file_sizes.values()) or 1
    bytes_done = 0
    print(
        f"Shard {shard_id}: {total} files, {total_bytes / 1e9:.1f} GB",
        flush=True,
    )

    cats = CATEGORIES
    as_names = audioset_labels()

    for i, uid in enumerate(file_ids_to_process, 1):
        audio_path = Path(uid_to_path[uid])

        try:
            raw_probs, step_s = extract_panns(
                at, audio_path, window_s=inference_window,
                batch_size=batch_size,
            )
            pooled_cats, _, pool_step = pool_to_categories(
                raw_probs, step_s, pool_window,
            )

            # Full 527-class pooled probabilities for fine-grained analysis
            pooled_full, _ = pool_esc(raw_probs, step_s, pool_window)

            # Save per-file: coarse categories + full 527 classes
            np.savez_compressed(
                esc_dir / f"{uid}.npz",
                categories=pooled_cats,             # (n_bins, 13) float16
                category_names=np.array(cats),      # (13,) string
                audioset_probs=pooled_full,          # (n_bins, 527) float16
                audioset_names=np.array(as_names),   # (527,) string
                pool_step_s=np.float32(pool_step),
                inference_step_s=np.float32(step_s),
                n_inference_windows=np.int32(raw_probs.shape[0]),
            )

            # Per-file summary: dominant categories
            mean_cat = pooled_cats.mean(axis=0).astype(np.float32)
            dominant_idx = int(np.argmax(mean_cat))
            dominant_cat = cats[dominant_idx]
            dominant_prob = float(mean_cat[dominant_idx])

            import soundfile as sf
            info = sf.info(str(audio_path))
            file_dur = info.duration

            meta_row = {
                "uid": uid,
                "esc_status": "ok",
                "duration": round(file_dur, 3),
                "n_inference_windows": raw_probs.shape[0],
                "n_pooled_bins": pooled_cats.shape[0],
                "inference_step_s": round(step_s, 6),
                "pool_step_s": round(pool_step, 6),
                "dominant_category": dominant_cat,
                "dominant_prob": round(dominant_prob, 4),
                "error": "",
            }
            # Add per-category mean probabilities
            for j, cat in enumerate(cats):
                meta_row[f"prob_{cat}"] = round(float(mean_cat[j]), 4)

            meta_rows.append(meta_row)

        except Exception as e:
            n_errors += 1
            logger.warning(f"{uid}: {e}")
            meta_row = {
                "uid": uid,
                "esc_status": "error",
                "duration": 0.0,
                "n_inference_windows": 0,
                "n_pooled_bins": 0,
                "inference_step_s": 0.0,
                "pool_step_s": 0.0,
                "dominant_category": "",
                "dominant_prob": 0.0,
                "error": str(e),
            }
            for cat in cats:
                meta_row[f"prob_{cat}"] = 0.0
            meta_rows.append(meta_row)

        bytes_done += file_sizes.get(uid, 0)
        now = time.time()
        elapsed = now - t0
        rate = bytes_done / elapsed if elapsed > 0 else 0
        remaining_bytes = total_bytes - bytes_done
        remaining_s = remaining_bytes / rate if rate > 0 else 0
        eta = (
            f"{remaining_s / 60:.0f}m"
            if remaining_s < 3600
            else f"{remaining_s / 3600:.1f}h"
        )
        pct = 100.0 * bytes_done / total_bytes
        if i % log_every == 0 or i == total:
            print(
                f"  ESC   {i:>4}/{total}"
                f"  {bytes_done / 1e9:.1f}/{total_bytes / 1e9:.1f} GB"
                f" ({pct:.0f}%)  ETA {eta}",
                flush=True,
            )

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

    if not meta_df.is_empty():
        before = len(meta_df)
        meta_df = meta_df.unique(subset=["uid"], keep="last")
        if len(meta_df) < before:
            logger.warning(
                f"Dedup: removed {before - len(meta_df)} duplicate "
                f"meta rows (shard {shard_id})"
            )

    # ------------------------------------------------------------------
    # Save metadata
    # ------------------------------------------------------------------
    atomic_write_parquet(meta_df, meta_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    processed = total - n_errors
    wall = time.time() - t0
    print(flush=True)
    logger.info(f"{'─' * 50}")
    logger.info(f"Shard {shard_id} complete")
    logger.info(f"  Files     : {processed}/{total}  ({n_errors} errors)")
    if not meta_df.is_empty():
        ok = meta_df.filter(pl.col("esc_status") == "ok")
        if not ok.is_empty():
            logger.info(f"  Dominant  : {ok['dominant_category'].value_counts()}")
            logger.info(
                f"  Bins      : {ok['n_pooled_bins'].sum():,} "
                f"({pool_window}s windows)"
            )
    logger.info(f"  Wall time : {hhmmss(wall)}")
    logger.info(f"{'─' * 50}")

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------
    total_bytes_final = sum(
        os.path.getsize(uid_to_path[uid])
        for uid in file_ids_to_process
        if os.path.exists(uid_to_path[uid])
    )
    log_benchmark(
        step="esc",
        dataset=dataset,
        n_files=total,
        wall_seconds=wall,
        total_bytes=total_bytes_final,
        n_workers=1,
        extra={"device": dev, "shard_id": shard_id, "pool_window": pool_window},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PANNs ESC (Environmental Sound Classification) pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.pipeline.esc chunks30\n"
            "  python -m src.pipeline.esc chunks30 --pool_window 0.5\n"
        ),
    )
    parser.add_argument(
        "dataset",
        help="Dataset name — derives output/ and figures/ directories.",
    )
    parser.add_argument(
        "--pool_window", type=float, default=1.0,
        help="Pooled time resolution in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--inference_window", type=float, default=10.0,
        help="PANNs inference window in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
        help="Device for inference (default: cuda)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=0,
        help=(
            "Number of windows per forward pass. "
            "0 = auto-detect from GPU VRAM (default: 0)."
        ),
    )
    parser.add_argument("--array_id", type=int, help="SLURM array task ID")
    parser.add_argument("--array_count", type=int, help="Total SLURM array tasks")
    add_sample_argument(parser)

    args = parser.parse_args()
    main(**vars(args))
