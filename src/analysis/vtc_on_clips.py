#!/usr/bin/env python3
"""
Run VTC inference on audio clips packed in WebDataset shards.

For each clip in output/{dataset}/shards/*.tar, extracts the audio in memory,
runs VTC inference, and saves clip-relative segments to:

    output/{dataset}/vtc_clips/vtc_raw/shard_{id}.parquet
    output/{dataset}/vtc_clips/vtc_merged/shard_{id}.parquet

Segments use 'clip_id' ({uid}_{clip_idx:04d}) as identifier with
onset/offset relative to the start of each clip (0 = clip start).

Run on a compute node via slurm/vtc_clips.slurm.

Usage:
    python -m src.analysis.vtc_on_clips seedlings_10
    python -m src.analysis.vtc_on_clips seedlings_10 \\
        --array_id $SLURM_ARRAY_TASK_ID \\
        --array_count $SLURM_ARRAY_TASK_COUNT
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal

import polars as pl
import torch

from src.compat import patch_torchaudio

patch_torchaudio()

from segma.config import load_config
from segma.inference import apply_model_on_audio
from segma.models import Models
from segma.utils.encoders import MultiLabelEncoder

from src.core.intervals import intervals_to_segments
from src.core.vad_processing import set_seeds
from src.pipeline.vtc import _apply_threshold
from src.utils import (
    atomic_write_parquet,
    get_dataset_paths,
    hhmmss,
    merge_segments_df,
    shard_list,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("vtc_clips")

_EMPTY_SCHEMA = {
    "clip_id": pl.String,
    "onset": pl.Float64,
    "offset": pl.Float64,
    "duration": pl.Float64,
    "label": pl.String,
}


def main(
    dataset: str,
    config: str = "VTC-2.0/model/config.yml",
    checkpoint: str = "VTC-2.0/model/best.ckpt",
    threshold: float = 0.5,
    min_duration_on_s: float = 0.1,
    min_duration_off_s: float = 0.3,
    batch_size: int = 128,
    device: Literal["cuda", "cpu", "mps"] = "cuda",
    array_id: int | None = None,
    array_count: int | None = None,
) -> None:
    import webdataset as wds

    set_seeds(42)

    paths = get_dataset_paths(dataset)
    shard_dir = paths.output / "shards"
    out_dir = paths.output / "vtc_clips"
    shard_id = array_id if array_id is not None else 0

    logger.info(f"Dataset : {dataset}")
    logger.info(f"Shards  : {shard_dir}")
    logger.info(f"Output  : {out_dir}")

    # ------------------------------------------------------------------
    # Find tar files and assign this worker's subset
    # ------------------------------------------------------------------
    tar_files = sorted(shard_dir.glob("*.tar"))
    if not tar_files:
        logger.error(f"No .tar files found in {shard_dir}")
        sys.exit(1)
    logger.info(f"Found {len(tar_files)} tar shards")

    if array_id is not None and array_count is not None:
        tar_files = shard_list(tar_files, array_id, array_count)
        logger.info(f"Shard {array_id}/{array_count - 1}: {len(tar_files)} tars")

    # ------------------------------------------------------------------
    # Resume: collect already-completed clip_ids across ALL shards
    # ------------------------------------------------------------------
    raw_dir = out_dir / "vtc_raw"
    merged_dir = out_dir / "vtc_merged"
    raw_path = raw_dir / f"shard_{shard_id}.parquet"

    completed_clip_ids: set[str] = set()
    if raw_dir.exists():
        existing = sorted(raw_dir.glob("shard_*.parquet"))
        if existing:
            prev_all = pl.read_parquet(existing)
            completed_clip_ids = set(prev_all["clip_id"].to_list())
            logger.info(f"Resume: {len(completed_clip_ids)} clips already done")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    logger.info(f"Loading model: {Path(config).stem}")
    model_config = load_config(config)
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
    # Process clips
    # ------------------------------------------------------------------
    seg_rows: list[dict] = []
    n_processed = 0
    n_skipped = 0
    n_errors = 0
    t0 = time.time()
    last_log_t = t0

    for tar_path in tar_files:
        ds = wds.WebDataset(str(tar_path))  # type: ignore
        for sample in ds:
            clip_id: str = sample["__key__"]

            if clip_id in completed_clip_ids:
                n_skipped += 1
                continue

            # Find audio bytes (prefer flac, fall back to wav)
            audio_bytes: bytes | None = None
            audio_ext: str | None = None
            for ext in ("flac", "wav"):
                if ext in sample:
                    audio_bytes = sample[ext]
                    audio_ext = ext
                    break

            if audio_bytes is None:
                logger.warning(f"{clip_id}: no audio found in sample, skipping")
                n_errors += 1
                continue

            # Write audio to a temp file, run VTC, then immediately discard
            tmp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=f".{audio_ext}", delete=False
                ) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = Path(tmp.name)

                with torch.no_grad():
                    logits_t = apply_model_on_audio(
                        audio_path=tmp_path,
                        model=model,
                        conv_settings=conv_settings,
                        device=device,
                        batch_size=batch_size,
                        chunk_duration_s=chunk_duration_s,
                    )

                region_data = [(0, logits_t.cpu())]
                intervals = _apply_threshold(
                    region_data, threshold, conv_settings, l_encoder
                )
                # intervals_to_segments uses the second arg as "uid" field name
                clip_segs = intervals_to_segments(intervals, clip_id)
                seg_rows.extend(clip_segs)

            except Exception as e:
                logger.warning(f"{clip_id}: {e}")
                n_errors += 1
                continue

            finally:
                if tmp_path is not None and tmp_path.exists():
                    os.unlink(tmp_path)

            n_processed += 1
            now = time.time()
            if now - last_log_t >= 60:
                elapsed = now - t0
                rate = n_processed / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  {n_processed} processed, {n_skipped} skipped, "
                    f"{n_errors} errors  ({rate:.1f} clips/s)"
                )
                last_log_t = now

    # ------------------------------------------------------------------
    # Build output DataFrame
    # intervals_to_segments returns {"uid": clip_id, ...}; rename to clip_id
    # ------------------------------------------------------------------
    new_seg_df = (
        pl.DataFrame(seg_rows).rename({"uid": "clip_id"})
        if seg_rows
        else pl.DataFrame(schema=_EMPTY_SCHEMA)
    )

    # Merge with previously saved rows for this shard (resume case)
    if raw_path.exists() and completed_clip_ids:
        prev_df = pl.read_parquet(raw_path)
        new_ids = set(new_seg_df["clip_id"].to_list())
        kept = prev_df.filter(~pl.col("clip_id").is_in(list(new_ids)))
        parts = [p for p in [kept, new_seg_df] if not p.is_empty()]
        seg_df = pl.concat(parts) if parts else pl.DataFrame(schema=_EMPTY_SCHEMA)
    else:
        seg_df = new_seg_df

    # ------------------------------------------------------------------
    # Save raw segments
    # ------------------------------------------------------------------
    raw_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(seg_df, raw_path)

    # ------------------------------------------------------------------
    # Apply collar/min-duration merge and save
    # merge_segments_df requires a "uid" column; rename and restore
    # ------------------------------------------------------------------
    merged_df = merge_segments_df(
        seg_df.rename({"clip_id": "uid"}),
        min_duration_off_s,
        min_duration_on_s,
    ).rename({"uid": "clip_id"})

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_path = merged_dir / f"shard_{shard_id}.parquet"
    atomic_write_parquet(merged_df, merged_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    wall = time.time() - t0
    logger.info("─" * 50)
    logger.info(f"Shard {shard_id} complete")
    logger.info(
        f"  Processed : {n_processed} clips  "
        f"({n_errors} errors, {n_skipped} skipped)"
    )
    logger.info(f"  Segments  : {len(seg_df):,} raw, {len(merged_df):,} merged")
    logger.info(f"  Wall time : {hhmmss(wall)}")
    logger.info("─" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run VTC inference on clips from WebDataset shards, "
            "storing clip-relative segments for alignment analysis."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  sbatch slurm/vtc_clips.slurm seedlings_10\n"
            "  python -m src.analysis.vtc_on_clips seedlings_10 --device cpu\n"
        ),
    )
    parser.add_argument("dataset", help="Dataset name (e.g. seedlings_10)")
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
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold (default: 0.5)",
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
        default=0.3,
        help="Merge same-label gaps smaller than this (default: 0.3s)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Model forward-pass batch size (default: 128)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Inference device (default: cuda)",
    )
    parser.add_argument("--array_id", type=int, help="SLURM array task ID")
    parser.add_argument("--array_count", type=int, help="Total SLURM array tasks")

    args = parser.parse_args()
    main(**vars(args))
