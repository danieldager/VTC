#!/usr/bin/env python3
"""Per-VTC-segment SNR & C50 extraction using Brouhaha.

For each audio file, runs Brouhaha to get per-frame (~16 ms) SNR and C50,
then averages the raw frames falling within each VTC segment's [onset, offset].

Requires VTC segments to already exist in output/{dataset}/vtc_merged/.

Output:
    output/{dataset}/segment_snr.parquet
        columns: uid, onset, offset, label, snr_mean, c50_mean

Usage:
    python -m src.pipeline.segment_snr seedlings_10
    sbatch slurm/segment_snr.slurm seedlings_10

SLURM array support:
    sbatch --array=0-1 slurm/segment_snr.slurm seedlings_10
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from src.core.vad_processing import set_seeds
from src.utils import (
    add_sample_argument,
    get_dataset_paths,
    hhmmss,
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
logger = logging.getLogger("segment_snr")

_DEFAULT_MODEL = "models/best/checkpoints/best.ckpt"


def _load_vtc_segments(output_dir: Path) -> pl.DataFrame:
    """Load all VTC segments from vtc_merged/*.parquet."""
    vtc_dir = output_dir / "vtc_merged"
    if not vtc_dir.exists():
        raise FileNotFoundError(f"VTC segments not found: {vtc_dir}")
    files = sorted(vtc_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {vtc_dir}")
    return pl.concat([pl.read_parquet(f) for f in files])


def _load_brouhaha_pipeline(model_path: str, device: str):
    """Load Brouhaha model and return the pipeline object."""
    import io
    import warnings

    for _mod in (
        "speechbrain",
        "pytorch_lightning",
        "lightning",
        "lightning.fabric",
        "lightning_fabric",
    ):
        logging.getLogger(_mod).setLevel(logging.WARNING)

    from pyannote.audio import Model
    from brouhaha.pipeline import RegressiveActivityDetectionPipeline

    p = Path(model_path)
    if not p.exists():
        if model_path != _DEFAULT_MODEL:
            raise FileNotFoundError(f"Brouhaha checkpoint not found: {model_path}")
        logger.info("Brouhaha checkpoint not found — downloading from Hugging Face …")
        from scripts.download_brouhaha import ensure_brouhaha_checkpoint

        ensure_brouhaha_checkpoint(p)

    logger.info(f"Model: {model_path}")
    _real_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Model.from_pretrained(Path(model_path), strict=False)
    finally:
        sys.stdout = _real_stdout

    import torch

    if model.device.type == "cpu" and device != "cpu":
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        model.to(dev)
        logger.info(f"Device: {dev}")
    else:
        logger.info(f"Device: {model.device}")

    sys.stdout = io.StringIO()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline = RegressiveActivityDetectionPipeline(segmentation=model)
    finally:
        sys.stdout = _real_stdout

    warnings.filterwarnings("ignore", module=r"pyannote\.audio")
    warnings.filterwarnings("ignore", message=".*backend.*parameter.*not used.*")

    return pipeline


def _extract_brouhaha(
    pipeline, audio_path: Path
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run Brouhaha on one file. Returns (raw_snr, raw_c50, step_s)."""
    file = {"uri": audio_path.stem, "audio": str(audio_path)}
    seg = pipeline._segmentation
    segmentations = seg(file)
    data = segmentations.data  # (n_frames, 3): [vad_prob, snr, c50]
    snr = data[:, 1].astype(np.float32)
    c50 = data[:, 2].astype(np.float32)

    if hasattr(seg.model, "receptive_field"):
        step_s = float(seg.model.receptive_field.step)
    elif hasattr(seg.model, "introspection"):
        step_s = float(seg.model.introspection.frames.step)
    elif hasattr(seg, "_frames"):
        step_s = float(seg._frames.step)
    else:
        import soundfile as sf

        info = sf.info(str(audio_path))
        step_s = info.duration / max(len(snr), 1)

    return snr, c50, step_s


def _segment_means(
    raw_snr: np.ndarray,
    raw_c50: np.ndarray,
    step_s: float,
    segments: list[dict],
) -> list[dict]:
    """Compute mean SNR and C50 for each VTC segment from raw frames."""
    results = []
    for seg in segments:
        onset = seg["onset"]
        offset = seg["offset"]
        i0 = max(0, int(onset / step_s))
        i1 = min(len(raw_snr), int(np.ceil(offset / step_s)))
        if i0 >= i1:
            snr_mean = None
            c50_mean = None
        else:
            snr_slice = raw_snr[i0:i1]
            c50_slice = raw_c50[i0:i1]
            snr_mean = round(float(np.mean(snr_slice)), 2)
            c50_mean = round(float(np.mean(c50_slice)), 2)
        results.append(
            {
                "uid": seg["uid"],
                "onset": seg["onset"],
                "offset": seg["offset"],
                "label": seg["label"],
                "snr_mean": snr_mean,
                "c50_mean": c50_mean,
            }
        )
    return results


def main(
    dataset: str,
    model_path: str = _DEFAULT_MODEL,
    device: str = "cuda",
    array_id: int | None = None,
    array_count: int | None = None,
    sample: int | float | None = None,
):
    import warnings

    warnings.filterwarnings("ignore", message=".*TF32.*deprecated.*")
    set_seeds(42)

    paths = get_dataset_paths(dataset)
    logger.info(f"Dataset: {dataset}")
    logger.info(f"  output    : {paths.output}")

    # Load VTC segments
    all_vtc = _load_vtc_segments(paths.output)
    logger.info(f"  VTC segments: {len(all_vtc):,} total")

    # Load manifest for audio paths
    manifest_df = load_manifest(paths.manifest)
    manifest_df = sample_manifest(manifest_df, sample)

    resolved_paths = manifest_df["path"].drop_nulls().to_list()
    file_ids = [Path(p).stem for p in resolved_paths]
    uid_to_path = dict(zip(file_ids, resolved_paths))

    if array_id is not None and array_count is not None:
        file_ids = shard_list(file_ids, array_id, array_count)
        logger.info(f"Shard {array_id}/{array_count - 1}: {len(file_ids)} files")

    shard_id = array_id if array_id is not None else 0

    # Load model
    from src.compat import patch_torchaudio

    patch_torchaudio()
    pipeline = _load_brouhaha_pipeline(model_path, device)

    # Resolve frame step for logging
    seg_model = pipeline._segmentation
    if hasattr(seg_model.model, "receptive_field"):
        step_ms = float(seg_model.model.receptive_field.step) * 1000
    elif hasattr(seg_model.model, "introspection"):
        step_ms = float(seg_model.model.introspection.frames.step) * 1000  # type: ignore
    else:
        step_ms = 16.9
    logger.info(f"  frame step: {step_ms:.1f} ms")

    # Process files
    all_rows: list[dict] = []
    n_errors = 0
    total = len(file_ids)
    t0 = time.time()
    log_every = max(1, total // 20)

    for i, uid in enumerate(file_ids, 1):
        audio_path = Path(uid_to_path[uid])
        try:
            raw_snr, raw_c50, step_s = _extract_brouhaha(pipeline, audio_path)

            # Get VTC segments for this file
            file_segs = all_vtc.filter(pl.col("uid") == uid)
            seg_dicts = file_segs.to_dicts()

            rows = _segment_means(raw_snr, raw_c50, step_s, seg_dicts)
            all_rows.extend(rows)

            if i % log_every == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (total - i) / rate if rate > 0 else 0
                eta = (
                    f"{remaining / 60:.0f}m"
                    if remaining < 3600
                    else f"{remaining / 3600:.1f}h"
                )
                print(
                    f"  {i:>4}/{total}  segments={len(all_rows):,}  ETA {eta}",
                    flush=True,
                )

        except Exception as e:
            n_errors += 1
            logger.warning(f"{uid}: {e}")

    # Save results
    out_dir = paths.output / "segment_snr"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard_{shard_id}.parquet"

    if all_rows:
        df = pl.DataFrame(all_rows)
        df.write_parquet(out_path)
        logger.info(f"Saved {len(df):,} segment SNR rows to {out_path}")

        # Summary
        snr_vals = df["snr_mean"].drop_nulls()
        c50_vals = df["c50_mean"].drop_nulls()
        if len(snr_vals) > 0:
            logger.info(
                f"  SNR : mean={snr_vals.mean():.1f} dB  "
                f"std={snr_vals.std():.1f} dB  "
                f"range=[{snr_vals.min():.1f}, {snr_vals.max():.1f}]"
            )
        if len(c50_vals) > 0:
            logger.info(
                f"  C50 : mean={c50_vals.mean():.1f} dB  "
                f"std={c50_vals.std():.1f} dB"
            )
    else:
        logger.info("No segments processed.")

    wall = time.time() - t0
    logger.info(f"{'─' * 50}")
    logger.info(
        f"Shard {shard_id}: {total - n_errors}/{total} files, {n_errors} errors"
    )
    logger.info(f"Wall time: {hhmmss(wall)}")

    log_benchmark(
        step="segment_snr",
        dataset=dataset,
        n_files=total,
        wall_seconds=wall,
        total_bytes=0,
        n_workers=1,
        extra={"shard_id": shard_id},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-VTC-segment SNR & C50 via Brouhaha.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.pipeline.segment_snr seedlings_10\n"
            "  sbatch slurm/segment_snr.slurm seedlings_10\n"
        ),
    )
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument(
        "--model_path",
        default=_DEFAULT_MODEL,
        help=f"Brouhaha model checkpoint (default: {_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference (default: cuda)",
    )
    parser.add_argument("--array_id", type=int, help="SLURM array task ID")
    parser.add_argument("--array_count", type=int, help="Total SLURM array tasks")
    add_sample_argument(parser)

    args = parser.parse_args()
    main(**vars(args))
