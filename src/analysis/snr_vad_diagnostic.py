#!/usr/bin/env python3
"""Diagnostic: compare SNR computed over ALL frames vs Brouhaha-VAD-masked frames.

For each file in the dataset, re-runs Brouhaha inference at full frame
resolution (~16 ms) and computes:
  1. SNR over ALL frames (what we currently store)
  2. SNR over frames where Brouhaha VAD prob > threshold (speech-only)
  3. C50 with the same masking

This validates whether masking with Brouhaha's own VAD brings SNR from
~3 dB up to the expected ~10 dB.

Usage (must run on GPU node):
    uv run python -m src.analysis.snr_vad_diagnostic seedlings_1
"""

import argparse
import io
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl

from src.core.vad_processing import set_seeds
from src.utils import get_dataset_paths, load_manifest

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("snr_diag")

_DEFAULT_MODEL = "models/best/checkpoints/best.ckpt"
_VAD_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def _load_model(model_path: str, device: str):
    """Load Brouhaha model and pipeline, suppressing noisy logs."""
    import torch
    from src.compat import patch_torchaudio

    patch_torchaudio()

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

    _real_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Model.from_pretrained(Path(model_path), strict=False)
    finally:
        sys.stdout = _real_stdout

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev)

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


def _run_brouhaha(pipeline, audio_path: Path):
    """Run Brouhaha and return (vad_prob, snr, c50, step_s) — all raw frames."""
    file = {"uri": audio_path.stem, "audio": str(audio_path)}
    seg = pipeline._segmentation
    segmentations = seg(file)

    # segmentations.data shape: (n_frames, 3) → [vad_prob, snr, c50]
    data = segmentations.data
    vad_prob = data[:, 0].astype(np.float32)
    snr = data[:, 1].astype(np.float32)
    c50 = data[:, 2].astype(np.float32)

    if hasattr(seg.model, "receptive_field"):
        step_s = float(seg.model.receptive_field.step)
    elif hasattr(seg.model, "introspection"):
        step_s = float(seg.model.introspection.frames.step)
    else:
        import soundfile as sf

        info = sf.info(str(audio_path))
        step_s = info.duration / max(len(snr), 1)

    return vad_prob, snr, c50, step_s


def main(dataset: str, device: str = "cuda"):
    warnings.filterwarnings("ignore", message=".*TF32.*deprecated.*")
    set_seeds(42)

    paths = get_dataset_paths(dataset)
    manifest_df = load_manifest(paths.manifest)
    resolved_paths = manifest_df["path"].drop_nulls().to_list()
    file_ids = [Path(p).stem for p in resolved_paths]
    uid_to_path = dict(zip(file_ids, resolved_paths))

    logger.info(f"Dataset: {dataset}  ({len(file_ids)} files)")
    logger.info("Loading Brouhaha model...")
    pipeline = _load_model(_DEFAULT_MODEL, device)
    logger.info("Model loaded.\n")

    # ---- Per-file results ----
    rows = []

    for i, uid in enumerate(file_ids):
        audio_path = Path(uid_to_path[uid])
        logger.info(f"[{i+1}/{len(file_ids)}] {uid}")
        t0 = time.time()

        vad_prob, snr, c50, step_s = _run_brouhaha(pipeline, audio_path)
        n_frames = len(snr)
        duration_s = n_frames * step_s

        logger.info(
            f"  {n_frames:,} frames, {duration_s/3600:.1f}h, step={step_s*1000:.1f}ms"
        )

        # All-frame stats (what we currently compute)
        snr_all = float(np.mean(snr))
        c50_all = float(np.mean(c50))

        # VAD probability distribution
        logger.info(
            f"  VAD prob: mean={vad_prob.mean():.3f}, "
            f"median={np.median(vad_prob):.3f}, "
            f">0.5: {(vad_prob > 0.5).mean():.1%}"
        )

        row = {
            "uid": uid,
            "n_frames": n_frames,
            "duration_h": round(duration_s / 3600, 2),
            "step_ms": round(step_s * 1000, 2),
            "vad_prob_mean": round(float(vad_prob.mean()), 4),
            "vad_frac_gt50": round(float((vad_prob > 0.5).mean()), 4),
            "snr_all_frames": round(snr_all, 2),
            "c50_all_frames": round(c50_all, 2),
        }

        # Masked stats at each VAD threshold
        for thresh in _VAD_THRESHOLDS:
            mask = vad_prob > thresh
            n_speech = int(mask.sum())
            frac = n_speech / n_frames if n_frames > 0 else 0

            if n_speech > 0:
                snr_masked = float(np.mean(snr[mask]))
                c50_masked = float(np.mean(c50[mask]))
            else:
                snr_masked = float("nan")
                c50_masked = float("nan")

            t_str = str(int(thresh * 100))
            row[f"speech_frac_{t_str}"] = round(frac, 4)
            row[f"snr_vad{t_str}"] = round(snr_masked, 2)
            row[f"c50_vad{t_str}"] = round(c50_masked, 2)

            if thresh == 0.5:
                logger.info(
                    f"  VAD>{thresh}: {frac:.1%} speech → "
                    f"SNR={snr_masked:.1f} dB, C50={c50_masked:.1f} dB"
                )

        # Compare to our existing pooled values
        existing_path = paths.output / "snr" / f"{uid}.npz"
        if existing_path.exists():
            existing = np.load(existing_path)
            ex_snr = existing["snr"].astype(np.float32)
            row["snr_existing_pooled"] = round(float(ex_snr.mean()), 2)
        else:
            row["snr_existing_pooled"] = float("nan")

        elapsed = time.time() - t0
        logger.info(
            f"  SNR all={snr_all:.1f} dB → VAD-masked={row.get('snr_vad50', float('nan')):.1f} dB  "
            f"({elapsed:.0f}s)\n"
        )
        rows.append(row)

    # ---- Summary ----
    df = pl.DataFrame(rows)

    print("\n" + "=" * 80)
    print("DIAGNOSTIC RESULTS: SNR with Brouhaha VAD masking")
    print("=" * 80)

    print(f"\nDataset: {dataset}  ({len(file_ids)} files)")
    print(f"{'':30s} {'Mean':>8s} {'Min':>8s} {'Max':>8s}")
    print("-" * 60)

    snr_all = df["snr_all_frames"].to_numpy()
    print(
        f"{'SNR (all frames — current)':30s} {snr_all.mean():8.2f} {snr_all.min():8.2f} {snr_all.max():8.2f}"
    )

    if "snr_existing_pooled" in df.columns:
        ex = df["snr_existing_pooled"].drop_nulls().to_numpy()
        if len(ex) > 0:
            print(
                f"{'SNR (existing 1s pooled)':30s} {ex.mean():8.2f} {ex.min():8.2f} {ex.max():8.2f}"
            )

    for thresh in _VAD_THRESHOLDS:
        t_str = str(int(thresh * 100))
        col = f"snr_vad{t_str}"
        vals = df[col].drop_nulls().to_numpy()
        if len(vals) > 0:
            frac_col = f"speech_frac_{t_str}"
            fracs = df[frac_col].to_numpy()
            print(
                f"{'SNR (VAD>' + str(thresh) + ')':30s} {vals.mean():8.2f} {vals.min():8.2f} {vals.max():8.2f}  "
                f"(speech: {fracs.mean():.1%})"
            )

    print()
    c50_all = df["c50_all_frames"].to_numpy()
    print(
        f"{'C50 (all frames)':30s} {c50_all.mean():8.2f} {c50_all.min():8.2f} {c50_all.max():8.2f}"
    )
    for thresh in [0.5]:
        t_str = str(int(thresh * 100))
        col = f"c50_vad{t_str}"
        vals = df[col].drop_nulls().to_numpy()
        if len(vals) > 0:
            print(
                f"{'C50 (VAD>' + str(thresh) + ')':30s} {vals.mean():8.2f} {vals.min():8.2f} {vals.max():8.2f}"
            )

    # Save results
    out_path = paths.output / "snr_vad_diagnostic.csv"
    df.write_csv(out_path)
    print(f"\nResults saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNR VAD masking diagnostic")
    parser.add_argument("dataset", help="Dataset name (e.g. seedlings_1)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args.dataset, args.device)
