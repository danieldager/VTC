#!/usr/bin/env python3
"""
Retest VTC on a sample of failing files with multiple thresholds.

For each file, the model runs once (saving logits), then thresholds are swept
from 0.1 to 0.5 to measure how many speech seconds are recovered at each level.
The script also runs the same file multiple times (--repeats) at the default
threshold to test for non-determinism between runs.

Usage:
    python scripts/retest_thresholds.py \
        --manifest manifests/retest_sample.parquet \
        --output output/retest \
        --thresholds 0.1 0.2 0.3 0.4 0.5 \
        --repeats 3
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Literal

import polars as pl
import torch
import yaml

from segma.config import load_config
from segma.config.base import Config
from segma.inference import (
    apply_model_on_audio,
    apply_thresholds,
    create_intervals,
)
from segma.models import Models
from segma.utils.conversions import frames_to_seconds
from segma.utils.encoders import MultiLabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] - %(message)s",
    datefmt="%Y.%m.%d %H:%M:%S",
)
logger = logging.getLogger("retest")


def intervals_to_total_dur(intervals: list[tuple[int, int, str]]) -> float:
    """Sum the duration of all intervals in seconds."""
    total = 0.0
    for start_f, end_f, _label in intervals:
        total += frames_to_seconds(end_f - start_f)
    return round(total, 3)  # type: ignore


def intervals_to_label_durs(intervals: list[tuple[int, int, str]]) -> dict[str, float]:
    """Sum duration per label."""
    durs: dict[str, float] = {}
    for start_f, end_f, label in intervals:
        d = frames_to_seconds(end_f - start_f)
        durs[label] = round(durs.get(label, 0.0) + d, 3)  # type: ignore
    return durs


def main(
    manifest: str,
    output: str,
    config_path: str = "VTC-2.0/model/config.yml",
    checkpoint: str = "VTC-2.0/model/best.ckpt",
    thresholds: list[float] | None = None,
    repeats: int = 3,
    batch_size: int = 128,
    device: Literal["gpu", "cuda", "cpu", "mps"] = "cuda",
):
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    device = "cuda" if device == "gpu" else device
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load manifest
    # ------------------------------------------------------------------
    manifest_path = Path(manifest)
    if manifest_path.suffix == ".parquet":
        df = pl.read_parquet(manifest_path)
    else:
        df = pl.read_csv(manifest_path)

    paths = df.select("path").to_series().to_list()
    logger.info(f"Loaded {len(paths)} files from {manifest}")

    # ------------------------------------------------------------------
    # Load model (once)
    # ------------------------------------------------------------------
    config: Config = load_config(config_path)
    l_encoder = MultiLabelEncoder(labels=config.data.classes)
    model = Models[config.model.name].load_from_checkpoint(
        checkpoint_path=checkpoint,
        label_encoder=l_encoder,
        config=config,
        train=False,
    )
    model.eval()
    model.to(torch.device(device))
    labels = list(l_encoder._labels)
    logger.info(f"Model loaded. Labels: {labels}")

    inference_settings = model.conv_settings

    # ------------------------------------------------------------------
    # Run threshold sweep
    # ------------------------------------------------------------------
    threshold_results: list[dict] = []

    for file_i, audio_path_str in enumerate(paths, 1):
        audio_path = Path(audio_path_str)
        uid = audio_path.stem
        logger.info(f"({file_i}/{len(paths)}) {uid}")

        # Get logits (single forward pass)
        try:
            logits_t = apply_model_on_audio(
                audio_path=audio_path,
                model=model,
                batch_size=batch_size,
                chunk_duration_s=config.audio.chunk_duration_s,
                conv_settings=inference_settings,
                device=device,
            )
        except Exception as e:
            logger.error(f"  Failed: {e}")
            for thresh in thresholds:
                threshold_results.append(
                    {
                        "uid": uid,
                        "threshold": thresh,
                        "repeat": 0,
                        "total_speech_s": 0.0,
                        "n_intervals": 0,
                        "label_durs": "{}",
                        "error": str(e),
                    }
                )
            continue

        # Save logits for later inspection
        logits_dir = output_dir / "logits"
        logits_dir.mkdir(exist_ok=True)
        torch.save(
            {
                l_encoder.inv_transform(i): logits_t[:, i]
                for i in range(l_encoder.n_labels)
            },
            logits_dir / f"{uid}-logits.pt",
        )

        # Log max sigmoid per label
        probs = logits_t.sigmoid()
        for li, label in enumerate(labels):
            max_prob = probs[:, li].max().item()
            mean_prob = probs[:, li].mean().item()
            logger.info(f"  {label}: max={max_prob:.4f}  mean={mean_prob:.4f}")

        # Sweep thresholds
        for thresh in thresholds:
            thresh_dict = {
                label: {"lower_bound": thresh, "upper_bound": 1.0}
                for label in l_encoder._labels
            }
            thresholded = apply_thresholds(logits_t, thresh_dict, device).detach().cpu()
            intervals = create_intervals(thresholded, inference_settings, l_encoder)

            total_dur = intervals_to_total_dur(intervals)
            label_durs = intervals_to_label_durs(intervals)

            threshold_results.append(
                {
                    "uid": uid,
                    "threshold": thresh,
                    "repeat": 0,
                    "total_speech_s": total_dur,
                    "n_intervals": len(intervals),
                    "label_durs": json.dumps(label_durs),
                    "error": "",
                }
            )
            logger.info(
                f"  thresh={thresh:.2f}: {total_dur:.1f}s ({len(intervals)} intervals)"
            )

    # ------------------------------------------------------------------
    # Determinism test: re-run at default threshold N times
    # ------------------------------------------------------------------
    logger.info(f"\n=== Determinism test: {repeats} repeats at threshold=0.5 ===")
    determinism_results: list[dict] = []

    for rep in range(1, repeats + 1):
        logger.info(f"--- Repeat {rep}/{repeats} ---")
        for file_i, audio_path_str in enumerate(paths, 1):
            audio_path = Path(audio_path_str)
            uid = audio_path.stem

            try:
                logits_t = apply_model_on_audio(
                    audio_path=audio_path,
                    model=model,
                    batch_size=batch_size,
                    chunk_duration_s=config.audio.chunk_duration_s,
                    conv_settings=inference_settings,
                    device=device,
                )
            except Exception as e:
                determinism_results.append(
                    {
                        "uid": uid,
                        "threshold": 0.5,
                        "repeat": rep,
                        "total_speech_s": 0.0,
                        "n_intervals": 0,
                        "label_durs": "{}",
                        "error": str(e),
                    }
                )
                continue

            thresh_dict = {
                label: {"lower_bound": 0.5, "upper_bound": 1.0}
                for label in l_encoder._labels
            }
            thresholded = apply_thresholds(logits_t, thresh_dict, device).detach().cpu()
            intervals = create_intervals(thresholded, inference_settings, l_encoder)
            total_dur = intervals_to_total_dur(intervals)
            label_durs = intervals_to_label_durs(intervals)

            determinism_results.append(
                {
                    "uid": uid,
                    "threshold": 0.5,
                    "repeat": rep,
                    "total_speech_s": total_dur,
                    "n_intervals": len(intervals),
                    "label_durs": json.dumps(label_durs),
                    "error": "",
                }
            )

        logger.info(f"  Repeat {rep} done")

    # ------------------------------------------------------------------
    # Save combined results
    # ------------------------------------------------------------------
    all_results = threshold_results + determinism_results
    results_df = pl.DataFrame(all_results)
    out_path = output_dir / "retest_results.parquet"
    results_df.write_parquet(out_path, compression="zstd")
    logger.info(f"Saved {out_path} ({len(results_df)} rows)")

    # Also write a human-readable summary
    summary_path = output_dir / "retest_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=== Threshold sweep ===\n\n")
        sweep = results_df.filter(pl.col("repeat") == 0)
        for thresh in thresholds:
            t_rows = sweep.filter(pl.col("threshold") == thresh)
            n_detected = t_rows.filter(pl.col("total_speech_s") > 0).height
            mean_dur = t_rows["total_speech_s"].mean()
            f.write(
                f"threshold={thresh:.2f}: "
                f"{n_detected}/{t_rows.height} files detected, "
                f"mean speech={mean_dur:.1f}s\n"
            )

        f.write("\n=== Determinism test (threshold=0.5) ===\n\n")
        det = results_df.filter(pl.col("repeat") > 0)
        for uid in det["uid"].unique().sort().to_list():
            uid_rows = det.filter(pl.col("uid") == uid)
            durs = uid_rows["total_speech_s"].to_list()
            f.write(f"{uid}: repeats={durs}\n")
            if len(set(durs)) > 1:
                f.write(
                    f"  *** NON-DETERMINISTIC: range={max(durs)-min(durs):.3f}s ***\n"
                )

    logger.info(f"Saved {summary_path}")
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retest VTC with threshold sweep")
    parser.add_argument(
        "--manifest", required=True, help="Parquet/CSV with 'path' column"
    )
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument(
        "--config",
        dest="config_path",
        default="VTC-2.0/model/config.yml",
        help="VTC model config",
    )
    parser.add_argument(
        "--checkpoint", default="VTC-2.0/model/best.ckpt", help="VTC model checkpoint"
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.25, 0.5],
        help="Threshold values to sweep",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeat runs at default threshold for determinism test",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["gpu", "cuda", "cpu", "mps"],
    )

    args = parser.parse_args()
    main(**vars(args))
