#!/usr/bin/env python3
"""
Investigate anomalously long VTC turns (turns where activity from one speaker
type is uninterrupted for longer than a threshold).

Steps
-----
1. Load vtc_merged for the dataset.
2. Filter turns whose duration exceeds ``--threshold`` (default 20 s).
3. Print a text summary and save a CSV log of all long turns.
4. Produce diagnostic plots (saved to figures/{dataset}/long_turns/).
5. Extract the ``--top-n`` longest turns (default 50) from the source audio
   files, with ``--pad`` seconds (default 10 s) of context on each side, and
   write them as WAV files to ``data/long_turns/``.

Requires:
    output/{dataset}/vtc_merged/      VTC segments (shard_*.parquet)
    manifests/{dataset}.*             recording manifest with 'path' column

Outputs:
    output/{dataset}/long_turns.csv           All long turns tabulated
    figures/{dataset}/long_turns/             Diagnostic plots
    data/long_turns/                          Extracted audio clips

Usage (login node — no GPU required):
    uv run python -m src.analysis.long_turns seedlings_10
    uv run python -m src.analysis.long_turns seedlings_10 --threshold 30 --top-n 20 --pad 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
import soundfile as sf

from src.core import LABEL_COLORS, VTC_LABELS
from src.utils import get_dataset_paths

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("long_turns")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LABELS = VTC_LABELS  # ["FEM", "MAL", "KCHI", "OCH"]


def _load_vtc_merged(vtc_dir: Path) -> pl.DataFrame:
    files = sorted(vtc_dir.glob("shard_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No shard_*.parquet files in {vtc_dir}")
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical")


def _hhmmss(seconds: float) -> str:
    """Format seconds as HH:MM:SS (for filenames / logs)."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


def _safe_stem(uid: str) -> str:
    """Shorten a uid to a filesystem-safe stem (last two underscore-parts)."""
    parts = uid.rsplit("_", 2)
    return "_".join(parts[-2:]) if len(parts) >= 3 else uid


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def print_summary(long: pl.DataFrame, threshold: float) -> None:
    n_total = len(long)
    n_files = long["uid"].n_unique()
    total_dur = long["duration"].sum()

    logger.info("=" * 60)
    logger.info(f"Long turns (> {threshold} s): {n_total} across {n_files} files")
    logger.info(f"Total duration of long turns: {total_dur / 3600:.2f} h")
    logger.info("")
    logger.info("Breakdown by speaker type:")
    for row in (
        long.group_by("label")
        .agg(
            pl.len().alias("count"),
            pl.col("duration").sum().alias("total_s"),
            pl.col("duration").mean().alias("mean_s"),
            pl.col("duration").max().alias("max_s"),
        )
        .sort("count", descending=True)
        .iter_rows(named=True)
    ):
        logger.info(
            f"  {row['label']:4s}  count={row['count']:5d}  "
            f"total={row['total_s']/3600:5.2f} h  "
            f"mean={row['mean_s']:.1f} s  max={row['max_s']:.1f} s"
        )
    logger.info("")
    logger.info("Duration buckets:")
    buckets = [(threshold, 30), (30, 60), (60, 120), (120, 300), (300, None)]
    for lo, hi in buckets:
        if hi is not None:
            n = long.filter((pl.col("duration") >= lo) & (pl.col("duration") < hi))
            label = f"{lo:.0f}–{hi:.0f} s"
        else:
            n = long.filter(pl.col("duration") >= lo)
            label = f"≥{lo:.0f} s"
        logger.info(f"  {label:10s}: {len(n):5d} turns")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _ordered_labels(long: pl.DataFrame) -> list[str]:
    """Return labels present in `long`, sorted by count descending."""
    counts = long.group_by("label").agg(pl.len().alias("n")).sort("n", descending=True)
    return [r for r in counts["label"].to_list() if r in set(LABELS)]


def plot_duration_histogram(
    long: pl.DataFrame, threshold: float, out_dir: Path
) -> None:
    """Overlaid per-label histogram of long-turn durations."""
    fig, ax = plt.subplots(figsize=(9, 5))
    cap = long["duration"].quantile(0.99)
    bins = np.linspace(threshold, cap, 60)  # type: ignore

    for label in _ordered_labels(long):
        sub = long.filter(pl.col("label") == label)["duration"].to_numpy()
        sub = sub[sub <= cap]
        n = len(long.filter(pl.col("label") == label))
        ax.hist(
            sub,
            bins=bins,
            alpha=0.6,
            color=LABEL_COLORS[label],
            label=f"{label} (n={n:,})",
        )

    ax.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"{threshold} s threshold",
    )
    ax.set_xlabel("Turn duration (s)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Long turns (> {threshold} s) — duration distribution\n"
        f"(capped at p99 = {cap:.0f} s)"
    )
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    path = out_dir / "duration_histogram.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_label_summary(long: pl.DataFrame, threshold: float, out_dir: Path) -> None:
    """Side-by-side bar chart: count and total duration per label."""
    stats = (
        long.group_by("label")
        .agg(
            pl.len().alias("count"),
            (pl.col("duration").sum() / 3600).alias("total_h"),
        )
        .sort("count", descending=True)
    )

    labels = stats["label"].to_list()
    counts = stats["count"].to_list()
    hours = stats["total_h"].to_list()
    colors = [LABEL_COLORS.get(lb, "#888888") for lb in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    bars1 = ax1.bar(labels, counts, color=colors)
    ax1.bar_label(bars1, fmt="%d", padding=3, fontsize=9)
    ax1.set_ylabel("Number of turns")
    ax1.set_title(f"Count of long turns (> {threshold} s)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    bars2 = ax2.bar(labels, hours, color=colors)
    ax2.bar_label(bars2, fmt="%.2f h", padding=3, fontsize=9)
    ax2.set_ylabel("Total duration (hours)")
    ax2.set_title(f"Total duration of long turns (> {threshold} s)")

    fig.suptitle("Long turns by speaker type", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "label_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_per_file(long: pl.DataFrame, threshold: float, out_dir: Path) -> None:
    """Stacked bar chart of long-turn counts per source file, coloured by label."""
    # short display name = last two underscore parts of uid
    long = long.with_columns(
        pl.col("uid").map_elements(_safe_stem, return_dtype=pl.String).alias("file")
    )

    files_ordered = (
        long.group_by("file")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)["file"]
        .to_list()
    )

    pivot = (
        long.group_by(["file", "label"])
        .agg(pl.len().alias("count"))
        .pivot(on="label", index="file", values="count", aggregate_function="sum")
        .fill_null(0)
    )
    # reorder rows
    pivot = pivot.with_columns(pl.col("file").cast(pl.Enum(files_ordered))).sort("file")

    present_labels = [lb for lb in LABELS if lb in pivot.columns]
    bottoms = np.zeros(len(pivot))
    fig, ax = plt.subplots(figsize=(max(10, len(files_ordered) * 0.4), 5))

    for lb in present_labels:
        vals = pivot[lb].to_numpy().astype(float)
        ax.bar(
            range(len(pivot)),
            vals,
            bottom=bottoms,
            color=LABEL_COLORS[lb],
            label=lb,
        )
        bottoms += vals

    ax.set_xticks(range(len(pivot)))
    ax.set_xticklabels(
        pivot["file"].cast(pl.String).to_list(), rotation=75, ha="right", fontsize=7
    )
    ax.set_ylabel("Number of long turns")
    ax.set_title(f"Long turns (> {threshold} s) per source file")
    ax.legend(title="Speaker")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    path = out_dir / "per_file.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_top_n_ladder(top: pl.DataFrame, out_dir: Path) -> None:
    """Horizontal bar chart of the top-N longest turns."""
    top = top.sort("duration", descending=True)
    labels_col = top["label"].to_list()
    durations = top["duration"].to_numpy()
    short_uid = [_safe_stem(u) for u in top["uid"].to_list()]
    onsets = top["onset"].to_list()
    yticks = [
        f"{lb} | {su} @ {_hhmmss(on)}"
        for lb, su, on in zip(labels_col, short_uid, onsets)
    ]
    colors = [LABEL_COLORS.get(lb, "#888888") for lb in labels_col]

    fig, ax = plt.subplots(figsize=(10, max(6, len(top) * 0.28)))
    ax.barh(range(len(top)), durations, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(yticks, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Turn duration (s)")
    ax.set_title(f"Top {len(top)} longest turns")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f} s"))
    # Add legend
    seen = set()
    for lb in labels_col:
        if lb not in seen:
            ax.barh([], [], color=LABEL_COLORS.get(lb, "#888888"), label=lb)
            seen.add(lb)
    ax.legend(title="Speaker", fontsize=8)
    fig.tight_layout()
    path = out_dir / "top_n_ladder.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------


def extract_audio(
    top: pl.DataFrame,
    uid_to_path: dict[str, str],
    out_dir: Path,
    pad: float,
) -> None:
    """Extract each turn in `top` from its source WAV file with padding."""
    out_dir.mkdir(parents=True, exist_ok=True)
    skipped = 0

    for rank, row in enumerate(
        top.sort("duration", descending=True).iter_rows(named=True), start=1
    ):
        uid = row["uid"]
        onset = row["onset"]
        offset = row["offset"]
        label = row["label"]
        duration = row["duration"]

        if uid not in uid_to_path:
            logger.warning("No audio path for uid %s — skipping", uid)
            skipped += 1
            continue

        src = Path(uid_to_path[uid])
        if not src.exists():
            logger.warning("Audio file not found: %s — skipping", src)
            skipped += 1
            continue

        # Read just the needed window
        info = sf.info(str(src))
        sr = info.samplerate
        total_frames = info.frames

        start_s = max(0.0, onset - pad)
        end_s = min(onset + duration + pad, total_frames / sr)
        start_frame = int(start_s * sr)
        end_frame = int(end_s * sr)
        n_frames = end_frame - start_frame

        audio, _ = sf.read(
            str(src), start=start_frame, frames=n_frames, dtype="float32"
        )

        stem = _safe_stem(uid)
        fname = f"{rank:03d}_{label}_{stem}_{_hhmmss(onset)}_{duration:.1f}s.wav"
        out_path = out_dir / fname
        sf.write(str(out_path), audio, sr)
        logger.info(
            "  [%3d/%d] %s  (%.1f s + %.1f s pad)", rank, len(top), fname, duration, pad
        )

    if skipped:
        logger.warning("%d turn(s) could not be extracted (missing audio).", skipped)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Analyse and extract anomalously long VTC turns."
    )
    parser.add_argument("dataset", help="Dataset name, e.g. seedlings_10")
    parser.add_argument(
        "--threshold",
        type=float,
        default=20.0,
        metavar="S",
        help="Minimum turn duration in seconds (default: 20)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        metavar="N",
        help="Number of longest turns to extract as audio (default: 50)",
    )
    parser.add_argument(
        "--pad",
        type=float,
        default=10.0,
        metavar="S",
        help="Seconds of audio context to add before/after each extracted turn (default: 10)",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip audio extraction (statistics + plots only)",
    )
    args = parser.parse_args(argv)

    paths = get_dataset_paths(args.dataset)
    vtc_dir = paths.output / "vtc_merged"
    fig_dir = paths.figures / "long_turns"
    fig_dir.mkdir(parents=True, exist_ok=True)
    audio_out = Path("data") / "long_turns"

    # ---- Load VTC segments ------------------------------------------------
    logger.info("Loading vtc_merged from %s", vtc_dir)
    df = _load_vtc_merged(vtc_dir)
    logger.info("  %d total segments across %d files", len(df), df["uid"].n_unique())

    # ---- Filter long turns ------------------------------------------------
    long = df.filter(pl.col("duration") >= args.threshold)
    logger.info(
        "  %d turns >= %.0f s (%d files)",
        len(long),
        args.threshold,
        long["uid"].n_unique(),
    )

    if len(long) == 0:
        logger.info("No long turns found. Exiting.")
        return

    # ---- Print summary ----------------------------------------------------
    print_summary(long, args.threshold)

    # ---- Save CSV log -----------------------------------------------------
    csv_path = paths.output / "long_turns.csv"
    long.sort(["uid", "onset"]).write_csv(str(csv_path))
    logger.info("Saved turn log to %s", csv_path)

    # ---- Plots ------------------------------------------------------------
    logger.info("Generating plots → %s", fig_dir)
    plot_duration_histogram(long, args.threshold, fig_dir)
    plot_label_summary(long, args.threshold, fig_dir)
    plot_per_file(long, args.threshold, fig_dir)

    top = long.sort("duration", descending=True).head(args.top_n)
    plot_top_n_ladder(top, fig_dir)

    # ---- Audio extraction -------------------------------------------------
    if args.no_audio:
        logger.info("--no-audio set; skipping extraction.")
        return

    manifest = pl.read_csv(str(paths.manifest))
    # uid in vtc is recording_id without the .wav extension
    manifest = manifest.with_columns(
        pl.col("recording_id").str.replace(r"\.wav$", "").alias("uid")
    )
    uid_to_path: dict[str, str] = dict(
        zip(manifest["uid"].to_list(), manifest["path"].to_list())
    )

    logger.info(
        "Extracting top-%d longest turns → %s  (%.0f s pad each side)",
        args.top_n,
        audio_out,
        args.pad,
    )
    extract_audio(top, uid_to_path, audio_out, args.pad)
    logger.info("Done.")


if __name__ == "__main__":
    main()
