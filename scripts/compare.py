#!/usr/bin/env python3
"""
Compare VAD and VTC pipeline outputs.

Loads segments from ``output/{dataset}/`` (vad_raw, vtc_raw, vad_merged,
vtc_merged), computes per-file IoU / Precision / Recall, generates matplotlib
figures, and merges VTC per-file metadata into the VAD metadata file.

Usage:
    python scripts/compare.py chunks30
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from scripts.utils import atomic_write_parquet, get_dataset_paths


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_parquet_dir(directory: Path) -> pl.DataFrame:
    """Read all .parquet files in *directory* into one DataFrame."""
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {directory}")
    return pl.read_parquet(files)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def flatten_intervals(
    df: pl.LazyFrame,
    partition_cols: list[str] | None = None,
) -> pl.LazyFrame:
    """Merge overlapping intervals within each partition.

    Expects columns ``uid, onset, offset``.
    Returns ``uid, onset, offset, duration`` (+ any partition cols).
    """
    if partition_cols is None:
        partition_cols = ["uid"]

    return (
        df.sort(*partition_cols, "onset")
        .with_columns(
            _prev_max_off=pl.col("offset")
            .shift(1)
            .cum_max()
            .over(*partition_cols)
            .fill_null(0.0)
        )
        .with_columns(
            _new_grp=(pl.col("onset") > pl.col("_prev_max_off")).cast(pl.Int32)
        )
        .with_columns(_grp_id=pl.col("_new_grp").cum_sum().over(*partition_cols))
        .group_by(*partition_cols, "_grp_id")
        .agg(
            onset=pl.col("onset").min(),
            offset=pl.col("offset").max(),
        )
        .with_columns(duration=(pl.col("offset") - pl.col("onset")).round(3))
        .drop("_grp_id")
    )


def calculate_metrics(
    df_vad: pl.DataFrame,
    df_vtc: pl.DataFrame,
    vad_meta: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Per-file IoU, Precision, Recall between a VAD and a VTC DataFrame.

    Both inputs must contain ``uid, onset, offset, duration``.
    *vad_meta* (optional) may supply ``file_total`` (audio duration) per uid
    for TN calculation.
    """
    vad_flat = flatten_intervals(df_vad.lazy())
    vtc_flat = flatten_intervals(df_vtc.lazy())

    # Total speech duration per file
    vad_stats = vad_flat.group_by("uid").agg(
        pl.col("duration").sum().alias("vad_dur"),
    )
    vtc_stats = vtc_flat.group_by("uid").agg(
        pl.col("duration").sum().alias("vtc_dur"),
    )

    # Intersection (true positive seconds)
    intersection = (
        vad_flat.join(vtc_flat, on="uid", suffix="_vtc")
        .filter(
            (pl.col("onset") < pl.col("offset_vtc"))
            & (pl.col("offset") > pl.col("onset_vtc"))
        )
        .select(
            "uid",
            pl.max_horizontal("onset", "onset_vtc").alias("s"),
            pl.min_horizontal("offset", "offset_vtc").alias("e"),
        )
        .group_by("uid")
        .agg((pl.col("e") - pl.col("s")).sum().alias("TP"))
    )

    results = (
        vad_stats.join(vtc_stats, on="uid", how="full", coalesce=True)
        .join(intersection, on="uid", how="left")
        .fill_null(0)
        .with_columns(
            FP=pl.col("vtc_dur") - pl.col("TP"),
            FN=pl.col("vad_dur") - pl.col("TP"),
        )
        .with_columns(
            IoU=(pl.col("TP") / (pl.col("TP") + pl.col("FP") + pl.col("FN"))),
            Precision=(pl.col("TP") / (pl.col("TP") + pl.col("FP"))),
            Recall=(pl.col("TP") / (pl.col("TP") + pl.col("FN"))),
        )
        .collect()
    )

    # Attach per-file total duration if provided
    if vad_meta is not None and "file_total" in vad_meta.columns:
        results = results.join(
            vad_meta.select("uid", "file_total"), on="uid", how="left"
        )

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(results: pl.DataFrame, label: str, low_thresh: float = 0.5) -> dict:
    """Print executive summary and return global aggregates."""
    tot = results.select(
        pl.sum("TP"),
        pl.sum("FP"),
        pl.sum("FN"),
        pl.sum("vad_dur"),
        pl.sum("vtc_dur"),
    ).to_dicts()[0]

    g_iou = (
        tot["TP"] / (tot["TP"] + tot["FP"] + tot["FN"])
        if (tot["TP"] + tot["FP"] + tot["FN"])
        else 0
    )
    g_prec = tot["TP"] / (tot["TP"] + tot["FP"]) if (tot["TP"] + tot["FP"]) else 0
    g_rec = tot["TP"] / (tot["TP"] + tot["FN"]) if (tot["TP"] + tot["FN"]) else 0

    n_low = results.filter(pl.col("IoU") < low_thresh).height
    n_high = results.filter(pl.col("IoU") >= low_thresh).height

    print("=" * 60)
    print(f"  {label}")
    print("=" * 60)
    print(f"  Global IoU:       {g_iou:.2%}")
    print(f"  Precision:        {g_prec:.2%}")
    print(f"  Recall:           {g_rec:.2%}")
    print(f"  VAD speech:       {tot['vad_dur'] / 3600:.1f} h")
    print(f"  VTC speech:       {tot['vtc_dur'] / 3600:.1f} h")
    print(f"  High IoU (≥{low_thresh:.0%}): {n_high:>6d} ({n_high / len(results):.1%})")
    print(f"  Low  IoU (<{low_thresh:.0%}): {n_low:>6d} ({n_low / len(results):.1%})")
    print("=" * 60)

    return {
        "vad_h": tot["vad_dur"] / 3600,
        "vtc_h": tot["vtc_dur"] / 3600,
        "iou": g_iou,
        "precision": g_prec,
        "recall": g_rec,
    }


# ---------------------------------------------------------------------------
# Matplotlib dashboard
# ---------------------------------------------------------------------------


def plot_dashboard(
    results: pl.DataFrame,
    global_stats: dict,
    title: str,
    output_path: Path,
    low_thresh: float = 0.5,
) -> None:
    """Generate and save a 2×3 matplotlib dashboard."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    df_low = results.filter(pl.col("IoU") < low_thresh)
    df_high = results.filter(pl.col("IoU") >= low_thresh)

    # 1. Volume bar chart
    ax = axes[0, 0]
    ax.bar(
        ["VAD", "VTC"],
        [global_stats["vad_h"], global_stats["vtc_h"]],
        color=["#3498db", "#e74c3c"],
    )
    for i, v in enumerate([global_stats["vad_h"], global_stats["vtc_h"]]):
        ax.text(i, v + 0.02 * v, f"{v:.1f}h", ha="center", fontsize=9)
    ax.set_title("Volume (hours)")
    ax.set_ylabel("Hours")

    # 2. Precision vs Recall scatter
    ax = axes[0, 1]
    ax.scatter(
        results["Recall"].to_numpy(),
        results["Precision"].to_numpy(),
        alpha=0.3,
        s=8,
        color="#AB63FA",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs Recall")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # 3. IoU histogram
    ax = axes[0, 2]
    iou_vals = results["IoU"].drop_nulls().drop_nans().to_numpy()
    ax.hist(iou_vals, bins=50, color="#636EFA", edgecolor="white", linewidth=0.3)
    ax.axvline(
        low_thresh,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"thresh={low_thresh}",
    )
    ax.set_xlabel("IoU")
    ax.set_title("IoU Distribution")
    ax.legend(fontsize=8)

    # 4. Duration split – high vs low IoU
    ax = axes[1, 0]
    if "file_total" in results.columns:
        p99 = results["file_total"].quantile(0.99)
        for grp, name, color in [
            (df_high, "High IoU", "#1f77b4"),
            (df_low, "Low IoU", "#d62728"),
        ]:
            if "file_total" in grp.columns and grp.height > 0:
                vals = grp.filter(pl.col("file_total") < p99)["file_total"].to_numpy()
                ax.hist(vals, bins=40, alpha=0.6, label=name, color=color, density=True)
        ax.set_xlabel(f"Duration (0–{p99:.0f}s)")
        ax.legend(fontsize=8)
    ax.set_title("Duration (High vs Low IoU)")

    # 5. VAD speech ratio (vad_dur / file_total)
    ax = axes[1, 1]
    if "file_total" in results.columns:
        for grp, name, color in [
            (df_high, "High IoU", "#1f77b4"),
            (df_low, "Low IoU", "#d62728"),
        ]:
            if grp.height > 0 and "file_total" in grp.columns:
                ratio = (grp["vad_dur"] / grp["file_total"]).to_numpy()
                ax.hist(
                    ratio, bins=40, alpha=0.6, label=name, color=color, density=True
                )
        ax.set_xlabel("VAD speech ratio")
        ax.legend(fontsize=8)
    ax.set_title("Speech Ratio (High vs Low IoU)")

    # 6. VTC duration vs IoU scatter
    ax = axes[1, 2]
    ax.scatter(
        results["vtc_dur"].to_numpy(),
        results["IoU"].drop_nulls().drop_nans().to_numpy(),
        alpha=0.25,
        s=4,
        color="gray",
    )
    ax.set_xlabel("VTC speech (s)")
    ax.set_ylabel("IoU")
    ax.set_title("VTC Duration vs IoU")

    fig.tight_layout(rect=[0, 0, 1, 0.95])  # type: ignore
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {output_path}")


# ---------------------------------------------------------------------------
# Low-IoU diagnostics & file export
# ---------------------------------------------------------------------------


def diagnose_low_iou(
    results: pl.DataFrame,
    vtc_df: pl.DataFrame,
    low_thresh: float = 0.5,
) -> pl.DataFrame:
    """Categorize low-IoU files and print a summary.

    Categories:
      - vtc_silent:       VTC produced zero segments
      - vtc_under_detect: VTC speech < 50 % of VAD speech
      - vtc_over_detect:  VTC speech > 200 % of VAD speech
      - moderate_mismatch: everything else
    """
    low = results.filter(pl.col("IoU") < low_thresh)

    # Count VTC segments per uid
    vtc_seg_counts = vtc_df.group_by("uid").agg(pl.len().alias("n_vtc_segs"))

    diag = (
        low.join(vtc_seg_counts, on="uid", how="left")
        .fill_null(0)
        .with_columns(
            category=pl.when(pl.col("n_vtc_segs") == 0)
            .then(pl.lit("vtc_silent"))
            .when(pl.col("vtc_dur") < 0.5 * pl.col("vad_dur"))
            .then(pl.lit("vtc_under_detect"))
            .when(pl.col("vtc_dur") > 2.0 * pl.col("vad_dur"))
            .then(pl.lit("vtc_over_detect"))
            .otherwise(pl.lit("moderate_mismatch"))
        )
    )

    print(f"\n  Low-IoU diagnostics (IoU < {low_thresh:.0%}, n={diag.height})")
    print("  " + "-" * 40)
    for row in (
        diag.group_by("category")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .iter_rows(named=True)
    ):
        print(f"    {row['category']:<22s} {row['n']:>6d}")

    return diag



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare VAD and VTC pipeline outputs."
    )
    parser.add_argument(
        "dataset",
        help="Dataset name (derives manifests/{name}.parquet, output/{name}/, etc.)",
    )
    parser.add_argument(
        "--low-iou-thresh",
        type=float,
        default=0.3,
        help="IoU threshold below which files are flagged as failures.",
    )
    args = parser.parse_args()

    ds = get_dataset_paths(args.dataset)
    low_thresh = args.low_iou_thresh

    print(f"Dataset:  {args.dataset}")
    print(f"Output:   {ds.output}")
    print(f"Figures:  {ds.figures}")

    # ---- Load the four segment datasets ----
    print("\nLoading data...")
    vad_raw = load_parquet_dir(ds.output / "vad_raw")
    vad_merged = load_parquet_dir(ds.output / "vad_merged")
    vtc_raw = load_parquet_dir(ds.output / "vtc_raw")
    vtc_merged = load_parquet_dir(ds.output / "vtc_merged")

    print(f"  vad_raw:    {vad_raw.height:>8,d} segments")
    print(f"  vad_merged: {vad_merged.height:>8,d} segments")
    print(f"  vtc_raw:    {vtc_raw.height:>8,d} segments")
    print(f"  vtc_merged: {vtc_merged.height:>8,d} segments")

    # ---- Build per-file metadata (total audio duration) from VAD metadata ----
    vad_meta = None
    meta_path = ds.metadata / "metadata.parquet"
    if meta_path.exists():
        vad_meta_full = pl.read_parquet(meta_path)
        vad_meta = vad_meta_full.select(
            pl.col("file_id").str.replace(r"\.wav$", "").alias("uid"),
            pl.col("duration").alias("file_total"),
        )

    # ---- Compute metrics: raw ----
    print("\n--- RAW comparison (VAD raw vs VTC raw) ---")
    results_raw = calculate_metrics(vad_raw, vtc_raw, vad_meta)
    stats_raw = print_summary(results_raw, "RAW: VAD vs VTC", low_thresh)
    results_raw.write_csv(ds.output / "compare_raw.csv")

    # ---- Compute metrics: merged ----
    print("\n--- MERGED comparison (VAD merged vs VTC merged) ---")
    results_merged = calculate_metrics(vad_merged, vtc_merged, vad_meta)
    stats_merged = print_summary(results_merged, "MERGED: VAD vs VTC", low_thresh)
    results_merged.write_csv(ds.output / "compare_merged.csv")

    # ---- Plots ----
    if vad_meta is not None:
        results_raw = results_raw.join(
            vad_meta.select("uid", "file_total"), on="uid", how="left"
        )
        results_merged = results_merged.join(
            vad_meta.select("uid", "file_total"), on="uid", how="left"
        )

    plot_dashboard(
        results_raw,
        stats_raw,
        "RAW: VAD vs VTC",
        ds.figures / "compare_raw.png",
        low_thresh,
    )
    plot_dashboard(
        results_merged,
        stats_merged,
        "MERGED: VAD vs VTC",
        ds.figures / "compare_merged.png",
        low_thresh,
    )

    # ---- Diagnostics on merged results ----
    diag = diagnose_low_iou(results_merged, vtc_merged, low_thresh)
    diag.write_csv(ds.output / "diagnostics.csv")

    # ---- Merge VTC metadata into VAD metadata file ----
    vtc_meta_dir = ds.output / "vtc_meta"
    if vtc_meta_dir.exists() and meta_path.exists():
        vtc_meta_files = sorted(vtc_meta_dir.glob("*.parquet"))
        if vtc_meta_files:
            vtc_meta_df = pl.read_parquet(vtc_meta_files)
            vad_meta_full = pl.read_parquet(meta_path)

            # Prepare join key: VAD uses file_id, VTC uses uid (no .wav)
            vtc_cols = vtc_meta_df.with_columns(
                pl.col("uid").alias("_join_key")
            )
            if vad_meta_full["file_id"].str.ends_with(".wav").any():
                vtc_cols = vtc_cols.with_columns(
                    (pl.col("_join_key") + ".wav").alias("_join_key")
                )

            # Drop existing VTC columns before merge (idempotent re-runs)
            vtc_col_names = [c for c in vtc_meta_df.columns if c != "uid"]
            for col in vtc_col_names:
                if col in vad_meta_full.columns:
                    vad_meta_full = vad_meta_full.drop(col)

            updated = vad_meta_full.join(
                vtc_cols.drop("uid"),
                left_on="file_id",
                right_on="_join_key",
                how="left",
            )
            atomic_write_parquet(updated, meta_path)
            n_matched = updated.filter(
                pl.col("vtc_threshold").is_not_nan()
            ).height
            print(f"\nUpdated metadata: {n_matched}/{updated.height} "
                  f"rows with VTC columns → {meta_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
