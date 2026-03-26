#!/usr/bin/env python3
"""Compare VTC segments from full-file inference vs re-inference on clips.

Approach
--------
1. Convert clip-relative segments to absolute time using clip metadata.
2. **Coverage analysis**: for each label, compute hours where both systems
   agree, hours only detected by full-file VTC, and hours only detected
   by clip VTC.  Uses interval arithmetic (merge + intersect).
3. **Segment matching**: for each full-file segment, find the clip segment
   (same label, same source file) with highest temporal IoU.  Report match
   rates and onset/offset error distributions.

Requires:
    output/{dataset}/vtc_merged/*.parquet            full-file VTC (merged)
    output/{dataset}/vtc_clips/vtc_merged/*.parquet  clip VTC (merged, clip-relative)
    output/{dataset}/stats/clip_stats.parquet         clip boundaries

Usage (login node):
    uv run python -m src.analysis.vtc_clip_alignment seedlings_10
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl

from src.utils import get_dataset_paths

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("vtc_align")

LABELS = ["KCHI", "OCH", "MAL", "FEM"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_parquet_dir(path: Path) -> pl.DataFrame:
    """Concatenate all shard_*.parquet files in a directory."""
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {path}")
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical")


def _to_absolute(clip_df: pl.DataFrame, clip_stats: pl.DataFrame) -> pl.DataFrame:
    """Convert clip-relative segments to absolute time.

    Joins on ``clip_id`` to obtain the clip's absolute onset within the
    source recording, then shifts segment onsets/offsets accordingly.

    Returns a DataFrame with columns: ``uid, onset, offset, label``.
    """
    return (
        clip_df.join(
            clip_stats.select("clip_id", "uid", "abs_onset"),
            on="clip_id",
        )
        .with_columns(
            (pl.col("onset") + pl.col("abs_onset")).alias("onset"),
            (pl.col("offset") + pl.col("abs_onset")).alias("offset"),
        )
        .select("uid", "onset", "offset", "label")
    )


# ---------------------------------------------------------------------------
# Interval helpers
# ---------------------------------------------------------------------------


def _merge_intervals(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Merge overlapping or touching intervals.  Returns sorted, disjoint list."""
    if not intervals:
        return []
    s = sorted(intervals)
    merged: list[list[float]] = [[s[0][0], s[0][1]]]
    for on, off in s[1:]:
        if on <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], off)
        else:
            merged.append([on, off])
    return [(a, b) for a, b in merged]


def _interval_total(ivs: list[tuple[float, float]]) -> float:
    return sum(b - a for a, b in ivs)


def _interval_overlap(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> float:
    """Total overlap duration between two sorted, merged interval lists."""
    total = 0.0
    i = j = 0
    while i < len(a) and j < len(b):
        ov_start = max(a[i][0], b[j][0])
        ov_end = min(a[i][1], b[j][1])
        if ov_end > ov_start:
            total += ov_end - ov_start
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


# ---------------------------------------------------------------------------
# Coverage analysis (per-label interval overlap)
# ---------------------------------------------------------------------------


def _group_intervals(
    df: pl.DataFrame,
) -> dict[tuple[str, str], list[tuple[float, float]]]:
    """Group segments into ``{(uid, label): [(onset, offset), ...]}``."""
    groups: dict[tuple[str, str], list[tuple[float, float]]] = {}
    uids = df["uid"].to_list()
    onsets = df["onset"].to_list()
    offsets = df["offset"].to_list()
    labels = df["label"].to_list()
    for uid, on, off, lbl in zip(uids, onsets, offsets, labels):
        groups.setdefault((uid, lbl), []).append((on, off))
    return groups


def compute_coverage(
    full_df: pl.DataFrame,
    clip_abs_df: pl.DataFrame,
) -> dict[str, dict[str, float]]:
    """Per-label and all-speech coverage in hours: both, only_full, only_clip."""
    full_groups = _group_intervals(full_df)
    clip_groups = _group_intervals(clip_abs_df)
    all_uids = {uid for uid, _ in list(full_groups) + list(clip_groups)}

    result: dict[str, dict[str, float]] = {}

    for label in LABELS:
        both = only_full = only_clip = 0.0
        for uid in all_uids:
            f = _merge_intervals(full_groups.get((uid, label), []))
            c = _merge_intervals(clip_groups.get((uid, label), []))
            ov = _interval_overlap(f, c)
            both += ov
            only_full += _interval_total(f) - ov
            only_clip += _interval_total(c) - ov
        result[label] = {
            "both_h": both / 3600,
            "only_full_h": only_full / 3600,
            "only_clip_h": only_clip / 3600,
        }

    # All-speech: union across labels per uid
    both_all = only_full_all = only_clip_all = 0.0
    for uid in all_uids:
        full_all: list[tuple[float, float]] = []
        clip_all: list[tuple[float, float]] = []
        for label in LABELS:
            full_all.extend(full_groups.get((uid, label), []))
            clip_all.extend(clip_groups.get((uid, label), []))
        f = _merge_intervals(full_all)
        c = _merge_intervals(clip_all)
        ov = _interval_overlap(f, c)
        both_all += ov
        only_full_all += _interval_total(f) - ov
        only_clip_all += _interval_total(c) - ov
    result["all_speech"] = {
        "both_h": both_all / 3600,
        "only_full_h": only_full_all / 3600,
        "only_clip_h": only_clip_all / 3600,
    }

    return result


# ---------------------------------------------------------------------------
# Segment matching
# ---------------------------------------------------------------------------


def _iou(a_on: float, a_off: float, b_on: float, b_off: float) -> float:
    overlap = max(0.0, min(a_off, b_off) - max(a_on, b_on))
    union = max(a_off, b_off) - min(a_on, b_on)
    return overlap / union if union > 0 else 0.0


def compute_segment_matches(
    full_df: pl.DataFrame,
    clip_abs_df: pl.DataFrame,
    collar_s: float = 0.5,
) -> dict[str, dict]:
    """Match full-file segments → clip segments (same label, best IoU).

    For each full-file segment, find the clip segment with highest temporal
    IoU.  A match is declared if IoU > 0.  A *well-match* further requires
    both onset and offset to agree within ``collar_s``.

    Returns per-label dict with keys:
        n_full, n_clip, n_matched, n_well_matched,
        onset_errors, offset_errors, ious
    """

    def _build(df: pl.DataFrame) -> dict[tuple[str, str], list[tuple[float, float]]]:
        g: dict[tuple[str, str], list[tuple[float, float]]] = {}
        uids = df["uid"].to_list()
        onsets = df["onset"].to_list()
        offsets = df["offset"].to_list()
        labels = df["label"].to_list()
        for uid, on, off, lbl in zip(uids, onsets, offsets, labels):
            g.setdefault((uid, lbl), []).append((on, off))
        for v in g.values():
            v.sort()
        return g

    full_g = _build(full_df)
    clip_g = _build(clip_abs_df)

    results: dict[str, dict] = {}
    for label in LABELS:
        uids = {uid for uid, lbl in list(full_g) + list(clip_g) if lbl == label}
        onset_errs: list[float] = []
        offset_errs: list[float] = []
        ious_list: list[float] = []
        n_full = n_clip = n_matched = n_well = 0

        for uid in uids:
            f_segs = full_g.get((uid, label), [])
            c_segs = clip_g.get((uid, label), [])
            n_full += len(f_segs)
            n_clip += len(c_segs)

            c_start = 0  # sliding window for sorted clip segments
            for f_on, f_off in f_segs:
                # Advance past clip segments that ended well before this one
                while c_start < len(c_segs) and c_segs[c_start][1] < f_on - 5.0:
                    c_start += 1

                best_iou = 0.0
                best_c: tuple[float, float] | None = None
                j = c_start
                while j < len(c_segs):
                    c_on, c_off = c_segs[j]
                    if c_on > f_off + 5.0:
                        break
                    iou_val = _iou(f_on, f_off, c_on, c_off)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_c = (c_on, c_off)
                    j += 1

                if best_c is not None and best_iou > 0:
                    n_matched += 1
                    on_e = best_c[0] - f_on
                    off_e = best_c[1] - f_off
                    onset_errs.append(on_e)
                    offset_errs.append(off_e)
                    ious_list.append(best_iou)
                    if abs(on_e) <= collar_s and abs(off_e) <= collar_s:
                        n_well += 1

        results[label] = {
            "n_full": n_full,
            "n_clip": n_clip,
            "n_matched": n_matched,
            "n_well_matched": n_well,
            "onset_errors": np.array(onset_errs),
            "offset_errors": np.array(offset_errs),
            "ious": np.array(ious_list),
        }
    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


COLLARS = [0.5, 1.0, 2.0, 3.0, 5.0]


def _save_figures(
    coverage: dict[str, dict[str, float]],
    matches: dict[str, dict],
    totals: dict[str, dict[str, float]],
    fig_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)

    order = LABELS + ["all_speech"]
    display = ["KCHI", "OCH", "MAL", "FEM", "All speech"]

    # ------------------------------------------------------------------
    # Figure 1: Coverage breakdown + total comparison (2 panels)
    # ------------------------------------------------------------------
    fig, (ax, ax_tot) = plt.subplots(
        1, 2, figsize=(16, 4.5), gridspec_kw={"width_ratios": [3, 2]}
    )
    both_vals = [coverage[l]["both_h"] for l in order]
    only_full = [coverage[l]["only_full_h"] for l in order]
    only_clip = [coverage[l]["only_clip_h"] for l in order]

    y = np.arange(len(order))
    h = 0.6

    ax.barh(y, both_vals, h, label="Both agree", color="#2ecc71")
    ax.barh(y, only_full, h, left=both_vals, label="Only full-file", color="#e67e22")
    ax.barh(
        y,
        only_clip,
        h,
        left=[b + f for b, f in zip(both_vals, only_full)],
        label="Only clip",
        color="#9b59b6",
    )

    max_total = max(b + f + c for b, f, c in zip(both_vals, only_full, only_clip))
    for i in range(len(order)):
        total = both_vals[i] + only_full[i] + only_clip[i]
        ax.text(
            total + max_total * 0.01,
            i,
            f"{both_vals[i]:.0f} + {only_full[i]:.0f} + {only_clip[i]:.0f} h",
            va="center",
            fontsize=8,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(display, fontsize=11)
    ax.set_xlabel("Hours", fontsize=11)
    ax.set_xlim(0, max_total * 1.35)
    ax.set_title(
        "Coverage Agreement: Full-file VTC vs Clip VTC",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.invert_yaxis()

    # Panel b: total speech per label
    tot_full = [totals[l]["full_h"] for l in order]
    tot_clip = [totals[l]["clip_h"] for l in order]
    y2 = np.arange(len(order))
    h2 = 0.3
    ax_tot.barh(y2 - h2 / 2, tot_full, h2, label="Full-file", color="#3498db")
    ax_tot.barh(y2 + h2 / 2, tot_clip, h2, label="Clip", color="#e74c3c")
    max_tot = max(max(tot_full), max(tot_clip))
    for i in range(len(order)):
        diff_pct = (
            100 * (tot_clip[i] - tot_full[i]) / tot_full[i] if tot_full[i] > 0 else 0
        )
        ax_tot.text(
            max(tot_full[i], tot_clip[i]) + max_tot * 0.01,
            i,
            f"{tot_full[i]:.1f} vs {tot_clip[i]:.1f}h ({diff_pct:+.2f}%)",
            va="center",
            fontsize=8,
        )
    ax_tot.set_yticks(y2)
    ax_tot.set_yticklabels(display, fontsize=11)
    ax_tot.set_xlabel("Hours", fontsize=11)
    ax_tot.set_xlim(0, max_tot * 1.55)
    ax_tot.set_title("Total Speech per Label", fontsize=11, fontweight="bold")
    ax_tot.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax_tot.invert_yaxis()

    fig.suptitle("VTC Clip Alignment \u2014 Coverage", fontsize=12, fontweight="bold")
    fig.tight_layout()

    out = fig_dir / "clip_alignment_coverage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure -> {out}")

    # ------------------------------------------------------------------
    # Figure 2: Segment matching  (3-panel)
    #   a) multi-collar match rate per label
    #   b) boundary-error CDF
    #   c) onset / offset box plots
    # ------------------------------------------------------------------
    match_labels = LABELS
    match_display = ["KCHI", "OCH", "MAL", "FEM"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5))

    # Panel a: multi-collar match rates
    x = np.arange(len(COLLARS))
    w = 0.18
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    for k, (lbl, disp) in enumerate(zip(match_labels, match_display)):
        m = matches[lbl]
        n = m["n_full"]
        pcts = []
        for c in COLLARS:
            n_well = int(
                np.sum(
                    (np.abs(m["onset_errors"]) <= c) & (np.abs(m["offset_errors"]) <= c)
                )
            )
            pcts.append(100 * n_well / n if n > 0 else 0)
        offset = (k - 1.5) * w
        bars = ax1.bar(x + offset, pcts, w, label=disp, color=colors[k], alpha=0.85)
        for bar, pct in zip(bars, pcts):
            if pct > 5:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{pct:.0f}",
                    ha="center",
                    fontsize=6,
                )

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"\u00b1{c}s" for c in COLLARS], fontsize=10)
    ax1.set_ylabel("% of full-file segments", fontsize=10)
    ax1.set_ylim(0, 105)
    ax1.set_title("Match Rate by Collar", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Panel b: CDF of absolute boundary errors
    for k, (lbl, disp) in enumerate(zip(match_labels, match_display)):
        m = matches[lbl]
        if len(m["onset_errors"]) == 0:
            continue
        abs_errs = np.sort(
            np.maximum(np.abs(m["onset_errors"]), np.abs(m["offset_errors"]))
        )
        cdf = np.arange(1, len(abs_errs) + 1) / m["n_full"]
        ax2.plot(abs_errs, 100 * cdf, color=colors[k], label=disp, linewidth=1.5)

    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("Max boundary error (s)", fontsize=10)
    ax2.set_ylabel("% of full-file segments", fontsize=10)
    ax2.set_title("Cumulative Match Rate", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.axvline(0.5, color="gray", linewidth=0.8, linestyle=":")
    ax2.axvline(2.0, color="gray", linewidth=0.8, linestyle=":")

    # Panel c: onset / offset error box plots
    bp_data_on = []
    bp_data_off = []
    bp_labels = []
    bp_display = []
    for lbl, disp in zip(match_labels, match_display):
        if len(matches[lbl]["onset_errors"]) > 0:
            bp_data_on.append(matches[lbl]["onset_errors"])
            bp_data_off.append(matches[lbl]["offset_errors"])
            bp_labels.append(lbl)
            bp_display.append(disp)

    if bp_data_on:
        positions_on = np.arange(len(bp_labels)) * 2
        positions_off = positions_on + 0.6

        bp1 = ax3.boxplot(
            bp_data_on,
            positions=positions_on,
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor="#3498db", alpha=0.6),
            medianprops=dict(color="black"),
        )
        bp2 = ax3.boxplot(
            bp_data_off,
            positions=positions_off,
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor="#e67e22", alpha=0.6),
            medianprops=dict(color="black"),
        )

        ax3.set_xticks((positions_on + positions_off) / 2)
        ax3.set_xticklabels(bp_display, fontsize=11)
        ax3.set_ylabel("Error (s): clip \u2212 full-file", fontsize=10)
        ax3.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax3.set_title("Boundary Errors (matched)", fontsize=11, fontweight="bold")
        ax3.legend(
            [bp1["boxes"][0], bp2["boxes"][0]],
            ["Onset error", "Offset error"],
            fontsize=8,
            loc="upper right",
            framealpha=0.9,
        )
        ax3.grid(axis="y", alpha=0.3, linestyle="--")
    else:
        ax3.text(0.5, 0.5, "No matched segments", ha="center", va="center")

    fig.suptitle(
        "VTC Clip Alignment \u2014 Segment Matching", fontsize=12, fontweight="bold"
    )
    fig.tight_layout()

    out2 = fig_dir / "clip_alignment_matching.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure -> {out2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(dataset: str) -> None:
    paths = get_dataset_paths(dataset)
    orig_dir = paths.output / "vtc_merged"
    clip_dir = paths.output / "vtc_clips" / "vtc_merged"
    clip_stats_path = paths.output / "stats" / "clip_stats.parquet"

    for p in [orig_dir, clip_dir, clip_stats_path]:
        if not p.exists():
            logger.error(f"Missing: {p}")
            sys.exit(1)

    logger.info(f"Dataset : {dataset}")

    # Load
    logger.info("Loading segments...")
    full_df = _load_parquet_dir(orig_dir)
    clip_df = _load_parquet_dir(clip_dir)
    clip_stats = pl.read_parquet(clip_stats_path)

    logger.info(f"  Full-file segments : {len(full_df):,}")
    logger.info(f"  Clip segments      : {len(clip_df):,}")
    logger.info(f"  Clips              : {len(clip_stats):,}")

    # Convert clip segments to absolute time
    logger.info("Converting clip segments to absolute coordinates...")
    clip_abs = _to_absolute(clip_df, clip_stats)
    logger.info(f"  Clip segments (absolute) : {len(clip_abs):,}")

    # Coverage analysis
    logger.info("Computing coverage overlap per label...")
    coverage = compute_coverage(full_df, clip_abs)

    # Segment matching
    logger.info("Matching segments...")
    matches = compute_segment_matches(full_df, clip_abs, collar_s=max(COLLARS))

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print(f"VTC Clip Alignment — {dataset}")
    print("=" * 72)

    print("\n── Coverage (hours) ──")
    print(
        f"  {'Label':<14} {'Both':>8} {'Only full':>10} {'Only clip':>10} {'Agree%':>8}"
    )
    print("  " + "-" * 54)
    for lbl in LABELS + ["all_speech"]:
        c = coverage[lbl]
        total = c["both_h"] + c["only_full_h"] + c["only_clip_h"]
        agree_pct = 100 * c["both_h"] / total if total > 0 else 0
        print(
            f"  {lbl:<14} {c['both_h']:>8.1f} {c['only_full_h']:>10.1f}"
            f" {c['only_clip_h']:>10.1f} {agree_pct:>7.1f}%"
        )

    print(f"\n── Segment matching (multi-collar) ──")
    header = f"  {'Label':<8} {'N_full':>8} {'Matched':>8}"
    for c in COLLARS:
        header += f" {'±' + str(c) + 's':>8}"
    print(header)
    print("  " + "-" * (32 + 9 * len(COLLARS)))
    for lbl in LABELS:
        m = matches[lbl]
        n = m["n_full"]
        line = f"  {lbl:<8} {n:>8,} {m['n_matched']:>8,}"
        for c in COLLARS:
            nw = int(
                np.sum(
                    (np.abs(m["onset_errors"]) <= c) & (np.abs(m["offset_errors"]) <= c)
                )
            )
            pct = 100 * nw / n if n > 0 else 0
            line += f" {pct:>7.1f}%"
        print(line)

    print("\n── Boundary errors (matched segments, seconds) ──")
    print(
        f"  {'Label':<10} {'Onset med':>10} {'Onset IQR':>10}"
        f" {'Offset med':>11} {'Offset IQR':>11}"
    )
    print("  " + "-" * 56)
    for lbl in LABELS:
        m = matches[lbl]
        if len(m["onset_errors"]) > 0:
            on_med = np.median(m["onset_errors"])
            on_iqr = np.percentile(m["onset_errors"], 75) - np.percentile(
                m["onset_errors"], 25
            )
            off_med = np.median(m["offset_errors"])
            off_iqr = np.percentile(m["offset_errors"], 75) - np.percentile(
                m["offset_errors"], 25
            )
            print(
                f"  {lbl:<10} {on_med:>+10.3f} {on_iqr:>10.3f}"
                f" {off_med:>+11.3f} {off_iqr:>11.3f}"
            )
        else:
            print(f"  {lbl:<10} {'n/a':>10} {'n/a':>10} {'n/a':>11} {'n/a':>11}")

    # ------------------------------------------------------------------
    # Save CSVs
    # ------------------------------------------------------------------
    out_dir = paths.output

    # Coverage
    cov_rows = []
    for lbl in LABELS + ["all_speech"]:
        c = coverage[lbl]
        total = c["both_h"] + c["only_full_h"] + c["only_clip_h"]
        cov_rows.append(
            {
                "label": lbl,
                "both_h": round(c["both_h"], 2),
                "only_full_h": round(c["only_full_h"], 2),
                "only_clip_h": round(c["only_clip_h"], 2),
                "agreement_pct": (
                    round(100 * c["both_h"] / total, 2) if total > 0 else 0
                ),
            }
        )
    cov_csv = out_dir / "vtc_clip_alignment_coverage.csv"
    pl.DataFrame(cov_rows).write_csv(cov_csv)
    logger.info(f"Saved -> {cov_csv}")

    # Matches (multi-collar)
    match_rows = []
    for lbl in LABELS:
        m = matches[lbl]
        row = {
            "label": lbl,
            "n_full": m["n_full"],
            "n_clip": m["n_clip"],
            "n_matched": m["n_matched"],
            "match_pct": (
                round(100 * m["n_matched"] / m["n_full"], 2) if m["n_full"] > 0 else 0
            ),
        }
        for c in COLLARS:
            nw = int(
                np.sum(
                    (np.abs(m["onset_errors"]) <= c) & (np.abs(m["offset_errors"]) <= c)
                )
            )
            row[f"within_{c}s_pct"] = (
                round(100 * nw / m["n_full"], 2) if m["n_full"] > 0 else 0
            )
        row["onset_median_s"] = (
            round(float(np.median(m["onset_errors"])), 4)
            if len(m["onset_errors"]) > 0
            else None
        )
        row["offset_median_s"] = (
            round(float(np.median(m["offset_errors"])), 4)
            if len(m["offset_errors"]) > 0
            else None
        )
        match_rows.append(row)
    match_csv = out_dir / "vtc_clip_alignment_matches.csv"
    pl.DataFrame(match_rows).write_csv(match_csv)
    logger.info(f"Saved -> {match_csv}")

    # Compute totals per label
    totals: dict[str, dict[str, float]] = {}
    for lbl in LABELS:
        full_h = (
            float(
                full_df.filter(pl.col("label") == lbl)
                .select((pl.col("offset") - pl.col("onset")).sum())
                .item()
            )
            / 3600
        )
        clip_h = (
            float(
                clip_df.filter(pl.col("label") == lbl)
                .select((pl.col("offset") - pl.col("onset")).sum())
                .item()
            )
            / 3600
        )
        totals[lbl] = {"full_h": full_h, "clip_h": clip_h}
    totals["all_speech"] = {
        "full_h": sum(t["full_h"] for t in totals.values()),
        "clip_h": sum(t["clip_h"] for t in totals.values()),
    }

    # Print totals
    print("\n── Total speech per label (hours) ──")
    print(f"  {'Label':<14} {'Full-file':>10} {'Clip':>10} {'Diff%':>8}")
    print("  " + "-" * 44)
    for lbl in LABELS + ["all_speech"]:
        t = totals[lbl]
        diff_pct = (
            100 * (t["clip_h"] - t["full_h"]) / t["full_h"] if t["full_h"] > 0 else 0
        )
        print(
            f"  {lbl:<14} {t['full_h']:>10.2f} {t['clip_h']:>10.2f} {diff_pct:>+7.2f}%"
        )

    # Figures
    fig_dir = Path("figures") / dataset / "vtc"
    _save_figures(coverage, matches, totals, fig_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python -m src.analysis.vtc_clip_alignment <dataset>",
            file=sys.stderr,
        )
        sys.exit(1)
    main(sys.argv[1])
