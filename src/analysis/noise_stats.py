#!/usr/bin/env python3
"""
Fine-grained ESC statistics from PANNs CNN14 NPZ outputs.

Computes per-class and per-category metrics across all audio files in a
dataset by reading the pre-computed ``audioset_probs`` arrays (N_bins × 527)
stored in each NPZ file.  No GPU / model loading required.

Three cumulative metrics are computed for **every** AudioSet class and for
every coarse category under both the original (buggy) category map and the
redesigned map:

  1. ``pw_seconds``  — probability-weighted seconds:
                       Σ(prob × pool_step_s) over all 1 s windows.
                       Answers "how many effective seconds was this present?"

  2. ``mean_prob``   — mean probability across **all** 1 s windows in the
                       dataset.  Normalised to recording length.

  3. ``hit_rate``    — fraction of 1 s windows where this AudioSet class
                       appears among the top-10 non-speech classes (by
                       probability).  For coarse categories: fraction of
                       windows where the category max-probability > 0.05.

For coarse categories the "probability" per window is the *maximum* over all
member class probabilities (consistent with esc.py's ``map_to_categories``).

Outputs saved to  output/{dataset}/esc_stats/ :
  audioset_stats.parquet     per AudioSet class  (527 rows × metrics)
  category_stats.parquet     original + redesigned side-by-side per category
  per_file_summary.parquet   per recording (dominant cats, top-10 classes …)
  summary.json               scalar dataset-level numbers

Usage:
    python -m src.analysis.esc_stats seedlings_10
    python -m src.analysis.esc_stats seedlings_10 --active_threshold 0.1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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
logger = logging.getLogger("esc_stats")

# ─────────────────────────────────────────────────────────────────────────────
# Speech indices (excluded from all non-speech metrics)
# ─────────────────────────────────────────────────────────────────────────────
_SPEECH_INDICES: frozenset[int] = frozenset(range(16))  # 0–15

# ─────────────────────────────────────────────────────────────────────────────
# ORIGINAL category map (from initial pipeline — kept for comparison)
# Note the bugs in the original: tv_radio maps to {Mains hum, Distortion …},
# water maps to {Coin, Scissors, Typing …}, household includes aircraft (340–343),
# silence maps to {Plop, Jingle, Hum} instead of actual Silence (500).
# ─────────────────────────────────────────────────────────────────────────────
_CAT_ORIGINAL: dict[str, set[int]] = {
    "music": set(range(137, 285)),
    "crying": {22, 23, 24, 25},
    "laughter": {16, 17, 18, 19, 20, 21},
    "singing": {27, 28, 29, 30, 31, 32},
    "tv_radio": {516, 517, 518, 519, 520},  # WRONG – Mains hum, Distortion …
    "vehicle": set(range(300, 320)),
    "animal": set(range(73, 105)),
    "water": set(range(380, 395)),  # WRONG – Coin, Scissors, Typing …
    "household": set(range(340, 370)),  # includes aircraft (340–343)
    "impact": set(range(395, 420)),
    "alarm": set(range(395, 400)) | {489, 490, 491, 492},
    "silence": {494, 495, 496},  # WRONG – Plop, Jingle, Hum
}

# ─────────────────────────────────────────────────────────────────────────────
# REDESIGNED category map — covers all 511 non-speech AudioSet classes,
# so the residual "other" bucket is 0.
# ─────────────────────────────────────────────────────────────────────────────
_CAT_NEW: dict[str, set[int]] = {
    # Vocalizations
    "laughter": set(range(16, 22)),  # 16–21  Laughter variants
    "crying": set(range(22, 27)),  # 22–26  Crying, baby cry, whimper, wail, sigh
    "singing": set(range(27, 37)),  # 27–36  Singing, child singing, rapping
    # Human activity / body sounds
    "human_activity": set(range(37, 72)),  # 37–71  Humming, breath, cough, footsteps,
    #                                         #        clapping, cheering, children playing …
    # Animals
    "animal": set(range(72, 137)),  # 72–136 All animal sounds (pets → whale)
    # Music & instruments
    "music": set(range(137, 283)),  # 137–282 Music genres + instruments
    # Natural / outdoor sounds
    "nature": set(range(283, 300)),  # 283–299 Wind, rain, thunder, ocean, fire
    # Transport
    "vehicle": set(range(300, 344)),  # 300–343 Boats, cars, trains, aircraft, engine
    # Machinery & tools
    "machinery": set(range(344, 354)) | set(range(404, 426)),
    #                                         # 344–353 Chainsaw, drill, engine variants
    #                                         # 404–425 Mechanisms, clock, camera, drill
    # Domestic / household
    "household": set(range(354, 388)),  # 354–387 Doors, kitchen, taps, vacuum …
    # Alarms, signals & electronic beeps
    "alarm_signal": set(range(388, 404)) | set(range(481, 500)),
    #                                         # 388–403 Alarm, phone, siren, whistle
    #                                         # 481–499 Beep, ping, clang, rumble …
    # Impacts, crashes, explosions
    "impact": set(range(426, 481)),  # 426–480 Gunshot, boom, wood/glass/liquid
    # Silence & synthetic tones
    "silence": {500} | set(range(501, 506)),
    #                                         # 500      Silence
    #                                         # 501–505  Sine wave, chirp tone, pulse …
    # Acoustic environment descriptors
    "environment": set(range(506, 524)),  # 506–523 Inside/outside, reverb, echo,
    #                                         #         static, white/pink noise …
    # Media / broadcast
    "tv_radio": {524, 525, 526},  # 524 Television, 525 Radio, 526 Field rec.
}


def _build_reverse(cat_map: dict[str, set[int]]) -> dict[int, str]:
    result: dict[int, str] = {}
    for cat, indices in cat_map.items():
        for idx in indices:
            result[idx] = cat
    return result


def _other_indices(cat_map: dict[str, set[int]], n_classes: int = 527) -> list[int]:
    """Return class indices not covered by *cat_map* (excluding speech)."""
    mapped: set[int] = set()
    for indices in cat_map.values():
        mapped |= indices
    return [i for i in range(n_classes) if i not in mapped and i not in _SPEECH_INDICES]


def _category_probs(
    probs: np.ndarray,
    cat_map: dict[str, set[int]],
    n_classes: int = 527,
) -> tuple[np.ndarray, list[str]]:
    """Return (N, C) array of per-category max probabilities per window.

    Speech indices are already zeroed in *probs* before calling this.
    """
    cats = sorted(cat_map.keys())
    out = np.zeros((len(probs), len(cats)), dtype=np.float32)
    for j, cat in enumerate(cats):
        idxs = np.array(sorted(cat_map[cat] - _SPEECH_INDICES), dtype=np.intp)
        valid = idxs[(idxs >= 0) & (idxs < n_classes)]
        if len(valid):
            out[:, j] = probs[:, valid].max(axis=1)
    return out, cats


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main(dataset: str, active_threshold: float = 0.05) -> None:
    t0 = time.time()
    paths = get_dataset_paths(dataset)
    esc_dir = paths.output / "esc"

    if not esc_dir.exists():
        logger.error(f"No ESC directory: {esc_dir}")
        sys.exit(1)

    npz_files = sorted(esc_dir.glob("*.npz"))
    if not npz_files:
        logger.error(f"No NPZ files in {esc_dir}")
        sys.exit(1)

    logger.info(f"Dataset      : {dataset}")
    logger.info(f"Files        : {len(npz_files)}")
    logger.info(f"Active thresh: {active_threshold}")

    # ── Read class names from first file ─────────────────────────────────
    sample_npz = np.load(str(npz_files[0]), allow_pickle=True)
    audioset_names: list[str] = sample_npz["audioset_names"].tolist()
    n_classes = len(audioset_names)
    logger.info(f"AudioSet classes: {n_classes}")

    # ── Build reverse maps ────────────────────────────────────────────────
    orig_rev = _build_reverse(_CAT_ORIGINAL)
    new_rev = _build_reverse(_CAT_NEW)
    orig_other = _other_indices(_CAT_ORIGINAL, n_classes)
    new_other = _other_indices(_CAT_NEW, n_classes)
    logger.info(
        f"Category coverage  original={n_classes - len(orig_other) - 16} / {n_classes - 16}  "
        f"new={n_classes - len(new_other) - 16} / {n_classes - 16} non-speech classes"
    )

    orig_cats_sorted = sorted(_CAT_ORIGINAL.keys()) + ["other"]
    new_cats_sorted = sorted(_CAT_NEW.keys()) + (["other"] if new_other else [])

    # ── Fine-grained accumulators (per AudioSet class) ────────────────────
    pw_sec = np.zeros(n_classes, dtype=np.float64)  # prob-weighted seconds
    sum_prob = np.zeros(n_classes, dtype=np.float64)  # for mean_prob
    hit_cnt = np.zeros(n_classes, dtype=np.int64)  # for top-10 hit rate
    n_win_total: int = 0
    total_audio_s: float = 0.0

    # ── Coarse category accumulators ─────────────────────────────────────
    def _make_cat_accs(cats: list[str]) -> dict:
        n = len(cats)
        return dict(
            cats=cats,
            idx={c: i for i, c in enumerate(cats)},
            pw_sec=np.zeros(n, dtype=np.float64),
            sum_prob=np.zeros(n, dtype=np.float64),
            active_cnt=np.zeros(n, dtype=np.int64),
        )

    orig_acc = _make_cat_accs(orig_cats_sorted)
    new_acc = _make_cat_accs(new_cats_sorted)

    # ── Per-file records ──────────────────────────────────────────────────
    per_file_rows: list[dict] = []

    # ── Process each file ─────────────────────────────────────────────────
    log_every = max(1, len(npz_files) // 10)

    for i, npz_path in enumerate(npz_files, 1):
        uid = npz_path.stem
        try:
            data = np.load(str(npz_path), allow_pickle=True)
        except Exception as exc:
            logger.warning(f"  skip {npz_path.name}: {exc}")
            continue

        probs = data["audioset_probs"].astype(np.float32)  # (N, 527)
        pool_step = float(data["pool_step_s"])
        n_win = len(probs)
        file_dur = n_win * pool_step

        # Zero speech before all non-speech metrics
        probs[:, list(_SPEECH_INDICES)] = 0.0

        # ── Metric 1: probability-weighted seconds ────────────────────
        file_pw = probs.sum(axis=0) * pool_step  # (527,)
        pw_sec += file_pw.astype(np.float64)
        total_audio_s += file_dur

        # ── Metric 2: mean probability ────────────────────────────────
        sum_prob += probs.sum(axis=0).astype(np.float64)
        n_win_total += n_win

        # ── Metric 3: top-10 hit rate ─────────────────────────────────
        top10 = np.argpartition(probs, -10, axis=1)[:, -10:].ravel()  # (N*10,)
        hit_cnt += np.bincount(top10, minlength=n_classes).astype(np.int64)

        # ── Coarse accumulation — original map ───────────────────────
        def _accum(acc: dict, cp: np.ndarray, cat_list: list[str]) -> None:
            for j, cat in enumerate(cat_list):
                col = cp[:, j]
                ci = acc["idx"][cat]
                acc["pw_sec"][ci] += float((col * pool_step).sum())
                acc["sum_prob"][ci] += float(col.sum())
                acc["active_cnt"][ci] += int((col > active_threshold).sum())

        orig_cp, orig_cl = _category_probs(probs, _CAT_ORIGINAL, n_classes)
        new_cp, new_cl = _category_probs(probs, _CAT_NEW, n_classes)

        # "other" column for original map
        if orig_other:
            orig_other_col = probs[:, orig_other].max(axis=1)
            oi = orig_acc["idx"]["other"]
            orig_acc["pw_sec"][oi] += float((orig_other_col * pool_step).sum())
            orig_acc["sum_prob"][oi] += float(orig_other_col.sum())
            orig_acc["active_cnt"][oi] += int((orig_other_col > active_threshold).sum())
        if new_other:
            new_other_col = probs[:, new_other].max(axis=1)
            oi = new_acc["idx"]["other"]
            new_acc["pw_sec"][oi] += float((new_other_col * pool_step).sum())
            new_acc["sum_prob"][oi] += float(new_other_col.sum())
            new_acc["active_cnt"][oi] += int((new_other_col > active_threshold).sum())

        _accum(orig_acc, orig_cp, orig_cl)
        _accum(new_acc, new_cp, new_cl)

        # ── Per-file top-10 non-speech classes ───────────────────────
        file_mean_prob = probs.mean(axis=0)  # (527,) already speech-zeroed
        top10_file_idx = np.argsort(file_mean_prob)[::-1][:10]
        top10_names = [audioset_names[k] for k in top10_file_idx]
        top10_probs = [round(float(file_mean_prob[k]), 5) for k in top10_file_idx]
        dom_orig_idx = int(np.argmax(orig_cp.mean(axis=0)))
        dom_new_idx = int(np.argmax(new_cp.mean(axis=0)))

        per_file_rows.append(
            {
                "uid": uid,
                "duration_s": round(file_dur, 1),
                "n_windows": n_win,
                "dominant_cat_orig": orig_cl[dom_orig_idx],
                "dominant_cat_new": new_cl[dom_new_idx],
                "top10_classes": json.dumps(top10_names),
                "top10_mean_probs": json.dumps(top10_probs),
            }
        )

        if i % log_every == 0 or i == len(npz_files):
            elapsed = time.time() - t0
            logger.info(f"  {i}/{len(npz_files)} files  ({elapsed:.0f}s elapsed)")

    if n_win_total == 0:
        logger.error("No windows processed — check NPZ files.")
        sys.exit(1)

    total_hours = total_audio_s / 3600.0
    logger.info(f"\nTotal audio : {total_hours:.2f} h  ({n_win_total:,} × 1 s windows)")

    mean_prob = sum_prob / n_win_total  # (527,)
    hit_rate = hit_cnt / n_win_total  # (527,) fraction

    # ── Build AudioSet stats table ────────────────────────────────────────
    non_speech_mask = np.array([i not in _SPEECH_INDICES for i in range(n_classes)])
    rows_as = []
    for idx in range(n_classes):
        rows_as.append(
            {
                "class_idx": idx,
                "class_name": audioset_names[idx],
                "is_speech": idx in _SPEECH_INDICES,
                "orig_category": orig_rev.get(idx, "other"),
                "new_category": new_rev.get(idx, "other"),
                "pw_seconds": round(float(pw_sec[idx]), 2),
                "pw_hours": round(float(pw_sec[idx]) / 3600.0, 4),
                "mean_prob": round(float(mean_prob[idx]), 6),
                "top10_hit_rate": round(float(hit_rate[idx]), 6),
            }
        )
    as_df = pl.DataFrame(rows_as)

    # ── Build category stats table ────────────────────────────────────────
    cat_rows = []

    def _cat_row(label: str, acc: dict, cat: str, map_version: str) -> dict:
        ci = acc["idx"][cat]
        return {
            "map_version": map_version,
            "category": cat,
            "pw_seconds": round(float(acc["pw_sec"][ci]), 2),
            "pw_hours": round(float(acc["pw_sec"][ci]) / 3600.0, 4),
            "mean_prob": round(float(acc["sum_prob"][ci]) / n_win_total, 6),
            "active_rate": round(float(acc["active_cnt"][ci]) / n_win_total, 6),
        }

    for cat in orig_cats_sorted:
        cat_rows.append(_cat_row("original", orig_acc, cat, "original"))
    for cat in new_cats_sorted:
        cat_rows.append(_cat_row("new", new_acc, cat, "new"))
    cat_df = pl.DataFrame(cat_rows)

    # ── Per-file summary ──────────────────────────────────────────────────
    file_df = pl.DataFrame(per_file_rows)

    # ── Save outputs ──────────────────────────────────────────────────────
    stats_dir = paths.output / "esc_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    as_df.write_parquet(stats_dir / "audioset_stats.parquet")
    cat_df.write_parquet(stats_dir / "category_stats.parquet")
    file_df.write_parquet(stats_dir / "per_file_summary.parquet")
    logger.info(f"Saved parquet files to {stats_dir}")

    # ── Detailed logging ──────────────────────────────────────────────────
    hr = "─" * 70

    # 1. Top 50 non-speech AudioSet classes by prob-weighted seconds
    logger.info(f"\n{hr}")
    logger.info("TOP 50 NON-SPEECH CLASSES  —  probability-weighted seconds")
    logger.info(f"{hr}")
    top50 = (
        as_df.filter(~pl.col("is_speech")).sort("pw_seconds", descending=True).head(50)
    )
    for row in top50.iter_rows(named=True):
        change = ""
        if row["orig_category"] != row["new_category"]:
            change = f"  [{row['orig_category']} → {row['new_category']}]"
        logger.info(
            f"  [{row['class_idx']:3d}] {row['class_name']:<40s}"
            f"  pw={row['pw_hours']:7.2f}h  mean={row['mean_prob']:.4f}"
            f"  hit={row['top10_hit_rate']:.3f}"
            f"  cat={row['new_category']}"
            f"{change}"
        )

    # 2. Top 50 by mean probability
    logger.info(f"\n{hr}")
    logger.info("TOP 50 NON-SPEECH CLASSES  —  mean probability")
    logger.info(f"{hr}")
    top50_mean = (
        as_df.filter(~pl.col("is_speech")).sort("mean_prob", descending=True).head(50)
    )
    for row in top50_mean.iter_rows(named=True):
        logger.info(
            f"  [{row['class_idx']:3d}] {row['class_name']:<40s}"
            f"  mean={row['mean_prob']:.5f}  hit={row['top10_hit_rate']:.3f}"
            f"  cat={row['new_category']}"
        )

    # 3. Top 50 by hit rate
    logger.info(f"\n{hr}")
    logger.info("TOP 50 NON-SPEECH CLASSES  —  top-10 hit rate")
    logger.info(f"{hr}")
    top50_hit = (
        as_df.filter(~pl.col("is_speech"))
        .sort("top10_hit_rate", descending=True)
        .head(50)
    )
    for row in top50_hit.iter_rows(named=True):
        logger.info(
            f"  [{row['class_idx']:3d}] {row['class_name']:<40s}"
            f"  hit={row['top10_hit_rate']:.4f}  mean={row['mean_prob']:.5f}"
            f"  cat={row['new_category']}"
        )

    # 4. Original category breakdown
    logger.info(f"\n{hr}")
    logger.info("ORIGINAL CATEGORY MAP  —  all three metrics (sorted by pw_hours)")
    logger.info(f"{hr}")
    orig_cat_df = cat_df.filter(pl.col("map_version") == "original").sort(
        "pw_hours", descending=True
    )
    total_pw_orig = orig_cat_df["pw_seconds"].sum()
    for row in orig_cat_df.iter_rows(named=True):
        pct = 100.0 * row["pw_seconds"] / total_pw_orig if total_pw_orig else 0.0
        logger.info(
            f"  {row['category']:<16s}  pw={row['pw_hours']:7.2f}h ({pct:5.1f}%)"
            f"  mean={row['mean_prob']:.4f}  active={row['active_rate']:.3f}"
        )

    # 5. New category breakdown
    logger.info(f"\n{hr}")
    logger.info("REDESIGNED CATEGORY MAP  —  all three metrics (sorted by pw_hours)")
    logger.info(f"{hr}")
    new_cat_df = cat_df.filter(pl.col("map_version") == "new").sort(
        "pw_hours", descending=True
    )
    total_pw_new = new_cat_df["pw_seconds"].sum()
    for row in new_cat_df.iter_rows(named=True):
        pct = 100.0 * row["pw_seconds"] / total_pw_new if total_pw_new else 0.0
        logger.info(
            f"  {row['category']:<16s}  pw={row['pw_hours']:7.2f}h ({pct:5.1f}%)"
            f"  mean={row['mean_prob']:.4f}  active={row['active_rate']:.3f}"
        )

    # 6. What was in "other" under the original map?
    orig_other_df = as_df.filter(
        (~pl.col("is_speech")) & (pl.col("orig_category") == "other")
    ).sort("pw_seconds", descending=True)
    logger.info(f"\n{hr}")
    logger.info(
        f"ORIGINAL 'other' CLASSES  —  top 30 by pw_hours  "
        f"({len(orig_other_df)} total classes in 'other')"
    )
    logger.info(f"{hr}")
    for row in orig_other_df.head(30).iter_rows(named=True):
        logger.info(
            f"  [{row['class_idx']:3d}] {row['class_name']:<40s}"
            f"  pw={row['pw_hours']:6.2f}h  mean={row['mean_prob']:.4f}"
            f"  → new cat: {row['new_category']}"
        )

    # 7. Items still in "other" under new map (should be empty)
    if new_other:
        new_other_df = as_df.filter(
            (~pl.col("is_speech")) & (pl.col("new_category") == "other")
        ).sort("pw_seconds", descending=True)
        logger.info(f"\n{hr}")
        logger.info(
            f"NEW MAP 'other' CLASSES  ({len(new_other_df)} classes — should be 0)"
        )
        logger.info(f"{hr}")
        for row in new_other_df.head(20).iter_rows(named=True):
            logger.info(f"  [{row['class_idx']:3d}] {row['class_name']}")
    else:
        logger.info(f"\n{hr}")
        logger.info("NEW MAP 'other': 0 classes — full coverage achieved!")
        logger.info(f"{hr}")

    # 8. Per-file summary
    logger.info(f"\n{hr}")
    logger.info("PER-FILE DOMINANT CATEGORIES (new map)")
    logger.info(f"{hr}")
    for row in file_df.sort("duration_s", descending=True).iter_rows(named=True):
        top3 = json.loads(row["top10_classes"])[:3]
        logger.info(
            f"  {row['uid'][-30:]:>30s}"
            f"  {row['duration_s'] / 3600:.2f}h"
            f"  dom_new={row['dominant_cat_new']:<16s}"
            f"  top3: {', '.join(top3)}"
        )

    # ── Summary JSON ──────────────────────────────────────────────────────
    top5_ns = (
        as_df.filter(~pl.col("is_speech")).sort("pw_seconds", descending=True).head(5)
    )
    summary = {
        "dataset": dataset,
        "n_files": len(npz_files),
        "total_hours": round(total_hours, 3),
        "n_windows": n_win_total,
        "n_audioset_classes_computed": n_classes,
        "n_speech_excluded": 16,
        "orig_other_n_classes": len(orig_other),
        "new_other_n_classes": len(new_other),
        "orig_other_pw_hours": (
            round(float(orig_acc["pw_sec"][orig_acc["idx"]["other"]]) / 3600.0, 3)
            if "other" in orig_acc["idx"]
            else 0.0
        ),
        "new_other_pw_hours": (
            round(float(new_acc["pw_sec"][new_acc["idx"].get("other", 0)]) / 3600.0, 3)
            if "other" in new_acc["idx"]
            else 0.0
        ),
        "top5_non_speech_classes_by_pw": [
            {
                "class": r["class_name"],
                "pw_hours": r["pw_hours"],
                "category": r["new_category"],
            }
            for r in top5_ns.iter_rows(named=True)
        ],
    }
    with open(stats_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    logger.info(f"\nWall time: {time.time() - t0:.1f}s")
    logger.info(f"Outputs  : {stats_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute fine-grained ESC statistics from PANNs NPZ outputs."
    )
    parser.add_argument("dataset", help="Dataset name (e.g. seedlings_10)")
    parser.add_argument(
        "--active_threshold",
        type=float,
        default=0.05,
        help="Probability threshold for the 'active_rate' metric (default: 0.05)",
    )
    args = parser.parse_args()
    main(**vars(args))
