#!/usr/bin/env python3
"""Threshold comparison analysis — sweep VAD × VTC thresholds on saved logits.

Loads raw per-frame VAD probabilities (.npz) and VTC logits (.pt) produced
by ``--save_logits``, applies every combination of thresholds, and computes
per-file IoU / Precision / Recall plus global aggregate metrics.

Outputs (to ``figures/{dataset}/``):
    threshold_grid.csv       — full grid of aggregate metrics
    threshold_heatmap.png    — IoU heatmap across VAD × VTC thresholds
    volume_sensitivity.png   — total speech hours vs threshold for each system
    per_file_grid.csv        — per-file IoU at every VAD×VTC combo (optional)

Usage:
    python -m src.analysis.threshold_comparison seedlings_10
    python -m src.analysis.threshold_comparison seedlings_10 --vad_step 0.05
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from types import ModuleType

    import torch

# ---------------------------------------------------------------------------
# Lazy torch / segma imports — only needed for VTC re-thresholding
# NOTE: We do NOT import ``segma.inference`` because it transitively
# pulls in ``torchcodec`` which needs FFmpeg shared libs at import time.
# The two tiny functions we need (apply_thresholds, create_intervals) are
# inlined below to avoid that dependency chain.
# ---------------------------------------------------------------------------
_torch: ModuleType | None = None
_conv_settings_cls: Any = None
_label_encoder_cls: Any = None


def _ensure_torch() -> ModuleType:
    global _torch
    if _torch is None:
        import torch as _t

        _torch = _t
    return _torch  # type: ignore[return-value]


def _ensure_segma_types() -> tuple[Any, Any]:
    """Import ConvolutionSettings and MultiLabelEncoder (no torchcodec needed)."""
    global _conv_settings_cls, _label_encoder_cls
    if _conv_settings_cls is None:
        from segma.models.base import ConvolutionSettings
        from segma.utils.encoders import MultiLabelEncoder

        _conv_settings_cls = ConvolutionSettings
        _label_encoder_cls = MultiLabelEncoder
    return _conv_settings_cls, _label_encoder_cls


def _apply_thresholds(
    feature_tensor: Any,
    thresholds: dict[str, dict[str, float]],
    device: str,
) -> Any:
    """Apply sigmoid + threshold to raw logits (inlined from segma.inference)."""
    torch = _ensure_torch()
    feature_tensor = feature_tensor.sigmoid()
    threshold_tensor = torch.tensor(
        [label["lower_bound"] for label in thresholds.values()]
    ).to(torch.device(device))
    return feature_tensor > threshold_tensor


def _create_intervals(
    thresholded_features: Any,
    conv_settings: Any,
    label_encoder: Any,
) -> list[tuple[int, int, str]]:
    """Build detection intervals from thresholded logits (inlined from segma.inference)."""
    intervals: list[tuple[int, int, str]] = []
    slices = np.ma.notmasked_contiguous(
        np.ma.masked_values(thresholded_features, value=0), axis=0
    )
    for label_i, label in enumerate(label_encoder.base_labels):
        for sl in slices[label_i]:
            interval_start = max(0, conv_settings.rf_start_i(sl.start))
            interval_end = conv_settings.rf_end_i(sl.stop - 1) + 1
            intervals.append((interval_start, interval_end, label))
    return intervals


# ---------------------------------------------------------------------------
# VAD re-thresholding
# ---------------------------------------------------------------------------


def vad_probs_to_pairs(
    probs: np.ndarray,
    threshold: float,
    hop_size: int,
    sr: int = 16_000,
) -> list[tuple[float, float]]:
    """Apply *threshold* to raw VAD probabilities and return merged (onset, offset) pairs."""
    flags = (probs >= threshold).astype(np.uint8)
    if len(flags) == 0:
        return []

    # Edge detection → speech runs
    edges = np.flatnonzero(np.diff(flags))
    edges = np.r_[0, edges + 1, len(flags)]
    pairs = np.column_stack((edges[:-1], edges[1:]))
    if flags[0] == 1:
        speech = pairs[0::2]
    else:
        speech = pairs[1::2]

    if speech.size == 0:
        return []

    factor = hop_size / sr
    return [
        (round(float(s * factor), 3), round(float(e * factor), 3)) for s, e in speech
    ]


# ---------------------------------------------------------------------------
# VTC re-thresholding
# ---------------------------------------------------------------------------


def vtc_logits_to_pairs(
    logits_dict: dict[str, torch.Tensor],
    threshold: float,
    conv_settings: Any,
    l_encoder: Any,
) -> list[tuple[float, float]]:
    """Apply *threshold* to saved VTC logits and return merged label-agnostic pairs."""
    torch = _ensure_torch()
    from src.core.intervals import intervals_to_pairs

    labels = l_encoder._labels
    # Reconstruct the logit tensor: (T, n_labels)
    first_key = next(iter(logits_dict))
    T = logits_dict[first_key].shape[0]
    logit_tensor = torch.zeros(T, len(labels))
    for j, label in enumerate(labels):
        if label in logits_dict:
            logit_tensor[:, j] = logits_dict[label]

    thresh_dict = {
        label: {"lower_bound": threshold, "upper_bound": 1.0} for label in labels
    }

    thresholded = _apply_thresholds(logit_tensor, thresh_dict, "cpu").detach()
    intervals = _create_intervals(thresholded, conv_settings, l_encoder)
    return intervals_to_pairs(intervals)


# ---------------------------------------------------------------------------
# IoU / Precision / Recall
# ---------------------------------------------------------------------------


def compute_metrics(
    vtc_pairs: list[tuple[float, float]],
    vad_pairs: list[tuple[float, float]],
) -> dict[str, float]:
    """Compute IoU, precision, recall, and durations."""
    vtc_dur = sum(b - a for a, b in vtc_pairs)
    vad_dur = sum(b - a for a, b in vad_pairs)

    if not vtc_pairs and not vad_pairs:
        return {
            "iou": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "vtc_dur": 0.0,
            "vad_dur": 0.0,
            "tp": 0.0,
        }
    if not vtc_pairs or not vad_pairs:
        return {
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "vtc_dur": vtc_dur,
            "vad_dur": vad_dur,
            "tp": 0.0,
        }

    # Two-pointer intersection
    tp = 0.0
    vi, ai = 0, 0
    while vi < len(vtc_pairs) and ai < len(vad_pairs):
        v_on, v_off = vtc_pairs[vi]
        a_on, a_off = vad_pairs[ai]
        overlap = min(v_off, a_off) - max(v_on, a_on)
        if overlap > 0:
            tp += overlap
        if v_off <= a_off:
            vi += 1
        else:
            ai += 1

    union = vtc_dur + vad_dur - tp
    iou = tp / union if union > 0 else 0.0
    precision = tp / vtc_dur if vtc_dur > 0 else 0.0
    recall = tp / vad_dur if vad_dur > 0 else 0.0
    return {
        "iou": round(iou, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "vtc_dur": round(vtc_dur, 3),
        "vad_dur": round(vad_dur, 3),
        "tp": round(tp, 3),
    }


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def discover_files(
    output_dir: Path,
) -> list[tuple[str, Path, Path]]:
    """Return [(uid, vad_npz_path, vtc_pt_path), ...] for files that have both."""
    vad_dir = output_dir / "vad_probs"
    vtc_dir = output_dir / "logits"

    if not vad_dir.exists():
        print(
            f"ERROR: {vad_dir} does not exist. Run VAD with --save_logits.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not vtc_dir.exists():
        print(
            f"ERROR: {vtc_dir} does not exist. Run VTC with --save_logits.",
            file=sys.stderr,
        )
        sys.exit(1)

    vad_files = {p.stem.removesuffix("-probs"): p for p in vad_dir.glob("*.npz")}
    vtc_files = {p.stem.removesuffix("-logits_dict_t"): p for p in vtc_dir.glob("*.pt")}

    common = sorted(set(vad_files) & set(vtc_files))
    if not common:
        print(
            "ERROR: No matching files found between vad_probs/ and logits/.",
            file=sys.stderr,
        )
        sys.exit(1)

    vad_only = set(vad_files) - set(vtc_files)
    vtc_only = set(vtc_files) - set(vad_files)
    if vad_only:
        print(f"  WARN: {len(vad_only)} files have VAD probs but no VTC logits")
    if vtc_only:
        print(f"  WARN: {len(vtc_only)} files have VTC logits but no VAD probs")

    return [(uid, vad_files[uid], vtc_files[uid]) for uid in common]


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def run_analysis(
    dataset: str,
    vad_thresholds: list[float],
    vtc_thresholds: list[float],
    save_per_file: bool = False,
) -> None:
    """Run the full threshold grid analysis."""
    from src.utils import get_dataset_paths

    paths = get_dataset_paths(dataset)
    fig_dir = paths.figures
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {dataset}")
    print(f"  output : {paths.output}")
    print(f"  figures: {fig_dir}")

    # --- Discover files ---
    files = discover_files(paths.output)
    n_files = len(files)
    print(f"  files  : {n_files} matched")
    print(
        f"  grid   : {len(vad_thresholds)} VAD × {len(vtc_thresholds)} VTC"
        f" = {len(vad_thresholds) * len(vtc_thresholds)} combos"
    )

    # --- Load VTC model config for re-thresholding ---
    torch = _ensure_torch()
    ConvolutionSettings, MultiLabelEncoder = _ensure_segma_types()
    from segma.config import load_config, Config

    # NOTE: segma.models triggers torchcodec, but only at model.load — we
    # need it here once to extract conv_settings from the checkpoint.
    from segma.models import Models

    config_path = "VTC-2.0/model/config.yml"
    ckpt_path = "VTC-2.0/model/best.ckpt"
    model_config: Config = load_config(config_path)
    l_encoder = MultiLabelEncoder(labels=model_config.data.classes)

    model = Models[model_config.model.name].load_from_checkpoint(
        checkpoint_path=ckpt_path,
        label_encoder=l_encoder,
        config=model_config,
        train=False,
    )
    conv_settings = model.conv_settings
    del model  # free GPU memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Pre-load all files ---
    print("Loading VAD probs + VTC logits...", flush=True)
    t0 = time.time()

    vad_data: list[tuple[str, np.ndarray, int]] = []  # (uid, probs, hop_size)
    vtc_data: list[tuple[str, dict]] = []  # (uid, logits_dict)

    for uid, vad_path, vtc_path in files:
        # VAD
        npz = np.load(vad_path)
        probs = npz["probs"]
        hop_size = int(npz["hop_size"])
        vad_data.append((uid, probs, hop_size))

        # VTC
        logits_dict = torch.load(vtc_path, map_location="cpu", weights_only=True)
        vtc_data.append((uid, logits_dict))

    load_s = time.time() - t0
    print(f"  Loaded {n_files} files in {load_s:.1f}s", flush=True)

    # --- Volume sensitivity analysis (single-system threshold sweep) ---
    print("Running volume sensitivity analysis...", flush=True)
    all_thresholds = sorted(set(vad_thresholds) | set(vtc_thresholds))

    vol_rows: list[dict] = []
    for thresh in all_thresholds:
        vad_hours = 0.0
        vtc_hours = 0.0
        for uid, probs, hop_size in vad_data:
            pairs = vad_probs_to_pairs(probs, thresh, hop_size)
            vad_hours += sum(b - a for a, b in pairs) / 3600
        for uid, logits_dict in vtc_data:
            pairs = vtc_logits_to_pairs(logits_dict, thresh, conv_settings, l_encoder)
            vtc_hours += sum(b - a for a, b in pairs) / 3600
        vol_rows.append(
            {
                "threshold": thresh,
                "vad_hours": round(vad_hours, 2),
                "vtc_hours": round(vtc_hours, 2),
            }
        )
        print(
            f"  t={thresh:.2f}  VAD={vad_hours:.1f}h  VTC={vtc_hours:.1f}h",
            flush=True,
        )

    vol_df = pl.DataFrame(vol_rows)
    vol_df.write_csv(fig_dir / "volume_sensitivity.csv")

    # --- Full grid sweep ---
    print("Running threshold grid sweep...", flush=True)
    t0 = time.time()
    grid_rows: list[dict] = []
    per_file_rows: list[dict] = []
    n_combos = len(vad_thresholds) * len(vtc_thresholds)

    # Pre-compute VAD pairs at each threshold to avoid redundant work
    print("  Pre-computing VAD segments...", flush=True)
    vad_pairs_cache: dict[float, dict[str, list[tuple[float, float]]]] = {}
    for vt in vad_thresholds:
        by_uid: dict[str, list[tuple[float, float]]] = {}
        for uid, probs, hop_size in vad_data:
            by_uid[uid] = vad_probs_to_pairs(probs, vt, hop_size)
        vad_pairs_cache[vt] = by_uid

    # Pre-compute VTC pairs at each threshold
    print("  Pre-computing VTC segments...", flush=True)
    vtc_pairs_cache: dict[float, dict[str, list[tuple[float, float]]]] = {}
    for vt in vtc_thresholds:
        by_uid: dict[str, list[tuple[float, float]]] = {}
        for uid, logits_dict in vtc_data:
            by_uid[uid] = vtc_logits_to_pairs(logits_dict, vt, conv_settings, l_encoder)
        vtc_pairs_cache[vt] = by_uid

    print("  Computing metrics...", flush=True)
    combo_i = 0
    for vad_t in vad_thresholds:
        for vtc_t in vtc_thresholds:
            combo_i += 1
            ious, precs, recs = [], [], []
            total_vtc_dur, total_vad_dur, total_tp = 0.0, 0.0, 0.0

            for uid, _, _ in vad_data:
                vad_p = vad_pairs_cache[vad_t][uid]
                vtc_p = vtc_pairs_cache[vtc_t][uid]
                m = compute_metrics(vtc_p, vad_p)
                ious.append(m["iou"])
                precs.append(m["precision"])
                recs.append(m["recall"])
                total_vtc_dur += m["vtc_dur"]
                total_vad_dur += m["vad_dur"]
                total_tp += m["tp"]

                if save_per_file:
                    per_file_rows.append(
                        {
                            "uid": uid,
                            "vad_threshold": vad_t,
                            "vtc_threshold": vtc_t,
                            **m,
                        }
                    )

            mean_iou = np.mean(ious)
            grid_rows.append(
                {
                    "vad_threshold": vad_t,
                    "vtc_threshold": vtc_t,
                    "mean_iou": round(float(mean_iou), 4),
                    "median_iou": round(float(np.median(ious)), 4),
                    "mean_precision": round(float(np.mean(precs)), 4),
                    "mean_recall": round(float(np.mean(recs)), 4),
                    "total_vtc_hours": round(total_vtc_dur / 3600, 2),
                    "total_vad_hours": round(total_vad_dur / 3600, 2),
                    "total_tp_hours": round(total_tp / 3600, 2),
                    "vtc_vad_ratio": (
                        round(total_vtc_dur / total_vad_dur, 3)
                        if total_vad_dur > 0
                        else float("nan")
                    ),
                }
            )

            if combo_i % 10 == 0 or combo_i == n_combos:
                elapsed = time.time() - t0
                print(
                    f"  {combo_i}/{n_combos}  [{elapsed:.0f}s]  "
                    f"VAD={vad_t:.2f} VTC={vtc_t:.2f} → IoU={mean_iou:.3f}",
                    flush=True,
                )

    grid_df = pl.DataFrame(grid_rows)
    grid_df.write_csv(fig_dir / "threshold_grid.csv")
    print(f"\n  Saved: {fig_dir / 'threshold_grid.csv'}")

    if save_per_file and per_file_rows:
        pf_df = pl.DataFrame(per_file_rows)
        pf_df.write_csv(fig_dir / "per_file_grid.csv")
        print(f"  Saved: {fig_dir / 'per_file_grid.csv'}")

    sweep_s = time.time() - t0
    print(f"  Grid sweep completed in {sweep_s:.0f}s")

    # --- Generate plots ---
    print("\nGenerating plots...", flush=True)
    _plot_heatmap(grid_df, fig_dir)
    _plot_volume_sensitivity(vol_df, fig_dir)

    # --- Find best combo ---
    best = grid_df.sort("mean_iou", descending=True).row(0, named=True)
    print(f"\nBest combo:")
    print(f"  VAD threshold: {best['vad_threshold']:.2f}")
    print(f"  VTC threshold: {best['vtc_threshold']:.2f}")
    print(f"  Mean IoU:      {best['mean_iou']:.4f}")
    print(f"  Mean Precision:{best['mean_precision']:.4f}")
    print(f"  Mean Recall:   {best['mean_recall']:.4f}")
    print(f"  VTC/VAD ratio: {best['vtc_vad_ratio']:.3f}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_heatmap(grid_df: pl.DataFrame, fig_dir: Path) -> None:
    """Generate IoU + Precision + Recall heatmaps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vad_vals = sorted(grid_df["vad_threshold"].unique().to_list())
    vtc_vals = sorted(grid_df["vtc_threshold"].unique().to_list())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, metric, title in zip(
        axes,
        ["mean_iou", "mean_precision", "mean_recall"],
        ["Mean IoU", "Mean Precision", "Mean Recall"],
    ):
        matrix = np.zeros((len(vad_vals), len(vtc_vals)))
        for row in grid_df.iter_rows(named=True):
            vi = vad_vals.index(row["vad_threshold"])
            vj = vtc_vals.index(row["vtc_threshold"])
            matrix[vi, vj] = row[metric]

        im = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
        )
        ax.set_xticks(range(len(vtc_vals)))
        ax.set_xticklabels([f"{v:.2f}" for v in vtc_vals], rotation=45)
        ax.set_yticks(range(len(vad_vals)))
        ax.set_yticklabels([f"{v:.2f}" for v in vad_vals])
        ax.set_xlabel("VTC threshold")
        ax.set_ylabel("VAD threshold")
        ax.set_title(title)

        # Annotate cells
        for i in range(len(vad_vals)):
            for j in range(len(vtc_vals)):
                val = matrix[i, j]
                color = "white" if val < 0.4 or val > 0.85 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                )

        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("VAD × VTC Threshold Grid", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = fig_dir / "threshold_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def _plot_volume_sensitivity(vol_df: pl.DataFrame, fig_dir: Path) -> None:
    """Bar chart showing total speech hours vs threshold for each system."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = vol_df["threshold"].to_list()
    vad_hours = vol_df["vad_hours"].to_list()
    vtc_hours = vol_df["vtc_hours"].to_list()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(thresholds))
    width = 0.35

    ax.bar(
        x - width / 2,
        vad_hours,
        width,
        label="VAD (TenVAD)",
        color="#2196F3",
        alpha=0.8,
    )
    ax.bar(x + width / 2, vtc_hours, width, label="VTC", color="#FF5722", alpha=0.8)

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Total speech (hours)")
    ax.set_title("Speech volume sensitivity to threshold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = fig_dir / "volume_sensitivity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Threshold grid analysis: sweep VAD × VTC thresholds on saved logits.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.analysis.threshold_comparison seedlings_10\n"
            "  python -m src.analysis.threshold_comparison seedlings_10 --vad_step 0.05\n"
        ),
    )
    parser.add_argument(
        "dataset",
        help="Dataset name (must have vad_probs/ and logits/ in its output dir).",
    )
    parser.add_argument(
        "--vad_min",
        type=float,
        default=0.1,
        help="Minimum VAD threshold (default: 0.1)",
    )
    parser.add_argument(
        "--vad_max",
        type=float,
        default=0.9,
        help="Maximum VAD threshold (default: 0.9)",
    )
    parser.add_argument(
        "--vad_step",
        type=float,
        default=0.1,
        help="VAD threshold step (default: 0.1)",
    )
    parser.add_argument(
        "--vtc_min",
        type=float,
        default=0.1,
        help="Minimum VTC threshold (default: 0.1)",
    )
    parser.add_argument(
        "--vtc_max",
        type=float,
        default=0.9,
        help="Maximum VTC threshold (default: 0.9)",
    )
    parser.add_argument(
        "--vtc_step",
        type=float,
        default=0.1,
        help="VTC threshold step (default: 0.1)",
    )
    parser.add_argument(
        "--per_file",
        action="store_true",
        help="Save per-file metrics (large CSV).",
    )

    args = parser.parse_args()

    # Build threshold lists
    vad_thresholds = []
    t = args.vad_min
    while t <= args.vad_max + 1e-9:
        vad_thresholds.append(round(t, 4))
        t += args.vad_step

    vtc_thresholds = []
    t = args.vtc_min
    while t <= args.vtc_max + 1e-9:
        vtc_thresholds.append(round(t, 4))
        t += args.vtc_step

    print(f"VAD thresholds: {vad_thresholds}")
    print(f"VTC thresholds: {vtc_thresholds}")

    run_analysis(
        dataset=args.dataset,
        vad_thresholds=vad_thresholds,
        vtc_thresholds=vtc_thresholds,
        save_per_file=args.per_file,
    )


if __name__ == "__main__":
    main()
