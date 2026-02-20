"""Threshold sweeping — adaptive and default, for whole-file and region-based logits.

All functions accept logits (or region-data lists) and return
``(threshold, iou, intervals)`` or just ``intervals``.
"""

from __future__ import annotations

import torch

from segma.inference import apply_thresholds, create_intervals
from segma.models.base import ConvolutionSettings
from segma.utils.encoders import MultiLabelEncoder

from scripts.core.intervals import compute_iou, intervals_to_pairs


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _make_threshold_dict(labels: list[str], threshold: float) -> dict:
    return {
        label: {"lower_bound": threshold, "upper_bound": 1.0}
        for label in labels
    }


def _threshold_region_data(
    region_data: list[tuple[int, torch.Tensor]],
    thresh_dict: dict,
    conv_settings: ConvolutionSettings,
    l_encoder: MultiLabelEncoder,
) -> list[tuple[int, int, str]]:
    """Apply a threshold to every region and return file-absolute intervals."""
    all_intervals: list[tuple[int, int, str]] = []
    for region_start_f, logits_t in region_data:
        thresholded = apply_thresholds(logits_t, thresh_dict, "cpu").detach()
        intervals = create_intervals(thresholded, conv_settings, l_encoder)
        for start_f, end_f, label in intervals:
            all_intervals.append((start_f + region_start_f, end_f + region_start_f, label))
    return all_intervals


# ---------------------------------------------------------------------------
# Single-tensor API  (used when there are no activity regions)
# ---------------------------------------------------------------------------


def find_best_threshold(
    logits_t: torch.Tensor,
    vad_pairs: list[tuple[float, float]],
    conv_settings: ConvolutionSettings,
    l_encoder: MultiLabelEncoder,
    thresholds: list[float],
    target_iou: float,
) -> tuple[float, float, list[tuple[int, int, str]]]:
    """Sweep thresholds high→low.  Return ``(threshold, iou, intervals)``.

    Returns the highest threshold meeting *target_iou*, or the one with
    the best IoU if none meet the target.
    """
    best_thresh = thresholds[-1]
    best_iou = 0.0
    best_intervals: list[tuple[int, int, str]] = []

    for thresh in thresholds:
        thresh_dict = _make_threshold_dict(l_encoder._labels, thresh)
        thresholded = apply_thresholds(logits_t, thresh_dict, "cpu").detach()
        intervals = create_intervals(thresholded, conv_settings, l_encoder)
        vtc_pairs = intervals_to_pairs(intervals)
        iou = compute_iou(vtc_pairs, vad_pairs)

        if iou >= target_iou:
            return thresh, iou, intervals

        if iou > best_iou:
            best_iou = iou
            best_thresh = thresh
            best_intervals = intervals

    return best_thresh, best_iou, best_intervals


def apply_default_threshold(
    logits_t: torch.Tensor,
    threshold: float,
    conv_settings: ConvolutionSettings,
    l_encoder: MultiLabelEncoder,
) -> list[tuple[int, int, str]]:
    """Apply a single threshold and return intervals."""
    thresh_dict = _make_threshold_dict(l_encoder._labels, threshold)
    thresholded = apply_thresholds(logits_t, thresh_dict, "cpu").detach()
    return create_intervals(thresholded, conv_settings, l_encoder)


# ---------------------------------------------------------------------------
# Region-data API  (list of (region_start_f, logits) tuples)
# ---------------------------------------------------------------------------


def find_best_threshold_regions(
    region_data: list[tuple[int, torch.Tensor]],
    vad_pairs: list[tuple[float, float]],
    conv_settings: ConvolutionSettings,
    l_encoder: MultiLabelEncoder,
    thresholds: list[float],
    target_iou: float,
) -> tuple[float, float, list[tuple[int, int, str]]]:
    """Like :func:`find_best_threshold` but operating on per-region logits.

    Each element of *region_data* is ``(region_start_f, logits_tensor)``
    where logits are region-relative.  Intervals are offset to file-
    absolute positions before IoU is computed against *vad_pairs*.
    """
    best_thresh = thresholds[-1]
    best_iou = 0.0
    best_intervals: list[tuple[int, int, str]] = []

    for thresh in thresholds:
        thresh_dict = _make_threshold_dict(l_encoder._labels, thresh)
        intervals = _threshold_region_data(
            region_data, thresh_dict, conv_settings, l_encoder,
        )
        vtc_pairs = intervals_to_pairs(intervals)
        iou = compute_iou(vtc_pairs, vad_pairs)

        if iou >= target_iou:
            return thresh, iou, intervals

        if iou > best_iou:
            best_iou = iou
            best_thresh = thresh
            best_intervals = intervals

    return best_thresh, best_iou, best_intervals


def apply_default_threshold_regions(
    region_data: list[tuple[int, torch.Tensor]],
    threshold: float,
    conv_settings: ConvolutionSettings,
    l_encoder: MultiLabelEncoder,
) -> list[tuple[int, int, str]]:
    """Like :func:`apply_default_threshold` but for per-region logits."""
    thresh_dict = _make_threshold_dict(l_encoder._labels, threshold)
    return _threshold_region_data(region_data, thresh_dict, conv_settings, l_encoder)
