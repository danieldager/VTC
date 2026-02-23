"""Per-file metadata row constructors for VTC."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from src.core.intervals import merge_pairs


# ---------------------------------------------------------------------------
# VTC metadata templates
# ---------------------------------------------------------------------------

_EMPTY_VTC_META = dict(
    vtc_threshold=float("nan"),
    vtc_vad_iou=float("nan"),
    vtc_status="",
    vtc_speech_dur=0.0,
    vtc_n_segments=0,
    vtc_label_counts="{}",
    vtc_max_sigmoid=float("nan"),
    vtc_mean_sigmoid=float("nan"),
    error="",
)


def vtc_error_row(uid: str, error: str) -> dict:
    """Metadata row for a file that errored during VTC inference."""
    return {**_EMPTY_VTC_META, "uid": uid, "vtc_status": "error", "error": error}


def vtc_meta_row(
    uid: str,
    threshold: float,
    iou: float,
    status: str,
    segments: list[dict],
    max_sigmoid: float,
    mean_sigmoid: float,
) -> dict:
    """Build a metadata row from a file's VTC results."""
    label_counts: dict[str, int] = {}
    speech_dur = 0.0
    for s in segments:
        speech_dur += s["duration"]
        label_counts[s["label"]] = label_counts.get(s["label"], 0) + 1
    return {
        "uid": uid,
        "vtc_threshold": threshold,
        "vtc_vad_iou": round(iou, 4),
        "vtc_status": status,
        "vtc_speech_dur": round(speech_dur, 3),
        "vtc_n_segments": len(segments),
        "vtc_label_counts": json.dumps(label_counts),
        "vtc_max_sigmoid": max_sigmoid,
        "vtc_mean_sigmoid": mean_sigmoid,
        "error": "",
    }


# ---------------------------------------------------------------------------
# VAD reference loading
# ---------------------------------------------------------------------------


def load_vad_merged(
    output_dir: Path,
) -> dict[str, list[tuple[float, float]]]:
    """Load merged VAD segments and return ``{uid: [(onset, offset), ...]}``.

    Reads from ``output_dir/vad_merged/*.parquet``.
    """
    vad_dir = output_dir / "vad_merged"
    if not vad_dir.exists():
        return {}

    files = sorted(vad_dir.glob("*.parquet"))
    if not files:
        return {}

    vad_df = pl.read_parquet(files)
    vad_by_uid: dict[str, list[tuple[float, float]]] = {}
    for (uid,), group_df in vad_df.group_by("uid"):
        pairs = sorted(
            zip(group_df["onset"].to_list(), group_df["offset"].to_list())
        )
        vad_by_uid[uid] = merge_pairs(pairs)
    return vad_by_uid
