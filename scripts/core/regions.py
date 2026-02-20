"""Activity-region logic — restrict VTC inference to speech-active spans.

Given VAD speech segments, merge them into coarse *activity regions*
(large padded time spans) and run the segma model on each region
individually.  For long recordings with low speech ratios (e.g. 12-hour
seedlings files at ~5 % speech), this cuts GPU time by an order of
magnitude.
"""

from __future__ import annotations

from math import floor
from pathlib import Path

import torch

from segma.inference import Chunkyfier, apply_model_on_audio, prepare_audio
from segma.models.base import BaseSegmentationModel, ConvolutionSettings

from scripts.core.intervals import merge_pairs

# segma's minimum tail length — shorter regions produce no model output.
_MIN_REGION_SAMPLES = 400


# ---------------------------------------------------------------------------
# Region construction
# ---------------------------------------------------------------------------


def merge_into_activity_regions(
    vad_pairs: list[tuple[float, float]],
    file_duration_s: float,
    merge_gap_s: float = 30.0,
    pad_s: float = 5.0,
) -> list[tuple[float, float]]:
    """Merge VAD speech segments into coarse *activity regions*.

    Adjacent segments closer than *merge_gap_s* are merged, then each
    region is padded by *pad_s* on both sides and clipped to
    ``[0, file_duration_s]``.

    Returns a short list of non-overlapping ``(onset, offset)`` spans.
    """
    if not vad_pairs:
        return []
    sorted_pairs = sorted(vad_pairs)
    merged: list[list[float]] = [list(sorted_pairs[0])]
    for onset, offset in sorted_pairs[1:]:
        if onset - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = max(merged[-1][1], offset)
        else:
            merged.append([onset, offset])
    # Pad and clip
    padded: list[tuple[float, float]] = []
    for onset, offset in merged:
        padded.append((
            max(0.0, onset - pad_s),
            min(file_duration_s, offset + pad_s),
        ))
    # Final merge in case padding caused overlaps
    return merge_pairs(padded)


def activity_region_coverage(
    regions: list[tuple[float, float]],
    file_duration_s: float,
) -> float:
    """Fraction of the file covered by the activity regions."""
    if file_duration_s <= 0:
        return 1.0
    return sum(end - start for start, end in regions) / file_duration_s


# ---------------------------------------------------------------------------
# Region-aware model forward pass
# ---------------------------------------------------------------------------


def apply_model_on_region(
    audio_path: Path,
    model: BaseSegmentationModel,
    conv_settings: ConvolutionSettings,
    device: str,
    batch_size: int,
    chunk_duration_s: float,
    region_start_s: float,
    region_end_s: float,
    sample_rate: int = 16_000,
) -> torch.Tensor:
    """Run the segma model on a specific time region of an audio file.

    Mirrors ``segma.inference.apply_model_on_audio`` but restricts the
    forward pass to ``[region_start_s, region_end_s]``.  All
    ``prepare_audio`` calls are offset so that segma's torchcodec reader
    seeks to the correct absolute file position.

    Returns logits of shape ``(n_frames, n_classes)`` covering only this
    region.  Sample-index intervals derived from these logits are
    **region-relative** — the caller must add ``region_start_f`` to get
    file-absolute positions.
    """
    chunk_duration_f = int(chunk_duration_s * sample_rate)
    region_start_f = int(region_start_s * sample_rate)
    region_end_f = int(region_end_s * sample_rate)
    n_frames_region = region_end_f - region_start_f

    if n_frames_region < _MIN_REGION_SAMPLES:
        return torch.empty(0, model.label_encoder.n_labels)

    chunkyfier = Chunkyfier(batch_size, chunk_duration_f, conv_settings)

    n_fitting_chunks = chunkyfier.get_n_fitting_chunks(n_frames_region)
    n_full_batches = floor(n_fitting_chunks / batch_size)

    logits: list[torch.Tensor] = []

    # --- Full batches ---
    for i in range(n_full_batches):
        sub_audio_t = prepare_audio(
            audio_path,
            model,
            device=device,
            start_f=region_start_f + chunkyfier.batch_start_i(i),
            end_f=region_start_f + chunkyfier.batch_end_i_coverage(i) + chunkyfier.missing_n_frames,
        )
        batch_t = sub_audio_t.unfold(
            0,
            size=chunk_duration_f,
            step=chunk_duration_f - chunkyfier.missing_n_frames,
        )
        if batch_t.shape[0] != batch_size:
            raise ValueError(
                f"Unfolding error: got {batch_t.shape[0]} chunks, "
                f"expected {batch_size}"
            )
        with torch.inference_mode():
            out_t = model(batch_t).squeeze(2)
        logits.append(out_t)

    # --- Leftover chunks that don't fill a batch ---
    if n_full_batches > 0:
        leftover_frames = n_frames_region - chunkyfier.batch_end_i_coverage(
            n_full_batches - 1
        )
    else:
        leftover_frames = n_frames_region

    if leftover_frames and leftover_frames >= chunk_duration_f:
        sub_audio_t = prepare_audio(
            audio_path,
            model,
            device=device,
            start_f=region_start_f + chunkyfier.batch_start_i(n_full_batches),
            end_f=region_start_f + chunkyfier.chunk_start_i(
                n_full_batches * batch_size
                + chunkyfier.get_n_fitting_chunks(leftover_frames)
            ) + chunkyfier.missing_n_frames,
        )
        batch_t = sub_audio_t.unfold(
            0,
            size=chunk_duration_f,
            step=chunk_duration_f - chunkyfier.missing_n_frames,
        )
        with torch.inference_mode():
            out_t = model(batch_t)
        logits.append(out_t)

    # --- Tail frames (< 1 chunk but >= 400 samples) ---
    last_start_position = chunkyfier.chunk_start_i(
        n_full_batches * batch_size
        + chunkyfier.get_n_fitting_chunks(leftover_frames)
    )
    if n_frames_region - last_start_position >= _MIN_REGION_SAMPLES:
        last_audio_t = prepare_audio(
            audio_path,
            model,
            device=device,
            start_f=region_start_f + last_start_position,
            end_f=region_start_f + n_frames_region,
        )
        with torch.inference_mode():
            out_last_t = model(last_audio_t[None, :])
        logits.append(out_last_t)

    if not logits:
        return torch.empty(0, model.label_encoder.n_labels)

    return torch.concat(
        [t.view(-1, model.label_encoder.n_labels) for t in logits], dim=0
    )


def forward_pass_regions(
    audio_path: Path,
    model: BaseSegmentationModel,
    conv_settings: ConvolutionSettings,
    device: str,
    batch_size: int,
    chunk_duration_s: float,
    regions: list[tuple[float, float]],
    sample_rate: int = 16_000,
) -> list[tuple[int, torch.Tensor]]:
    """Run the model forward pass on each activity region.

    Returns ``[(region_start_f, logits_tensor), ...]``.  Logits are
    *region-relative* — the caller adds ``region_start_f`` when converting
    intervals back to file positions.
    """
    results: list[tuple[int, torch.Tensor]] = []
    for start_s, end_s in regions:
        region_start_f = int(start_s * sample_rate)
        logits_t = apply_model_on_region(
            audio_path, model, conv_settings, device,
            batch_size, chunk_duration_s, start_s, end_s,
            sample_rate=sample_rate,
        )
        if logits_t.numel() > 0:
            results.append((region_start_f, logits_t))
    return results


def forward_pass_full_file(
    audio_path: Path,
    model: BaseSegmentationModel,
    conv_settings: ConvolutionSettings,
    device: str,
    batch_size: int,
    chunk_duration_s: float,
) -> list[tuple[int, torch.Tensor]]:
    """Run the model on the entire file, returning the same format as
    :func:`forward_pass_regions` (a single ``(0, logits)`` entry)."""
    with torch.no_grad():
        logits_t = apply_model_on_audio(
            audio_path=audio_path,
            model=model,
            conv_settings=conv_settings,
            device=device,
            batch_size=batch_size,
            chunk_duration_s=chunk_duration_s,
        )
    return [(0, logits_t)]
