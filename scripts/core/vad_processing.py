"""Single-file VAD processing — streaming audio through TenVAD.

This module is designed to run inside *worker processes* spawned by the
parallel driver.  It deliberately avoids importing ``torch`` at module
level so that each worker stays lightweight (~31 MB instead of ~500 MB).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import soundfile as sf

from ten_vad import TenVad

TARGET_SR = 16_000
LONG_SEGMENT_THRESHOLD = 10.0  # seconds
# 10 minutes at 16 kHz → ~19 MB peak per worker
_BLOCK_SAMPLES = 10 * 60 * TARGET_SR  # 9_600_000 samples


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def set_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    ``torch`` is imported lazily to avoid bloating worker processes.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Flag → segment conversion
# ---------------------------------------------------------------------------


def get_runs(flags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(speech_runs, silence_runs)`` as Nx2 arrays of frame indices."""
    if len(flags) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int)

    edges = np.flatnonzero(np.diff(flags))
    edges = np.r_[0, edges + 1, len(flags)]
    pairs = np.column_stack((edges[:-1], edges[1:]))

    if flags[0] == 1:
        speech_runs, silence_runs = pairs[0::2], pairs[1::2]
    else:
        speech_runs, silence_runs = pairs[1::2], pairs[0::2]

    return speech_runs, silence_runs


def runs_to_segments(runs: np.ndarray, hop_size: int, sr: int) -> list[dict]:
    """Convert frame-index runs to ``[{onset, offset, duration}, ...]`` in seconds."""
    if runs.size == 0:
        return []
    factor = hop_size / sr
    onsets = np.round(runs[:, 0] * factor, 3)
    offsets = np.round(runs[:, 1] * factor, 3)
    durations = np.round(offsets - onsets, 3)
    return [
        {"onset": float(o), "offset": float(off), "duration": float(d)}
        for o, off, d in zip(onsets, offsets, durations)
    ]


def segment_stats(durations: list[float]) -> dict:
    """Summary statistics for a list of durations."""
    if not durations:
        return {"max": 0.0, "min": 0.0, "sum": 0.0, "num": 0, "avg": 0.0}
    arr = np.asarray(durations)
    return {
        "max": float(arr.max()),
        "min": float(arr.min()),
        "sum": float(arr.sum()),
        "num": len(durations),
        "avg": float(arr.mean()),
    }


# ---------------------------------------------------------------------------
# Audio resampling
# ---------------------------------------------------------------------------


def resample_block(data: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Resample a block of int16 samples using linear interpolation.

    Only called when the file sample rate differs from TARGET_SR.
    """
    if from_sr == to_sr:
        return data
    n_out = int(len(data) * to_sr / from_sr)
    indices = np.linspace(0, len(data) - 1, n_out)
    lo = np.floor(indices).astype(np.int64)
    hi = np.minimum(lo + 1, len(data) - 1)
    frac = (indices - lo).astype(np.float32)
    return (data[lo] * (1 - frac) + data[hi] * frac).astype(np.int16)


# ---------------------------------------------------------------------------
# VAD error-metadata template
# ---------------------------------------------------------------------------


def vad_error_metadata(path: str, error: str) -> dict:
    """Return a metadata row for a file that failed VAD processing."""
    return {
        "success": False,
        "path": path,
        "file_id": Path(path).stem,
        "duration": 0.0,
        "original_sr": 0,
        "speech_ratio": 0.0,
        "n_speech_segments": 0,
        "n_silence_segments": 0,
        "speech_max": 0.0,
        "speech_min": 0.0,
        "speech_sum": 0.0,
        "speech_num": 0,
        "speech_avg": 0.0,
        "nospch_max": 0.0,
        "nospch_min": 0.0,
        "nospch_sum": 0.0,
        "nospch_num": 0,
        "nospch_avg": 0.0,
        "has_long_segment": False,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Per-file VAD processing  (runs inside worker processes)
# ---------------------------------------------------------------------------


def process_file(args: tuple) -> tuple[dict, list[dict]]:
    """Process one WAV file using soundfile block streaming.

    ``soundfile.blocks()`` yields fixed-size numpy arrays directly from
    disk without ever materialising more than one block in memory.

    Peak memory per worker ≈ ``_BLOCK_SAMPLES × 2`` bytes + TenVAD overhead
    ≈ 19 MB for 10-minute blocks at 16 kHz.

    Returns ``(metadata_row, segment_rows)`` where ``segment_rows`` is a
    list of dicts each containing ``file_id, onset, offset, duration``.
    """
    wav_path, hop_size, threshold = args[:3]
    progress_q = args[3] if len(args) > 3 else None
    file_id_early = Path(wav_path).stem

    try:
        vad = TenVad(hop_size=hop_size, threshold=threshold)
    except Exception as e:
        return vad_error_metadata(str(wav_path), f"TenVad init: {e}"), []

    try:
        all_flags: list[np.ndarray] = []
        total_samples = 0
        proc = vad.process
        original_sr: int = TARGET_SR

        with sf.SoundFile(str(wav_path)) as audio_file:
            original_sr = audio_file.samplerate
            need_resample = original_sr != TARGET_SR

            for block in audio_file.blocks(
                blocksize=_BLOCK_SAMPLES,
                dtype="int16",
                always_2d=True,
                fill_value=0,
            ):
                data = block[:, 0] if block.ndim == 2 else block

                if need_resample:
                    data = resample_block(data, original_sr, TARGET_SR)

                total_samples += len(data)

                # Report samples processed so the parent can track intra-file progress.
                if progress_q is not None:
                    try:
                        progress_q.put_nowait((file_id_early, total_samples))
                    except Exception:
                        pass

                n_vad_frames = len(data) // hop_size
                if n_vad_frames == 0:
                    continue
                frames = data[: n_vad_frames * hop_size].reshape(-1, hop_size)
                flags = np.empty(n_vad_frames, dtype=np.uint8)
                for i in range(n_vad_frames):
                    _, flags[i] = proc(frames[i])
                all_flags.append(flags)

        flags = np.concatenate(all_flags) if all_flags else np.array([], dtype=np.uint8)
        duration = round(total_samples / TARGET_SR, 3)

        speech_runs, silence_runs = get_runs(flags)
        speech_segs = runs_to_segments(speech_runs, hop_size, TARGET_SR)
        silence_segs = runs_to_segments(silence_runs, hop_size, TARGET_SR)

        speech_durs = [s["duration"] for s in speech_segs]
        silence_durs = [s["duration"] for s in silence_segs]

        sp = segment_stats(speech_durs)
        ns = segment_stats(silence_durs)

        has_long = any(d >= LONG_SEGMENT_THRESHOLD for d in speech_durs)
        file_id = Path(wav_path).stem

        meta = {
            "success": True,
            "path": str(wav_path),
            "file_id": file_id,
            "duration": duration,
            "original_sr": int(original_sr),
            "speech_ratio": round(float(flags.mean()), 3) if len(flags) > 0 else 0.0,
            "n_speech_segments": sp["num"],
            "n_silence_segments": ns["num"],
            **{f"speech_{k}": v for k, v in sp.items()},
            **{f"nospch_{k}": v for k, v in ns.items()},
            "has_long_segment": has_long,
            "error": "",
        }

        seg_rows = [{"file_id": file_id, **s} for s in speech_segs]
        return meta, seg_rows

    except Exception as e:
        return vad_error_metadata(str(wav_path), str(e)), []
