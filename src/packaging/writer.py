"""Write clips to WebDataset tar shards.

Each sample in a shard is a pair of files:
    {clip_id}.{flac|wav}   — audio
    {clip_id}.json         — metadata + segment labels

Shards are named ``{prefix}-{shard_idx:06d}.tar`` and capped at
*max_shard_clips* samples each.

For S3 upload, the tar files can be synced directly via
``aws s3 sync shards/ s3://bucket/dataset/`` and streamed back
with ``wds.WebDataset("s3://bucket/dataset/shards-{000000..NNNNNN}.tar")``.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import webdataset as wds

from src.packaging.clips import Clip


def _encode_audio(
    audio_path: Path,
    abs_onset: float,
    abs_offset: float,
    target_sr: int = 16_000,
    fmt: str = "flac",
) -> bytes:
    """Extract a clip from *audio_path* and encode it.

    Uses ``soundfile.read`` with start/stop frame indices for efficient
    partial reads (no full-file decode needed).
    """
    # Get file sample rate to compute frame indices
    info = sf.info(str(audio_path))
    file_sr = info.samplerate

    start_frame = int(abs_onset * file_sr)
    stop_frame = int(abs_offset * file_sr)
    # Clamp to file bounds
    start_frame = max(0, start_frame)
    stop_frame = min(int(info.frames), stop_frame)

    data, sr = sf.read(
        str(audio_path),
        start=start_frame,
        stop=stop_frame,
        dtype="float32",
        always_2d=False,
    )
    # To mono
    if data.ndim == 2:
        data = data[:, 0]

    # Resample if needed (simple linear interp, same as vad_processing)
    if sr != target_sr:
        n_out = int(len(data) * target_sr / sr)
        indices = np.linspace(0, len(data) - 1, n_out)
        lo = np.floor(indices).astype(np.int64)
        hi = np.minimum(lo + 1, len(data) - 1)
        frac = (indices - lo).astype(np.float32)
        data = data[lo] * (1 - frac) + data[hi] * frac
        sr = target_sr

    buf = io.BytesIO()
    sf.write(buf, data, sr, format=fmt.upper())
    return buf.getvalue()


def write_shards(
    clips: list[tuple[str, Path, int, Clip]],
    output_dir: Path,
    prefix: str = "shards",
    max_shard_clips: int = 100,
    audio_fmt: str = "flac",
    target_sr: int = 16_000,
) -> list[Path]:
    """Write clips into WebDataset tar shards.

    Parameters
    ----------
    clips : list[tuple[str, Path, int, Clip]]
        ``(uid, audio_path, clip_idx, Clip)`` tuples.
    output_dir : Path
        Directory to write shards into.
    prefix : str
        Shard filename prefix.
    max_shard_clips : int
        Maximum clips per shard.
    audio_fmt : str
        ``"flac"`` or ``"wav"``.
    target_sr : int
        Target sample rate for audio clips.

    Returns
    -------
    list[Path]
        Paths to the written shard files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_paths: list[Path] = []
    shard_idx = 0
    shard_pattern = str(output_dir / f"{prefix}-%06d.tar")
    sink: wds.TarWriter | None = None  # type: ignore[attr-defined]
    count_in_shard = 0

    try:
        for uid, audio_path, clip_idx, clip in clips:
            # Rotate shards
            if sink is None or count_in_shard >= max_shard_clips:
                if sink is not None:
                    sink.close()
                shard_path = Path(shard_pattern % shard_idx)
                shard_paths.append(shard_path)
                sink = wds.TarWriter(str(shard_path))  # type: ignore[attr-defined]
                shard_idx += 1
                count_in_shard = 0

            clip_id = f"{uid}_{clip_idx:04d}"
            meta = clip.to_metadata(uid, clip_idx)
            meta["audio_fmt"] = audio_fmt
            meta["sample_rate"] = target_sr
            meta["source_path"] = str(audio_path)

            # Encode audio
            audio_bytes = _encode_audio(
                audio_path, clip.abs_onset, clip.abs_offset,
                target_sr=target_sr, fmt=audio_fmt,
            )

            sample = {
                "__key__": clip_id,
                f"{audio_fmt}": audio_bytes,
                "json": json.dumps(meta, ensure_ascii=False).encode("utf-8"),
            }
            sink.write(sample)  # type: ignore
            count_in_shard += 1

    finally:
        if sink is not None:
            sink.close()

    return shard_paths
