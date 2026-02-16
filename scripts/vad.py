#!/usr/bin/env python3
"""
TenVAD pipeline: voice activity detection with multiprocessing.

All paths derived from the dataset name:
    manifests/{dataset}.parquet          input manifest
    metadata/{dataset}/ten/metadata.parquet   per-file metadata
    output/{dataset}/vad_raw/            raw VAD segments
    output/{dataset}/vad_merged/         merged VAD segments
"""

import sys
import time
import random
import argparse
import warnings
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import polars as pl
import torch
import torchaudio

warnings.filterwarnings("ignore", message=".*In 2.9.*")
warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*")

from ten_vad import TenVad
from scripts.utils import get_dataset_paths, get_task_shard, load_manifest, merge_segments_df, log_progress, get_log_interval, atomic_write_parquet

TARGET_SR = 16_000
LONG_SEGMENT_THRESHOLD = 10.0  # seconds – copy file to data/ if any segment >= this


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_runs(flags: np.ndarray):
    """Return (speech_runs, silence_runs) as Nx2 arrays of frame indices."""
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
    """Convert frame-index runs to list of {onset, offset, duration} in seconds (3dp)."""
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
# Per-file processing (runs inside worker processes)
# ---------------------------------------------------------------------------


def _error_meta(path: str, error: str) -> dict:
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


def process_file(args: tuple) -> tuple[dict, list[dict]]:
    """
    Process one WAV file.

    Returns (metadata_row, segment_rows) where segment_rows is a list of
    dicts each containing file_id, onset, offset, duration.
    """
    wav_path, hop_size, threshold = args

    try:
        vad = TenVad(hop_size=hop_size, threshold=threshold)
    except Exception as e:
        return _error_meta(str(wav_path), f"TenVad init: {e}"), []

    try:
        waveform, sr = torchaudio.load(str(wav_path))
        original_sr = sr

        # Mono
        if waveform.size(0) > 1:
            waveform = waveform[0:1, :]

        # Resample to 16 kHz if needed
        if sr != TARGET_SR:
            waveform = torchaudio.transforms.Resample(sr, TARGET_SR)(waveform)
            sr = TARGET_SR

        data = (waveform.squeeze().numpy() * 32767).astype(np.int16)
        duration = round(len(data) / sr, 3)

        # Frame-level VAD
        n_frames = len(data) // hop_size
        frames = data[: n_frames * hop_size].reshape(-1, hop_size)
        flags = np.empty(n_frames, dtype=np.uint8)
        proc = vad.process
        for i in range(n_frames):
            _, flags[i] = proc(frames[i])

        # Compute segments
        speech_runs, silence_runs = get_runs(flags)
        speech_segs = runs_to_segments(speech_runs, hop_size, sr)
        silence_segs = runs_to_segments(silence_runs, hop_size, sr)

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
            "speech_ratio": round(float(flags.mean()), 3),
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
        return _error_meta(str(wav_path), str(e)), []


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------

_CKPT_META = "_checkpoint_meta.parquet"
_CKPT_SEGS = "_checkpoint_segs.parquet"


def _save_checkpoint(
    directory: Path, meta_rows: list[dict], seg_rows: list[dict]
) -> None:
    """Write in-progress results so the run can be resumed."""
    if meta_rows:
        pl.DataFrame(meta_rows).write_parquet(
            directory / _CKPT_META, compression="zstd"
        )
    if seg_rows:
        pl.DataFrame(seg_rows).write_parquet(
            directory / _CKPT_SEGS, compression="zstd"
        )


def _clear_checkpoint(directory: Path) -> None:
    """Remove checkpoint files after a successful run."""
    for name in (_CKPT_META, _CKPT_SEGS):
        p = directory / name
        if p.exists():
            p.unlink()


# ---------------------------------------------------------------------------
# Parallel driver
# ---------------------------------------------------------------------------


def process_parallel(
    wavs: list[Path],
    hop_size: int,
    threshold: float,
    workers: int,
    checkpoint_dir: Path | None = None,
    checkpoint_interval: int = 5000,
) -> tuple[list[dict], list[dict]]:
    """Run VAD on all files using a process pool. Returns (metadata_rows, segment_rows)."""
    tasks = [(w, hop_size, threshold) for w in wavs]
    meta_rows: list[dict] = []
    seg_rows: list[dict] = []
    errors = 0
    total = len(tasks)
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_file, t): t[0] for t in tasks}

        for i, future in enumerate(as_completed(futures), 1):
            path = futures[future]
            try:
                meta, segs = future.result()
                meta_rows.append(meta)
                seg_rows.extend(segs)
                if not meta.get("success", False):
                    errors += 1
                    print(f"  WARN: {Path(path).name}: {meta['error']}", file=sys.stderr)
            except Exception as e:
                errors += 1
                print(f"  ERROR: {Path(path).name}: {e}", file=sys.stderr)

            if i % get_log_interval(i) == 0 or i == total:
                log_progress(i, total, t0, "VAD")

            if checkpoint_dir and i % checkpoint_interval == 0:
                _save_checkpoint(checkpoint_dir, meta_rows, seg_rows)

    elapsed = time.time() - t0
    print(f"Processed {len(meta_rows)}/{total} files in {elapsed:.1f}s  ({errors} errors)")
    return meta_rows, seg_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="TenVAD pipeline")
    parser.add_argument("dataset", help="Dataset name (derives manifests/{name}.parquet, etc.)")
    parser.add_argument("--hop-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("-w", "--workers", type=int, default=None,
                        help="Parallel workers (default: all CPUs)")
    args = parser.parse_args()

    set_seeds(42)

    ds = get_dataset_paths(args.dataset)
    print(f"Dataset:  {args.dataset}")
    print(f"Manifest: {ds.manifest}")
    print(f"Output:   {ds.output}")
    print(f"Metadata: {ds.metadata}")

    # Resolve workers
    workers = args.workers or mp.cpu_count()
    print(f"Workers:  {workers}")

    # Load manifest
    manifest_df = load_manifest(ds.manifest)
    all_paths = manifest_df.select("path").to_series().to_list()
    wavs = [Path(p) for p in all_paths]
    print(f"Files:    {len(wavs)}")

    if not wavs:
        print("Nothing to process.")
        sys.exit(0)

    out_dir = ds.metadata
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Resume: load previous results ----
    meta_path = out_dir / "metadata.parquet"
    prev_meta_parts: list[pl.DataFrame] = []
    prev_seg_parts: list[pl.DataFrame] = []

    if meta_path.exists():
        prev_meta_parts.append(pl.read_parquet(meta_path))
    ckpt_meta = out_dir / _CKPT_META
    if ckpt_meta.exists():
        prev_meta_parts.append(pl.read_parquet(ckpt_meta))
        print("Found checkpoint from a previous interrupted run.")

    completed_ids: set[str] = set()
    if prev_meta_parts:
        combined = pl.concat(prev_meta_parts).unique(subset=["file_id"], keep="last")
        completed_ids = set(
            combined.filter(pl.col("success")).get_column("file_id").to_list()
        )

    # Load previous segments for completed files
    prev_raw = ds.output / "vad_raw" / "segments.parquet"
    if prev_raw.exists():
        df = pl.read_parquet(prev_raw)
        if "uid" in df.columns:
            df = df.rename({"uid": "file_id"})
        prev_seg_parts.append(df)
    ckpt_segs_p = out_dir / _CKPT_SEGS
    if ckpt_segs_p.exists():
        prev_seg_parts.append(pl.read_parquet(ckpt_segs_p))

    if completed_ids:
        before = len(wavs)
        wavs = [w for w in wavs if w.stem not in completed_ids]
        print(f"Resume: {before - len(wavs)} already done, {len(wavs)} remaining")

    # ---- Run VAD on remaining files ----
    if wavs:
        meta_rows, seg_rows = process_parallel(
            wavs, args.hop_size, args.threshold, workers,
            checkpoint_dir=out_dir,
        )
    else:
        print("All files already processed. Regenerating outputs.")
        meta_rows, seg_rows = [], []

    # ---- Merge with previous results ----
    new_meta_df = pl.DataFrame(meta_rows) if meta_rows else pl.DataFrame()
    new_ids = (
        set(new_meta_df.get_column("file_id").to_list())
        if not new_meta_df.is_empty()
        else set()
    )

    all_meta = [
        df.filter(~pl.col("file_id").is_in(list(new_ids)))
        for df in prev_meta_parts
    ]
    all_meta = [p for p in all_meta if not p.is_empty()]
    if not new_meta_df.is_empty():
        all_meta.append(new_meta_df)

    if not all_meta:
        print("ERROR: no results", file=sys.stderr)
        sys.exit(1)

    meta_df = pl.concat(all_meta).sort("path")

    # Segments: keep previous for files NOT reprocessed, add new
    empty_seg_schema = {
        "file_id": pl.Utf8, "onset": pl.Float64,
        "offset": pl.Float64, "duration": pl.Float64,
    }
    new_seg_df = (
        pl.DataFrame(seg_rows) if seg_rows
        else pl.DataFrame(schema=empty_seg_schema)
    )
    keep_seg = [
        df.filter(~pl.col("file_id").is_in(list(new_ids)))
        for df in prev_seg_parts
    ]
    all_seg = [p for p in keep_seg if not p.is_empty()]
    if not new_seg_df.is_empty():
        all_seg.append(new_seg_df)

    seg_df = (
        pl.concat(all_seg).sort("file_id", "onset")
        if all_seg
        else pl.DataFrame(schema=empty_seg_schema)
    )

    # ---- Save metadata.parquet (atomically) ----
    atomic_write_parquet(meta_df, meta_path)
    print(f"Saved {meta_path}  ({len(meta_df)} rows)")

    # ---- Summary ----
    ok = meta_df.filter(pl.col("success"))
    fail = meta_df.filter(~pl.col("success"))
    print(f"\nSuccess: {len(ok)}/{len(meta_df)}")
    if not fail.is_empty():
        print(f"Failed:  {len(fail)}", file=sys.stderr)
        for row in fail.head(5).iter_rows(named=True):
            print(f"  {Path(row['path']).name}: {row['error']}", file=sys.stderr)

    # ---- Write standardised vad_raw / vad_merged parquets ----
    # Standardise column name: file_id → uid
    vad_raw_df = seg_df.rename({"file_id": "uid"})

    vad_raw_dir = ds.output / "vad_raw"
    vad_raw_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(vad_raw_df, vad_raw_dir / "segments.parquet")
    print(f"Saved {vad_raw_dir / 'segments.parquet'}  ({len(vad_raw_df)} rows)")

    vad_merged_df = merge_segments_df(vad_raw_df)
    vad_merged_dir = ds.output / "vad_merged"
    vad_merged_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(vad_merged_df, vad_merged_dir / "segments.parquet")
    print(f"Saved {vad_merged_dir / 'segments.parquet'}  ({len(vad_merged_df)} rows)")

    # Clean up checkpoint files after all writes succeed
    _clear_checkpoint(out_dir)


if __name__ == "__main__":
    main()
