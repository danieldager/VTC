"""Parallel VAD driver — process pool with spawn context and progress logging."""

from __future__ import annotations

import queue as _queue
import sys
import time
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.core.checkpoint import save_checkpoint
from src.core.vad_processing import process_vad_file


def _log_bytes_progress(
    progress_q: mp.Queue,
    in_progress_samples: dict,
    file_sizes: dict,
    completed_ids: set,
    completed: int,
    total: int,
    t0: float,
) -> None:
    """Drain the progress queue and print GB-level progress across all in-flight files."""
    while True:
        try:
            file_id, samples = progress_q.get_nowait()
            if file_id not in completed_ids:
                in_progress_samples[file_id] = samples
        except _queue.Empty:
            break

    # bytes done = completed files + partial progress in in-flight files
    # int16 audio = 2 bytes/sample at TARGET_SR (after resampling)
    done_bytes = sum(file_sizes.get(fid, 0) for fid in completed_ids)
    inflight_bytes = sum(s * 2 for s in in_progress_samples.values())
    total_bytes = sum(file_sizes.values()) or 1
    all_done_bytes = min(done_bytes + inflight_bytes, total_bytes)

    gb = 1e9
    elapsed = time.time() - t0
    rate = all_done_bytes / elapsed if elapsed > 0 else 0
    rem = (total_bytes - all_done_bytes) / rate if rate > 0 else 0
    eta = f"{rem / 60:.0f}m" if rem < 3600 else f"{rem / 3600:.1f}h"
    pct = 100.0 * all_done_bytes / total_bytes
    print(
        f"  VAD  {completed:>4}/{total}"
        f"  {all_done_bytes/gb:.1f}/{total_bytes/gb:.1f} GB"
        f" ({pct:.0f}%)  ETA {eta}",
        flush=True,
    )


def run_vad_parallel(
    wavs: list[Path],
    hop_size: int,
    threshold: float,
    workers: int,
    checkpoint_dir: Path | None = None,
    checkpoint_interval: int = 5000,
    save_probs_dir: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """Run VAD on all files using a process pool.

    Uses ``mp.get_context("spawn")`` so that worker processes start
    fresh — no inherited ``torch`` pages from the parent.

    If *save_probs_dir* is given, each worker saves a compressed
    ``.npz`` file with the raw per-frame speech probabilities.

    Returns ``(metadata_rows, segment_rows)``.
    """
    # Pre-stat all files so we can track GB-level progress.
    file_sizes: dict[str, int] = {}
    for w in wavs:
        try:
            file_sizes[w.stem] = w.stat().st_size
        except OSError:
            file_sizes[w.stem] = 0
    total_gb = sum(file_sizes.values()) / 1e9
    print(
        f"  data     : {total_gb:.1f} GB across {len(wavs)} files",
        flush=True,
    )

    meta_rows: list[dict] = []
    seg_rows: list[dict] = []
    errors = 0
    total = len(wavs)
    t0 = time.time()
    # Log roughly every 5 % of files (at least every 1).
    log_every = max(1, total // 20)

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    progress_q = manager.Queue()  # workers → parent sample-count updates
    probs_dir_str = str(save_probs_dir) if save_probs_dir else None
    tasks = [(w, hop_size, threshold, progress_q, probs_dir_str) for w in wavs]

    print(
        f"Spawning {workers} workers for {total} files",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = {pool.submit(process_vad_file, t): t[0] for t in tasks}
        print(
            f"All {total} tasks submitted, waiting for results",
            flush=True,
        )

        completed = 0
        completed_ids: set[str] = set()
        in_progress_samples: dict[str, int] = {}
        last_heartbeat = time.time()
        heartbeat_interval = 300  # seconds

        for future in as_completed(futures):
            path = futures[future]
            file_id = Path(path).stem
            try:
                meta, segs = future.result()
                meta_rows.append(meta)
                seg_rows.extend(segs)
                if meta.get("success", False):
                    completed_ids.add(file_id)
                    in_progress_samples.pop(file_id, None)
                else:
                    errors += 1
                    print(
                        f"  WARN  {Path(path).stem}: " f"{meta['error']}",
                        file=sys.stderr,
                        flush=True,
                    )
            except Exception as e:
                errors += 1
                print(
                    f"  ERROR {Path(path).stem}: {e}",
                    file=sys.stderr,
                    flush=True,
                )

            completed += 1
            now = time.time()
            should_log = (
                completed % log_every == 0
                or completed == total
                or (now - last_heartbeat) >= heartbeat_interval
            )
            if should_log:
                _log_bytes_progress(
                    progress_q,  # type: ignore
                    in_progress_samples,
                    file_sizes,
                    completed_ids,
                    completed,
                    total,
                    t0,
                )
                last_heartbeat = now

            if checkpoint_dir and completed % checkpoint_interval == 0:
                save_checkpoint(checkpoint_dir, meta_rows, seg_rows)

    elapsed = time.time() - t0
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(
        f"Processed {len(meta_rows)}/{total} files "
        f"in {h:02d}:{m:02d}:{s:02d}  ({errors} errors)"
    )
    return meta_rows, seg_rows
