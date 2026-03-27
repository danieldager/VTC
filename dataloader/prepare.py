"""Feature extraction and packaging CLI.

:func:`prepare` submits the full pipeline as SLURM jobs (VAD + VTC + SNR +
Noise in parallel, then Package) and waits for completion.  It is the
complement to :func:`~dataloader.create.create_dataloader`: prepare produces
the shards, create_dataloader streams them.

CLI usage::

    # Run full extraction + packaging, then exit
    python -m dataloader.prepare config.json

    # Only submit jobs — don't wait (fire and forget)
    python -m dataloader.prepare config.json --no-wait

    # Limit to a fraction of the dataset (useful for quick tests)
    python -m dataloader.prepare config.json --sample 0.1

    # Wipe outputs and start clean
    python -m dataloader.prepare config.json --force

Python usage::

    from dataloader import prepare
    prepare("configs/seedlings.json")
    # ... blocks until shards are ready, then hand off to training
    loader = create_dataloader("configs/seedlings.json")

The config's ``pipeline.*`` fields are forwarded to the pipeline scripts:
- vad_threshold   → vad.py --threshold
- vtc_threshold   → vtc.py --threshold
- max_clip_s      → package.py --max_clip
- split_search_s  → package.py --split_search
- audio_fmt       → package.py --audio_fmt
- shard_size      → package.py --shard_size
- target_sr       → package.py --target_sr
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dataloader.config import DatasetConfig

log = logging.getLogger(__name__)

# ── Completion detection ─────────────────────────────────────────────────────


def _shards_exist(output_dir: Path) -> bool:
    return bool(list((output_dir / "shards").glob("*.tar")))


def _stage_done(output_dir: Path, stage: str) -> bool:
    """Return True if *stage* outputs are present."""
    if stage == "vad":
        return (
            (output_dir / "vad_meta" / "metadata.parquet").exists()
            and (output_dir / "vad_merged" / "segments.parquet").exists()
        )
    if stage == "vtc":
        return (
            bool(list((output_dir / "vtc_meta").glob("shard_*.parquet")))
            and bool(list((output_dir / "vtc_merged").glob("shard_*.parquet")))
        )
    if stage == "snr":
        return bool(list((output_dir / "snr_meta").glob("shard_*.parquet")))
    if stage == "esc":
        return bool(list((output_dir / "esc_meta").glob("shard_*.parquet")))
    if stage == "package":
        return _shards_exist(output_dir)
    raise ValueError(f"Unknown stage: {stage!r}")


_ALL_STAGES = ["vad", "vtc", "snr", "esc", "package"]


# ── SLURM helpers ────────────────────────────────────────────────────────────


def _sbatch(args: list[str]) -> str:
    """Submit a SLURM job and return its job ID."""
    result = subprocess.run(
        ["sbatch", "--parsable"] + args,
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip().split(";")[0]  # strip optional cluster name


def _job_alive(job_id: str) -> bool:
    """Return True if the job is still in the SLURM queue."""
    result = subprocess.run(
        ["squeue", "-j", job_id, "-h"],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def _wait_for_jobs(job_ids: list[str], poll_interval: int = 10) -> None:
    """Block until all SLURM jobs are no longer in the queue."""
    pending = set(job_ids)
    while pending:
        finished = {jid for jid in pending if not _job_alive(jid)}
        if finished:
            for jid in sorted(finished):
                log.info("  Job %s finished", jid)
        pending -= finished
        if pending:
            time.sleep(poll_interval)


def _check_jobs_ok(job_ids: list[str], label: str) -> None:
    """Raise if any job did not complete successfully."""
    failed = []
    for jid in job_ids:
        result = subprocess.run(
            ["sacct", "-j", jid, "--format=State", "-n", "-X"],
            capture_output=True, text=True,
        )
        state = result.stdout.strip()
        if state and "COMPLETED" not in state:
            failed.append((jid, state))
    if failed:
        details = ", ".join(f"{jid}={s}" for jid, s in failed)
        raise RuntimeError(
            f"{label} jobs did not complete successfully: {details}"
        )


# ── Core prepare logic ───────────────────────────────────────────────────────


def prepare(
    config: DatasetConfig | str | Path,
    *,
    sample: float | None = None,
    wait: bool = True,
    force: bool = False,
) -> None:
    """Submit SLURM pipeline jobs and optionally wait for completion.

    Reads a :class:`~dataloader.config.DatasetConfig` to determine the
    dataset and extraction parameters, then submits VAD + VTC + SNR + ESC
    jobs in parallel, followed by a Package job that depends on all four.

    Parameters
    ----------
    config:
        A :class:`DatasetConfig` instance, or a path to a JSON config file.
    sample:
        Fraction of dataset to process (e.g. ``0.1`` = 10%). If ``None``,
        processes the full dataset.
    wait:
        If ``True`` (default), block until all jobs finish and raise on
        failure. If ``False``, return immediately after submitting.
    force:
        If ``True``, wipe all existing pipeline outputs first.

    Raises
    ------
    RuntimeError
        If ``wait=True`` and any SLURM job fails.
    FileNotFoundError
        If shards are still missing after pipeline completes (sanity check).
    """
    if isinstance(config, (str, Path)):
        config = DatasetConfig.load(config)

    output_dir = Path(config.dataset_dir).resolve()
    # Dataset name is the name of the output directory (e.g., "seedlings_1")
    dataset = output_dir.name

    # ── Force-clean ───────────────────────────────────────────────────────
    if force:
        log.info("--force: removing existing outputs in %s", output_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Skip if already done ──────────────────────────────────────────────
    if _shards_exist(output_dir) and not force:
        n = len(list((output_dir / "shards").glob("*.tar")))
        log.info("Shards already exist (%d tar files) — nothing to do.", n)
        print(f"[prepare] Already have {n} shard(s) in {output_dir / 'shards'}.")
        print("[prepare] Pass force=True (or --force) to reprocess.")
        return

    # ── Run preflight to get resource hints ───────────────────────────────
    repo_root = Path(__file__).resolve().parents[1]
    extra_args: list[str] = []
    if sample is not None:
        extra_args += ["--sample", str(sample)]

    vtc_batch_size = 128
    vtc_array_count = 2
    snr_array_count = 2
    esc_array_count = 2

    try:
        preflight_result = subprocess.run(
            ["uv", "run", "python", "-m", "src.pipeline.preflight",
             dataset, "--emit-env"] + extra_args,
            capture_output=True, text=True, cwd=repo_root,
        )
        for line in preflight_result.stdout.splitlines():
            if line.startswith("VTC_BATCH_SIZE="):
                vtc_batch_size = int(line.split("=", 1)[1])
            elif line.startswith("VTC_ARRAY_COUNT="):
                vtc_array_count = int(line.split("=", 1)[1])
            elif line.startswith("SNR_ARRAY_COUNT="):
                snr_array_count = int(line.split("=", 1)[1])
            elif line.startswith("ESC_ARRAY_COUNT="):
                esc_array_count = int(line.split("=", 1)[1])
    except Exception as exc:
        log.warning("Preflight failed (%s) — using defaults", exc)

    slurm_dir = repo_root / "slurm"
    logs_dir = repo_root / "logs"
    (logs_dir / "vad").mkdir(parents=True, exist_ok=True)
    (logs_dir / "vtc").mkdir(parents=True, exist_ok=True)
    (logs_dir / "snr").mkdir(parents=True, exist_ok=True)
    (logs_dir / "esc").mkdir(parents=True, exist_ok=True)
    (logs_dir / "package").mkdir(parents=True, exist_ok=True)

    # ── Stages: figure out what needs running ─────────────────────────────
    run_vad   = not _stage_done(output_dir, "vad")
    run_vtc   = not _stage_done(output_dir, "vtc")
    run_snr   = not _stage_done(output_dir, "snr")
    run_esc = not _stage_done(output_dir, "esc")

    print(f"\n[prepare] Dataset : {dataset}")
    print(f"[prepare] Output  : {output_dir}")
    if sample:
        print(f"[prepare] Sample  : {sample * 100:.0f}%")

    p = config.pipeline

    # ── Submit feature jobs (only for incomplete stages) ──────────────────
    dep_job_ids: list[str] = []

    if run_vad:
        vad_jid = _sbatch([
            str(slurm_dir / "vad.slurm"), dataset,
            "--threshold", str(p.vad_threshold),
        ] + extra_args)
        dep_job_ids.append(vad_jid)
        print(f"[prepare]   VAD              : job {vad_jid}")
    else:
        print(f"[prepare]   VAD              : already done, skipping")

    if run_vtc:
        vtc_jid = _sbatch([
            "--array", f"0-{vtc_array_count - 1}",
            str(slurm_dir / "vtc.slurm"), dataset,
            "--threshold", str(p.vtc_threshold),
            "--batch_size", str(vtc_batch_size),
        ] + extra_args)
        dep_job_ids.append(vtc_jid)
        print(f"[prepare]   VTC (×{vtc_array_count} shards) : job {vtc_jid}")
    else:
        print(f"[prepare]   VTC              : already done, skipping")

    if run_snr:
        snr_jid = _sbatch([
            "--array", f"0-{snr_array_count - 1}",
            str(slurm_dir / "snr.slurm"), dataset,
        ] + extra_args)
        dep_job_ids.append(snr_jid)
        print(f"[prepare]   SNR (×{snr_array_count} shards) : job {snr_jid}")
    else:
        print(f"[prepare]   SNR              : already done, skipping")

    if run_esc:
        esc_jid = _sbatch([
            "--array", f"0-{esc_array_count - 1}",
            str(slurm_dir / "esc.slurm"), dataset,
        ] + extra_args)
        dep_job_ids.append(esc_jid)
        print(f"[prepare]   ESC   (×{esc_array_count} shards): job {esc_jid}")
    else:
        print(f"[prepare]   ESC              : already done, skipping")

    # ── Package job (depends on all feature jobs, or immediate if all done) ─
    pkg_cmd = (
        f"set -euo pipefail\n"
        f"module purge && module load ffmpeg\n"
        f"export LD_LIBRARY_PATH=/shared/opt/linux-rocky9-x86_64/gcc-11.4.1/"
        f"ffmpeg-6.1.1-gynsavpssxgp4ewikkmsa6jswfgi3ycg/lib:${{LD_LIBRARY_PATH:-}}\n"
        f"export PYTHONPATH={repo_root}:${{PYTHONPATH:-}}\n"
        f"export POLARS_SKIP_CPU_CHECK=1\n"
        f"PYTHONUNBUFFERED=1 uv run python -m src.pipeline.package {dataset}"
        f" --audio_fmt {p.audio_fmt}"
        f" --max_clip {p.max_clip_s}"
        f" --split_search {p.split_search_s}"
        f" --shard_size {p.shard_size}"
        f" --target_sr {p.target_sr}"
        + (f" --sample {sample}" if sample else "")
    )

    pkg_sbatch_args = [
        "--job-name", "dlpp_package",
        "--output", str(logs_dir / "package" / "pkg_%j.out"),
        "--error",  str(logs_dir / "package" / "pkg_%j.err"),
        "--cpus-per-task", "8",
        "--mem", "64G",
        "--time", "04:00:00",
        "--partition", "erc-dupoux,gpu-p1",
    ]
    if dep_job_ids:
        dep_str = ":".join(dep_job_ids)
        pkg_sbatch_args += ["--dependency", f"afterok:{dep_str}"]

    pkg_sbatch_args += ["--wrap", pkg_cmd]
    pkg_jid = _sbatch(pkg_sbatch_args)
    all_job_ids = dep_job_ids + [pkg_jid]
    print(f"[prepare]   Package          : job {pkg_jid}")

    cancel_cmd = "scancel " + " ".join(all_job_ids)
    print(f"\n[prepare] Monitor : squeue -u $USER")
    print(f"[prepare] Cancel  : {cancel_cmd}")
    print(f"[prepare] Pkg log : {logs_dir / 'package' / f'pkg_{pkg_jid}.out'}")

    if not wait:
        print("[prepare] --no-wait: returning immediately.")
        return

    # ── Wait for everything ───────────────────────────────────────────────
    print(f"\n[prepare] Waiting for {len(all_job_ids)} job(s)...", flush=True)
    _wait_for_jobs(all_job_ids)

    # Check feature jobs
    if dep_job_ids:
        _check_jobs_ok(dep_job_ids, "Feature extraction")

    # Check package job
    _check_jobs_ok([pkg_jid], "Package")

    # Final sanity check
    if not _shards_exist(output_dir):
        raise FileNotFoundError(
            f"Pipeline completed but no shards found in {output_dir / 'shards'}. "
            f"Check logs: {logs_dir / 'package'}"
        )

    n = len(list((output_dir / "shards").glob("*.tar")))
    print(f"\n[prepare] Done — {n} shard(s) ready in {output_dir / 'shards'}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m dataloader.prepare",
        description=(
            "Feature extraction + packaging for a dataset.\n\n"
            "Submits VAD, VTC, SNR, Noise jobs in parallel, then a Package job,\n"
            "all driven by a single DatasetConfig JSON file.\n\n"
            "After prepare completes, use create_dataloader() in your training script."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config",
        help="Path to a DatasetConfig JSON file (or directory containing dataset_config.json).",
    )
    parser.add_argument(
        "--sample", type=float, default=None, metavar="FRAC",
        help="Process only a random fraction of the dataset (e.g. 0.1 = 10%%).",
    )
    parser.add_argument(
        "--no-wait", action="store_true",
        help="Submit jobs and exit immediately (don't poll for completion).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Wipe all existing outputs and reprocess from scratch.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        prepare(
            args.config,
            sample=args.sample,
            wait=not args.no_wait,
            force=args.force,
        )
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _main()
