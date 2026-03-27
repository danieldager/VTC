"""
GPU resource detection and automatic batch-size / array-count allocation.

This module is designed to run on the login node (no torch import required)
by querying SLURM node info via ``sinfo``.  At runtime on a compute node,
it can also query ``torch.cuda`` directly.

Exported API
------------
GPUSpec(name, vram_gb, count_per_node)
    Dataclass describing the GPUs on a node.

query_partition_gpus(partitions) → GPUSpec | None
    Ask SLURM what GPU hardware is available.

query_local_gpu() → GPUSpec | None
    Query the GPU on the current node (compute-node only, needs torch).

recommend_vtc_batch_size(vram_gb) → int
    Map VRAM to a batch size for segma's ``apply_model_on_audio``.

recommend_esc_batch_size(vram_gb) → int
    Map VRAM to a PANN window-batch size (number of 10 s windows per call).

DatasetStats
    Lightweight stats about a manifest's files.

gather_dataset_stats(paths) → DatasetStats
    Scan file sizes and count.

ResourcePlan
    The final allocation: batch sizes, array counts, partition.

plan_resources(dataset_stats, gpu_spec) → ResourcePlan
    Combine dataset stats + GPU spec into concrete SLURM parameters.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field


# ──────────────────────────────────────────────────────────────────
# GPU specs
# ──────────────────────────────────────────────────────────────────

# Known VRAM per GPU model (GiB).  Add entries as new hardware appears.
_KNOWN_VRAM: dict[str, int] = {
    "A40": 48,
    "A100": 80,
    "H100": 94,
    "H100_NVL": 94,
    "V100": 32,
    "L40": 48,
    "L40S": 48,
    "A30": 24,
    "T4": 16,
    "RTX_3090": 24,
    "RTX_4090": 24,
}


@dataclass
class GPUSpec:
    """Description of GPUs available on a node."""

    name: str  # e.g. "A40", "H100"
    vram_gb: int  # per-GPU VRAM in GiB
    count: int  # total GPUs on the node


def _normalise_gpu_name(raw: str) -> str:
    """Normalise  'NVIDIA H100 NVL' → 'H100_NVL', 'gpu:A40:10' → 'A40'."""
    raw = raw.upper().replace("NVIDIA ", "").replace(" ", "_")
    for known in _KNOWN_VRAM:
        if known in raw:
            return known
    return raw


def query_partition_gpus(partitions: list[str]) -> GPUSpec | None:
    """Query SLURM ``sinfo`` for the GPUs across the requested partitions.

    Since SLURM jobs submitted to multiple partitions (e.g.
    ``--partition=erc-dupoux,gpu-p1``) can land on any of them, this
    returns an aggregate view:

    *  ``count`` = sum of GPUs across all matching partitions (total pool).
    *  ``vram_gb`` = *minimum* VRAM seen (conservative for batch sizing).
    *  ``name`` = name of the GPU with the lowest VRAM (the bottleneck).

    Returns ``None`` if SLURM is unavailable or no GPU partition matches.
    """
    if not shutil.which("sinfo"):
        return None

    try:
        out = subprocess.run(
            ["sinfo", "-o", "%P %G"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    # Build a lookup from sinfo output: partition → gres string
    sinfo_map: dict[str, str] = {}
    for line in out.stdout.strip().splitlines()[1:]:  # skip header
        parts = line.split()
        if len(parts) < 2:
            continue
        partition = parts[0].rstrip("*")
        sinfo_map[partition] = parts[1]

    # Collect specs from all matching partitions.
    # De-duplicate by GPU model (e.g. erc-dupoux and gpu-p2 share puck7).
    seen_models: dict[str, tuple[int, int]] = {}  # name → (vram, count)
    for partition in partitions:
        gres = sinfo_map.get(partition)
        if not gres:
            continue
        m = re.match(r"gpu:(\w+):(\d+)", gres)
        if not m:
            continue
        gpu_model = m.group(1)
        gpu_count = int(m.group(2))
        name = _normalise_gpu_name(gpu_model)
        vram = _KNOWN_VRAM.get(name, 16)
        # Keep the highest GPU count seen for this model (partitions may
        # overlap on the same node)
        if name not in seen_models or gpu_count > seen_models[name][1]:
            seen_models[name] = (vram, gpu_count)

    if not seen_models:
        return None

    # Aggregate: total GPU count, minimum VRAM for conservative batch sizing
    total_count = sum(count for _, count in seen_models.values())
    min_vram_name = min(seen_models, key=lambda n: seen_models[n][0])
    min_vram = seen_models[min_vram_name][0]

    return GPUSpec(name=min_vram_name, vram_gb=min_vram, count=total_count)


def query_local_gpu() -> GPUSpec | None:
    """Query the GPU on the *current* node via ``torch.cuda``.

    Only works on a compute node where torch + CUDA are available.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        name_raw = torch.cuda.get_device_name(0)
        name = _normalise_gpu_name(name_raw)
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_mem // (1024**3)
        count = torch.cuda.device_count()
        return GPUSpec(name=name, vram_gb=vram_gb, count=count)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────
# Batch-size recommendations
# ──────────────────────────────────────────────────────────────────

# VTC (segma) batch-size thresholds.
# segma chunks audio into 4 s windows and processes ``batch_size`` of them
# at once.  Larger batch → higher throughput but more VRAM.
# These were derived empirically on H100 NVL and A40.
_VTC_BATCH_TABLE: list[tuple[int, int]] = [
    # (min_vram_gb, batch_size)
    (80, 512),   # H100 NVL / A100-80GB — plenty of headroom
    (48, 256),   # A40 / L40 — comfortable
    (32, 192),   # V100-32GB
    (24, 128),   # A30 / RTX 3090 / RTX 4090
    (16, 64),    # T4 / smaller
    (0, 32),     # fallback
]


def recommend_vtc_batch_size(vram_gb: int) -> int:
    """Return a recommended VTC (segma) batch size for the given VRAM."""
    for min_vram, bs in _VTC_BATCH_TABLE:
        if vram_gb >= min_vram:
            return bs
    return 32


# PANNs ESC — currently processes one window at a time,
# but we can batch multiple 10 s windows in a single forward pass.
_NOISE_BATCH_TABLE: list[tuple[int, int]] = [
    (80, 64),
    (48, 32),
    (24, 16),
    (16, 8),
    (0, 4),
]


def recommend_esc_batch_size(vram_gb: int) -> int:
    """Return a recommended PANNs batch size for the given VRAM."""
    for min_vram, bs in _NOISE_BATCH_TABLE:
        if vram_gb >= min_vram:
            return bs
    return 4


# ──────────────────────────────────────────────────────────────────
# Dataset statistics
# ──────────────────────────────────────────────────────────────────

@dataclass
class DatasetStats:
    """Lightweight statistics about a set of audio files."""

    n_files: int = 0
    n_found: int = 0
    n_missing: int = 0
    total_bytes: int = 0
    min_bytes: int = 0
    max_bytes: int = 0
    mean_bytes: float = 0.0
    median_bytes: float = 0.0
    file_sizes: list[int] = field(default_factory=list, repr=False)


def gather_dataset_stats(paths: list[str]) -> DatasetStats:
    """Stat each path and return aggregate statistics."""
    sizes: list[int] = []
    missing = 0
    for p in paths:
        try:
            sizes.append(os.path.getsize(p))
        except OSError:
            missing += 1

    if not sizes:
        return DatasetStats(n_files=len(paths), n_missing=missing)

    sizes.sort()
    n = len(sizes)
    mid = n // 2
    median = sizes[mid] if n % 2 else (sizes[mid - 1] + sizes[mid]) / 2

    return DatasetStats(
        n_files=len(paths),
        n_found=n,
        n_missing=missing,
        total_bytes=sum(sizes),
        min_bytes=sizes[0],
        max_bytes=sizes[-1],
        mean_bytes=sum(sizes) / n,
        median_bytes=median,
        file_sizes=sizes,
    )


# ──────────────────────────────────────────────────────────────────
# Resource plan
# ──────────────────────────────────────────────────────────────────

@dataclass
class ResourcePlan:
    """Concrete resource allocation for a pipeline run."""

    gpu_name: str
    gpu_vram_gb: int
    gpus_available: int

    vtc_batch_size: int
    vtc_array_count: int

    snr_array_count: int
    esc_array_count: int

    notes: list[str] = field(default_factory=list)


# Minimum bytes per shard — below this, shards become I/O bound because
# model loading dominates.  ~500 MB is a reasonable floor.
_MIN_BYTES_PER_SHARD = 500 * 1024 * 1024  # 500 MB

# Minimum files per shard — if there are very few files (< 5), sharding
# is pointless and adds overhead.
_MIN_FILES_PER_SHARD = 5


def plan_resources(
    stats: DatasetStats,
    gpu: GPUSpec | None,
    *,
    max_vtc_shards: int | None = None,
    max_gpu_shards: int | None = None,
) -> ResourcePlan:
    """Combine dataset stats + GPU spec into a resource plan.

    Parameters
    ----------
    stats : DatasetStats
        Output of ``gather_dataset_stats``.
    gpu : GPUSpec | None
        Output of ``query_partition_gpus`` or ``query_local_gpu``.
        If ``None``, falls back to conservative defaults.
    max_vtc_shards : int | None
        Upper bound for VTC array count (default: gpu.count).
    max_gpu_shards : int | None
        Upper bound for SNR/ESC array count (default: gpu.count).
    """
    notes: list[str] = []

    if gpu is None:
        notes.append("No GPU info available — using conservative defaults")
        return ResourcePlan(
            gpu_name="unknown",
            gpu_vram_gb=0,
            gpus_available=0,
            vtc_batch_size=128,
            vtc_array_count=2,
            snr_array_count=2,
            esc_array_count=2,
            notes=notes,
        )

    vtc_bs = recommend_vtc_batch_size(gpu.vram_gb)
    notes.append(f"VTC batch_size={vtc_bs} for {gpu.name} ({gpu.vram_gb} GB)")

    # ---- Array count logic -------------------------------------------
    # Principle: use as many GPUs as sensible, but avoid the small-file
    # problem where model-loading overhead dominates.

    gpu_budget = gpu.count
    if max_vtc_shards is not None:
        gpu_budget = min(gpu_budget, max_vtc_shards)
    if max_gpu_shards is not None:
        gpu_budget = min(gpu_budget, max_gpu_shards)

    # VTC is the GPU bottleneck — give it as many GPUs as possible,
    # but cap based on dataset size.
    vtc_max_from_bytes = max(1, stats.total_bytes // _MIN_BYTES_PER_SHARD)
    vtc_max_from_files = max(1, stats.n_found // _MIN_FILES_PER_SHARD)
    vtc_cap = min(vtc_max_from_bytes, vtc_max_from_files)
    vtc_array = min(gpu_budget, vtc_cap)

    if vtc_array < gpu_budget:
        notes.append(
            f"VTC shards capped at {vtc_array} "
            f"(dataset too small for {gpu_budget} GPUs)"
        )

    # SNR and ESC are lighter — 2 shards is usually plenty, but
    # scale up for very large datasets.
    secondary_cap = min(vtc_max_from_bytes, vtc_max_from_files)
    snr_array = min(max(1, gpu_budget // 2), secondary_cap)
    esc_array = min(max(1, gpu_budget // 2), secondary_cap)

    # Ensure we don't exceed available GPUs across all concurrent jobs.
    # VTC + SNR + ESC all run in parallel, each requesting 1 GPU per
    # array task.
    total_requested = vtc_array + snr_array + esc_array
    if total_requested > gpu.count:
        # Scale back proportionally, prioritising VTC
        remaining = gpu.count
        vtc_array = min(vtc_array, max(1, remaining // 2))
        remaining -= vtc_array
        snr_array = min(snr_array, max(1, remaining // 2))
        remaining -= snr_array
        esc_array = max(1, remaining)
        notes.append(
            f"Scaled back to fit {gpu.count} GPUs: "
            f"VTC={vtc_array}, SNR={snr_array}, ESC={esc_array}"
        )

    notes.append(f"Total GPU tasks: {vtc_array + snr_array + esc_array} "
                 f"/ {gpu.count} available")

    return ResourcePlan(
        gpu_name=gpu.name,
        gpu_vram_gb=gpu.vram_gb,
        gpus_available=gpu.count,
        vtc_batch_size=vtc_bs,
        vtc_array_count=vtc_array,
        snr_array_count=snr_array,
        esc_array_count=esc_array,
        notes=notes,
    )
