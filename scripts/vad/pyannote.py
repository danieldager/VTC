# TODO: speed up start up time !

import os
import sys
import time
import argparse
import random
import numpy as np
import polars as pl
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

from scripts.vad.utils import get_task_shard


class AudioDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            waveform, sample_rate = torchaudio.load(str(path))
            return {
                "success": True,
                "path": path,
                "waveform": waveform,
                "sample_rate": sample_rate,
            }
        except Exception as e:
            print(f"Error loading {path}: {e}", file=sys.stderr)
            return {"success": False, "path": path, "error": str(e)}


def load_env():
    """Load .env file if it exists."""
    env_file = Path(__file__).parent.parent / ".env"
    print(f"Loading environment from {env_file}...")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())
                print(f"  Loaded {key.strip()}")


def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Seeds set to {seed}")


def setup_model_and_pipeline(hf_token: str, device: torch.device):
    """Load model and instantiate VAD pipeline."""

    print("\nLoading model from HuggingFace...")
    model = Model.from_pretrained("pyannote/segmentation-3.0", token=hf_token)
    model = model.to(device)  # type: ignore

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # Enable TF32 for faster computation on Ampere+ GPUs (compute capability >= 8.0)
        gpu_capability = torch.cuda.get_device_capability(0)
        if gpu_capability[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("TF32 enabled (Ampere+ GPU detected)")

    pipeline = VoiceActivityDetection(segmentation=model)
    pipeline.instantiate(
        {
            "min_duration_on": 0.0,  # remove speech regions shorter than this
            "min_duration_off": 0.0,  # fill non-speech regions shorter than this
        }
    )

    print("Pipeline ready")

    return pipeline


def log_progress(
    i: int,
    total: int,
    start_time: float,
    recent_times: list[float],
) -> None:
    """Log processing progress with rolling average rate and ETA."""
    elapsed = time.time() - start_time

    # Use rolling average if we have enough samples
    if len(recent_times) >= 100:
        rolling_rate = len(recent_times) / sum(recent_times)
    else:
        rolling_rate = i / elapsed if elapsed > 0 else 0

    files_remaining = total - i
    seconds_remaining = files_remaining / rolling_rate if rolling_rate > 0 else 0

    # Format ETA
    if seconds_remaining < 3600:
        eta_str = f"ETA {seconds_remaining / 60:.0f}m"
    else:
        eta_str = f"ETA {seconds_remaining / 3600:.1f}h"

    progress_line = f"[{i:>6}/{total}] {rolling_rate:.1f} files/s | {eta_str}"

    print(progress_line)


def get_log_interval(files_processed: int) -> int:
    """Get adaptive reporting interval based on progress."""
    if files_processed < 10000:
        return 1000
    elif files_processed < 50000:
        return 5000
    elif files_processed < 100000:
        return 10000
    else:
        return 20000


def compute_segment_stats(durations: list[float]) -> dict:
    if not durations:
        return {"max": 0.0, "min": 0.0, "avg": 0.0, "sum": 0.0, "num": 0}

    total = sum(durations)
    count = len(durations)

    return {
        "max": max(durations),
        "min": min(durations),
        "sum": total,
        "num": count,
        "avg": total / count,
    }


def compute_nospch_durations(
    speech_segments: list[list[float]], duration: float
) -> list[float]:
    if not speech_segments:
        return [duration]

    gaps = []
    # Gap before first speech
    if speech_segments[0][0] > 0.001:  # 1ms buffer for precision
        gaps.append(speech_segments[0][0])

    # Gaps between segments
    for i in range(len(speech_segments) - 1):
        gap = speech_segments[i + 1][0] - speech_segments[i][1]
        if gap > 0.001:
            gaps.append(gap)

    # Gap after last speech
    last_end = speech_segments[-1][1]
    if last_end < duration - 0.001:
        gaps.append(duration - last_end)

    return gaps


""" Core function """


def create_error_record(path: str, error: str) -> dict:
    """Create an error record with consistent schema for Parquet."""
    return {
        "success": False,
        "path": path,
        "file_id": Path(path).stem,
        "duration": 0.0,
        "original_sr": 0,
        "speech_ratio": 0.0,
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
        "speech_durations": [],
        "nospch_durations": [],
        "speech_segments": [],
        "speech_segments_detailed": [],
        "error": error,
    }


def process_single_file(audio_dict: dict, pipeline: VoiceActivityDetection) -> dict:
    """Process single audio file and return flat dict ready for Parquet storage."""

    if not audio_dict["success"]:
        return create_error_record(audio_dict["path"], audio_dict["error"])

    # Decompose audio dict
    path = audio_dict["path"]  # Keep as string for Parquet compatibility
    waveform = audio_dict["waveform"]
    sample_rate = audio_dict["sample_rate"]
    duration = waveform.shape[1] / sample_rate

    # If duration is too short
    # if duration <= 1.0:
    #     error_msg = f"Error: {path}, Only {duration} seconds long)"
    #     print(error_msg, file=sys.stderr)
    #     return create_error_record(path, error_msg)

    # Check VAD
    # Pyannote pipeline handles resampling internally
    # if provided sample_rate != model sample_rate (16k)
    vad = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    # Extract segments and durations
    speech_segments = [[s.start, s.end] for s in vad.get_timeline().support()]
    speech_durations = [s[1] - s[0] for s in speech_segments]
    nospch_durations = compute_nospch_durations(speech_segments, duration)

    # Calculate statistics
    speech_stats = compute_segment_stats(speech_durations)
    nospch_stats = compute_segment_stats(nospch_durations)
    speech_ratio = speech_stats["sum"] / duration

    speech_segments_detailed = []
    for segment in speech_segments:
        start_sec = segment[0]
        end_sec = segment[1]
        start_frames = int(start_sec * sample_rate)
        end_frames = int(end_sec * sample_rate)
        dur_sec = end_sec - start_sec
        dur_frames = int(dur_sec * sample_rate)
        speech_segments_detailed.append(
            {
                "start_sec": start_sec,
                "start_frames": start_frames,
                "end_sec": end_sec,
                "end_frames": end_frames,
                "duration_sec": dur_sec,
                "duration_frames": dur_frames,
            }
        )

    # Return flat dict for direct Parquet serialization
    return {
        "success": True,
        "path": path,
        "file_id": Path(path).stem,
        "duration": duration,
        "original_sr": sample_rate,
        "speech_ratio": speech_ratio,
        "speech_durations": speech_durations,
        "nospch_durations": nospch_durations,
        "speech_segments": speech_segments_detailed,
        # "speech_segments_detailed": speech_segments_detailed,
        **{f"speech_{k}": v for k, v in speech_stats.items()},
        **{f"nospch_{k}": v for k, v in nospch_stats.items()},
        "error": "",
    }


def process_files(
    dataloader: DataLoader,
    pipeline: VoiceActivityDetection,
    metadata_path: Path,
    task_id: int,
    chunk_size: int,
) -> int:
    results = []
    errors = 0
    fragment_id = 0
    flush_interval = 10000
    start_time = time.time()

    # Create a subfolder for this specific task
    task_path = metadata_path / f"task_{task_id}"
    task_path.mkdir(parents=True, exist_ok=True)

    # Check for existing fragments and skip ahead
    existing = list(task_path.glob("fragment_*.parquet"))
    if existing:
        max_id = max(int(f.stem.split("_")[1]) for f in existing) + 1
        files_to_skip = max_id * flush_interval
    else:
        files_to_skip = 0
    print(f"Skipping {files_to_skip} files")

    with torch.no_grad():
        for i, audio_dict in enumerate(dataloader, 1):
            # Skip already processed files
            if i <= files_to_skip:
                continue

            record = process_single_file(audio_dict, pipeline)
            results.append(record)
            if not record["success"]:
                errors += 1

            # --- PERIODIC FLUSH ---
            if len(results) >= flush_interval:
                fragment_path = task_path / f"fragment_{fragment_id}.parquet"
                pl.DataFrame(results).write_parquet(fragment_path, compression="zstd")
                results = []  # Clear RAM
                fragment_id += 1

            if i % get_log_interval(i) == 0 or i == chunk_size:
                log_progress(i, chunk_size, start_time, [])

    # Write any remaining results
    if results:
        fragment_path = task_path / f"fragment_{fragment_id}.parquet"
        pl.DataFrame(results).write_parquet(fragment_path, compression="zstd")

    # Merge task fragments into one task-level parquet
    merge_fragments(task_path, metadata_path / f"shard_{task_id}.parquet")

    return errors


def merge_fragments(fragment_dir: Path, final_path: Path) -> None:
    """Combines fragment_0, fragment_1... into shard_N.parquet."""
    fragment_files = sorted(
        list(fragment_dir.glob("fragment_*.parquet")),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    pl.read_parquet(fragment_files).write_parquet(final_path, compression="zstd")
    for f in fragment_files:
        f.unlink()
    fragment_dir.rmdir()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAD Pipeline for audio processing")
    parser.add_argument("manifest", type=str, help="Path to the manifest .parquet file")
    parser.add_argument("--array-id", type=int, default=0, help="SLURM array task ID")
    parser.add_argument("--array-count", type=int, default=1, help="Total array tasks")
    args = parser.parse_args()

    print(f"Task {args.array_id}/{int(args.array_count) - 1} starting")

    # Load environment
    load_env()

    # Set seeds for reproducibility
    set_seeds()

    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise RuntimeError(
            "HF_TOKEN not found. Copy .env.example to .env and add token, "
            "or: export HF_TOKEN='your_token_here'"
        )

    # Setup model and pipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = setup_model_and_pipeline(hf_token, device)

    # Get dataset shard by array task ID
    total_files, chunk_size, paths = get_task_shard(
        args.manifest, args.array_id, args.array_count
    )
    print(f"Processing {chunk_size}/{total_files} files")

    # Create dataset and dataloader objects
    dataset = AudioDataset(paths)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=None, pin_memory=True)

    # Setup output directory
    metadata_path = Path("metadata") / Path(args.manifest).stem / "pyannote"
    metadata_path.mkdir(parents=True, exist_ok=True)

    # Process files
    errors = process_files(
        dataloader,
        pipeline,
        metadata_path,
        args.array_id,
        chunk_size,
    )

    if errors > 0:
        print(f"WARNING: {errors} errors encountered", file=sys.stderr)

    print(f"Task {args.array_id} complete")
