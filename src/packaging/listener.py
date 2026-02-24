#!/usr/bin/env python3
"""Listen to clips from WebDataset shards — validation tool.

Extracts random clips from tar shards and saves them as individual
audio files alongside their metadata JSON, so you can listen and verify
that the clips contain mostly speech with correct labels.

Usage:
    python -m src.packaging.listener output/seedlings/shards
    python -m src.packaging.listener output/seedlings/shards --n 20 --seed 123
    python -m src.packaging.listener output/seedlings/shards --clip_id 9998_0003
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def extract_clips(
    shard_dir: Path,
    output_dir: Path,
    n: int = 10,
    seed: int = 42,
    clip_ids: list[str] | None = None,
    force_wav: bool = False,
) -> list[Path]:
    """Extract clips from shards into individual files.

    Returns paths to the extracted audio files.
    """
    import webdataset as wds

    shard_dir = Path(shard_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tar_files = sorted(shard_dir.glob("*.tar"))
    if not tar_files:
        print(f"ERROR: No .tar files in {shard_dir}", file=sys.stderr)
        sys.exit(1)

    # Load all samples (lazy — just keys + data)
    url = str(shard_dir / "shards-{000000..999999}.tar")
    all_samples: list[dict] = []

    for tar_path in tar_files:
        ds = wds.WebDataset(str(tar_path))  # type: ignore[attr-defined]
        for sample in ds:
            key = sample["__key__"]
            all_samples.append({"key": key, "sample": sample})

    print(f"Found {len(all_samples)} clips across {len(tar_files)} shards")

    # Filter or sample
    if clip_ids:
        selected = [s for s in all_samples if s["key"] in set(clip_ids)]
        if not selected:
            print(f"ERROR: None of {clip_ids} found", file=sys.stderr)
            sys.exit(1)
    else:
        random.seed(seed)
        selected = random.sample(all_samples, min(n, len(all_samples)))

    # Extract
    extracted: list[Path] = []
    for item in selected:
        key = item["key"]
        sample = item["sample"]

        # Find the audio key
        audio_ext = None
        for ext in ("flac", "wav"):
            if ext in sample:
                audio_ext = ext
                break

        if audio_ext is None:
            print(f"  WARN: no audio in {key}, skipping")
            continue

        # Convert to WAV if requested (VS Code can't play FLAC)
        if force_wav and audio_ext != "wav":
            import io
            import soundfile as sf
            data, sr = sf.read(io.BytesIO(sample[audio_ext]))
            audio_path = output_dir / f"{key}.wav"
            sf.write(str(audio_path), data, sr, format="WAV", subtype="PCM_16")
        else:
            audio_path = output_dir / f"{key}.{audio_ext}"
            audio_path.write_bytes(sample[audio_ext])

        # Write metadata
        if "json" in sample:
            meta = json.loads(sample["json"])
            meta_path = output_dir / f"{key}.json"
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

        extracted.append(audio_path)
        meta = json.loads(sample["json"]) if "json" in sample else {}
        dur = meta.get("duration", "?")
        density = meta.get("speech_density", "?")
        n_vad = len(meta.get("vad_segments", []))
        n_vtc = len(meta.get("vtc_segments", []))
        print(f"  {key}  dur={dur}s  density={density}  "
              f"vad={n_vad}  vtc={n_vtc}")

    print(f"\nExtracted {len(extracted)} clips to {output_dir}/")
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract clips from WebDataset shards for listening/validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "shard_dir",
        help="Path to the shards directory (e.g. output/seedlings/shards).",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory for extracted clips (default: {shard_dir}/samples/).",
    )
    parser.add_argument(
        "-n", type=int, default=10,
        help="Number of random clips to extract (default: 10).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for clip selection.",
    )
    parser.add_argument(
        "--clip_id", nargs="+", default=None,
        help="Extract specific clip(s) by ID.",
    )
    parser.add_argument(
        "--wav", action="store_true",
        help="Convert to WAV on extraction (VS Code can play .wav but not .flac).",
    )
    args = parser.parse_args()

    shard_dir = Path(args.shard_dir)
    output_dir = Path(args.output) if args.output else shard_dir / "samples"

    extract_clips(
        shard_dir=shard_dir,
        output_dir=output_dir,
        n=args.n,
        seed=args.seed,
        clip_ids=args.clip_id,
        force_wav=args.wav,
    )


if __name__ == "__main__":
    main()
