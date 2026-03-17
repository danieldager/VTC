#!/usr/bin/env python3
"""
Generate a normalized manifest CSV from a directory of audio files.

Scans the given directory recursively and writes manifests/{name}.csv with:
  - path: absolute path to the audio file  (first column, as expected by the pipeline)
  - uid:  filename stem without extension
  - ext:  audio format extension (without leading dot, lowercased)

The output is already in normalized format (absolute paths in the 'path'
column), so it can be fed directly to any pipeline subprocess or to
``src.pipeline.normalize`` which will treat it as a no-op.

Usage:
    python scripts/make_manifest.py /data/my_audio/
    python scripts/make_manifest.py /data/my_audio/ -name my_dataset
"""

import argparse
import csv
import os
import sys
from pathlib import Path

# All audio extensions that the pipeline can consume.
AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {
        "wav",
        "flac",
        "mp3",
        "ogg",
        "opus",
        "m4a",
        "aac",
        "aiff",
        "aif",
        "wma",
    }
)

# Manifest directory is resolved relative to this script's repo root.
REPO_ROOT = Path(__file__).parent.parent


def scan_audio_files(root: Path) -> list[tuple[str, str, str]]:
    """Recursively scan *root* and return ``(abs_path, uid, ext)`` tuples.

    Uses ``os.walk`` directly for maximum throughput on large directory trees.
    Files are returned in the order the OS yields them (typically inode order).
    """
    results: list[tuple[str, str, str]] = []
    root_str = str(root)
    for dirpath, _dirnames, filenames in os.walk(root_str):
        for fname in filenames:
            # Split on the last dot to get stem + extension.
            dot_idx = fname.rfind(".")
            if dot_idx <= 0:
                # No extension, or starts with a dot (hidden file) — skip.
                continue
            ext_lower = fname[dot_idx + 1 :].lower()
            if ext_lower not in AUDIO_EXTENSIONS:
                continue
            uid = fname[:dot_idx]
            abs_path = os.path.join(dirpath, fname)
            results.append((abs_path, uid, ext_lower))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a normalized manifest CSV from a directory of audio files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/make_manifest.py /data/my_audio/\n"
            "  python scripts/make_manifest.py /data/my_audio/ -name my_dataset\n"
        ),
    )
    parser.add_argument("directory", help="Root directory to scan for audio files.")
    parser.add_argument(
        "-name",
        dest="name",
        default=None,
        metavar="NAME",
        help=(
            "Dataset name — determines the output path (manifests/{name}.csv). "
            "Defaults to the name of the given directory."
        ),
    )
    args = parser.parse_args()

    root = Path(args.directory).resolve()
    if not root.is_dir():
        print(f"ERROR: Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    name = args.name if args.name else root.name

    print(f"Scanning : {root}")
    rows = scan_audio_files(root)
    print(f"Found    : {len(rows)} audio file(s)")

    if not rows:
        print("WARNING: No audio files found — manifest will be empty.", file=sys.stderr)

    # ------------------------------------------------------------------
    # Detect and log duplicate UIDs (same filename stem, different paths).
    # ------------------------------------------------------------------
    uid_to_paths: dict[str, list[str]] = {}
    for path, uid, _ in rows:
        uid_to_paths.setdefault(uid, []).append(path)

    duplicates = {uid: paths for uid, paths in uid_to_paths.items() if len(paths) > 1}
    if duplicates:
        n_dup_uids = len(duplicates)
        n_dup_files = sum(len(v) for v in duplicates.values())
        print(
            f"WARNING: {n_dup_uids} duplicate uid(s) across {n_dup_files} files "
            f"(same filename stem in multiple directories — all kept):",
            file=sys.stderr,
        )
        for uid in sorted(duplicates):
            paths = duplicates[uid]
            print(f"  uid={uid!r} ({len(paths)} files):", file=sys.stderr)
            for p in paths:
                print(f"    {p}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Write manifest.
    # ------------------------------------------------------------------
    manifests_dir = REPO_ROOT / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    output = manifests_dir / f"{name}.csv"

    if output.exists():
        print(f"WARNING: Overwriting existing manifest: {output}", file=sys.stderr)

    with open(output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "uid", "ext"])
        for path, uid, ext in rows:
            writer.writerow([path, uid, ext])

    print(f"Manifest : {output}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
