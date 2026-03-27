"""Convenience functions for building filtered manifests from pipeline outputs.

The main entry point is :func:`build_manifest`, which:

1. Instantiates adapters for each pipeline stage
2. Joins their metadata manifests via :class:`ManifestJoiner`
3. Optionally applies a :class:`FilterConfig` to select a subset

Example
-------
::

    from dataloader import build_manifest, FilterConfig

    manifest = build_manifest(
        "output/seedlings_1",
        filters=FilterConfig(min_snr_db=10.0, required_labels=["KCHI"]),
    )
    print(f"{len(manifest)} files pass filters")
    print(manifest.df.head())
"""

from __future__ import annotations

import logging
from pathlib import Path

from dataloader.adapters.esc import ESCAdapter
from dataloader.adapters.snr import SNRAdapter
from dataloader.adapters.vad import VADAdapter
from dataloader.adapters.vtc import VTCAdapter
from dataloader.config import FilterConfig
from dataloader.manifest.joiner import ManifestJoiner
from dataloader.manifest.schema import MetadataManifest
from dataloader.types import JoinStrategy

log = logging.getLogger(__name__)


def build_manifest(
    output_dir: Path | str,
    *,
    filters: FilterConfig | None = None,
    how: JoinStrategy | str = JoinStrategy.INNER,
    stages: list[str] | None = None,
) -> MetadataManifest:
    """Build a filtered, joined manifest from pipeline outputs.

    This is the main entry point for the "Big Join": it reads metadata from
    each pipeline adapter, joins on ``wav_id``, and applies optional filters.

    Parameters
    ----------
    output_dir:
        Root output directory for the dataset (e.g. ``output/seedlings_1``).
        Must contain the expected sub-directories for each pipeline stage.
    filters:
        Optional :class:`FilterConfig` for load-time data selection. If
        ``None``, all successfully processed files are included.
    how:
        Join strategy: ``"inner"`` (default, only files present in ALL
        stages), ``"left"`` (keep all from first stage), or ``"outer"``
        (keep all, fill gaps with null).
    stages:
        Which pipeline stages to include. Defaults to all available among
        ``["vad", "vtc", "snr", "esc"]``. Useful when not all stages
        have been run yet.

    Returns
    -------
    MetadataManifest
        A single manifest whose columns are the union of all stage metadata,
        filtered by *filters* if provided.

    Examples
    --------
    ::

        # All stages, no filtering
        manifest = build_manifest("output/seedlings_1")

        # Only VAD + VTC, with speech density filter
        manifest = build_manifest(
            "output/seedlings_1",
            stages=["vad", "vtc"],
            filters=FilterConfig(min_speech_ratio=0.3),
        )
    """
    output_dir = Path(output_dir)

    # Build adapter map — only include stages that have outputs.
    _ALL_ADAPTERS = {
        "vad": VADAdapter,
        "vtc": VTCAdapter,
        "snr": SNRAdapter,
        "esc": ESCAdapter,
    }
    requested = stages or list(_ALL_ADAPTERS.keys())

    manifests: list[tuple[str, MetadataManifest]] = []
    for name in requested:
        if name not in _ALL_ADAPTERS:
            log.warning("Unknown stage %r, skipping", name)
            continue
        adapter = _ALL_ADAPTERS[name](output_dir)
        try:
            df = adapter.as_manifest()
        except FileNotFoundError:
            log.info("Stage %r not available in %s, skipping", name, output_dir)
            continue
        manifests.append((name, MetadataManifest(df, source=name)))
        log.debug("Loaded %s manifest: %d rows", name, len(df))

    if not manifests:
        raise FileNotFoundError(
            f"No pipeline outputs found in {output_dir}. "
            f"Looked for stages: {requested}"
        )

    # Single stage — no join needed.
    if len(manifests) == 1:
        _, result = manifests[0]
    else:
        joiner = ManifestJoiner(how=how)
        for _, m in manifests:
            joiner.add(m)
        result = joiner.join()

    log.info(
        "Built manifest: %d rows, %d columns from stages %s",
        len(result),
        len(result.columns),
        [name for name, _ in manifests],
    )

    # Apply filters.
    if filters is not None:
        result = filters.apply(result)

    return result
