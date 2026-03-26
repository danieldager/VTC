"""Multi-manifest join (the "Big Join").

:class:`ManifestJoiner` composes :class:`MetadataManifest` objects from
heterogeneous pipeline stages into a single unified manifest, joined by
``wav_id``. This is the Dataloader++ equivalent of the implicit metadata
aggregation currently performed inside ``src/pipeline/package.py``.

Example
-------
::

    joiner = ManifestJoiner(how="inner")
    joiner.add(vad_manifest)
    joiner.add(vtc_manifest)
    joiner.add(snr_manifest)

    combined = joiner.join()
    print(combined.df.head())
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dataloader.types import JoinStrategy

if TYPE_CHECKING:
    from dataloader.manifest.schema import MetadataManifest

log = logging.getLogger(__name__)

_WAV_ID_COL = "wav_id"


class ManifestJoiner:
    """Declarative multi-manifest join on ``wav_id``.

    Parameters
    ----------
    how:
        Default join strategy applied to all :meth:`add` calls.
        One of ``"inner"`` (only wav_ids present in all manifests),
        ``"left"`` (keep all from first manifest), or ``"outer"``
        (keep all wav_ids, fill missing with null).
    suffix_template:
        Template for column-name deduplication. ``{source}`` is replaced
        with the manifest's :attr:`~MetadataManifest.source` attribute.
    """

    def __init__(
        self,
        how: JoinStrategy | str = JoinStrategy.INNER,
        suffix_template: str = "_{source}",
    ) -> None:
        self._how = JoinStrategy(how)
        self._suffix_template = suffix_template
        self._manifests: list[MetadataManifest] = []

    def add(self, manifest: MetadataManifest) -> ManifestJoiner:
        """Register a manifest for the upcoming join.

        Parameters
        ----------
        manifest:
            A :class:`MetadataManifest` to include in the join.

        Returns
        -------
        ManifestJoiner
            ``self``, for method chaining.
        """
        self._manifests.append(manifest)
        log.debug(
            "Added manifest %r (%d rows, %d columns)",
            manifest.source,
            len(manifest),
            len(manifest.columns),
        )
        return self

    def join(self) -> MetadataManifest:
        """Execute the Big Join across all registered manifests.

        Returns
        -------
        MetadataManifest
            A single manifest whose ``wav_id`` column is the join key and
            whose remaining columns are the union of all input columns
            (with suffixes applied to resolve conflicts).

        Raises
        ------
        ValueError
            If fewer than two manifests have been added.
        """
        from dataloader.manifest.schema import MetadataManifest

        if len(self._manifests) < 2:
            if len(self._manifests) == 1:
                return self._manifests[0]
            raise ValueError(
                "ManifestJoiner requires at least one manifest; "
                f"got {len(self._manifests)}"
            )

        result = self._manifests[0]
        for i, right in enumerate(self._manifests[1:], start=1):
            suffix = self._suffix_template.format(source=right.source)
            result = result.join(
                right,
                on=_WAV_ID_COL,
                how=self._how.value,
                suffix=suffix,
            )
            log.debug(
                "After joining manifest %d (%s): %d rows, %d columns",
                i,
                right.source,
                len(result),
                len(result.columns),
            )

        return result

    @property
    def manifests(self) -> list[MetadataManifest]:
        """Read-only view of registered manifests."""
        return list(self._manifests)

    def __len__(self) -> int:
        return len(self._manifests)

    def __repr__(self) -> str:
        sources = [m.source for m in self._manifests]
        return f"ManifestJoiner(how={self._how.value!r}, manifests={sources})"
