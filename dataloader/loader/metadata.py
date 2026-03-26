"""Metadata loading utilities.

:class:`MetadataLoader` provides a unified interface for reading precomputed
metadata from heterogeneous stores. It wraps one or more
:class:`~dataloader.manifest.store.MetadataStore` instances and merges their
outputs into a single :class:`~dataloader.types.MetadataDict` per wav_id.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dataloader.types import MetadataDict, WavID

if TYPE_CHECKING:
    from dataloader.manifest.store import MetadataStore

log = logging.getLogger(__name__)


class MetadataLoader:
    """Compose multiple :class:`MetadataStore` backends into one reader.

    Each store is registered under a namespace key. When :meth:`load` is
    called, the results from all stores are merged into a single dict with
    namespaced keys.

    Parameters
    ----------
    stores:
        Mapping from namespace → :class:`MetadataStore` instance.
        Example: ``{"vad": ParquetStore(...), "snr": NpzStore(...)}``.
    strict:
        If ``True``, raise :exc:`FileNotFoundError` when any store is
        missing data for a wav_id. If ``False``, missing stores are
        silently skipped and the corresponding namespace is absent from
        the result.
    """

    def __init__(
        self,
        stores: dict[str, MetadataStore] | None = None,
        strict: bool = False,
    ) -> None:
        self._stores: dict[str, MetadataStore] = dict(stores) if stores else {}
        self._strict = strict

    def add_store(self, namespace: str, store: MetadataStore) -> MetadataLoader:
        """Register a store under *namespace*.

        Returns ``self`` for chaining.

        Raises
        ------
        ValueError
            If *namespace* is already registered.
        """
        if namespace in self._stores:
            raise ValueError(f"Namespace {namespace!r} already registered.")
        self._stores[namespace] = store
        return self

    def load(self, wav_id: WavID) -> MetadataDict:
        """Load and merge metadata from all registered stores.

        Parameters
        ----------
        wav_id:
            Waveform identifier to look up across all stores.

        Returns
        -------
        MetadataDict
            Merged dict with keys namespaced as ``"{namespace}.{key}"``
            plus a top-level ``"wav_id"`` entry.
        """
        result: MetadataDict = {"wav_id": wav_id}

        for namespace, store in self._stores.items():
            try:
                data = store.load(wav_id)
            except FileNotFoundError:
                if self._strict:
                    raise
                log.debug(
                    "No %s metadata for wav_id=%s (strict=False, skipping)",
                    namespace,
                    wav_id,
                )
                continue

            for key, value in data.items():
                result[f"{namespace}.{key}"] = value

        return result

    def available_ids(self) -> list[WavID]:
        """Return wav_ids available in *all* registered stores.

        If ``strict=False``, returns the union (ids in any store).
        If ``strict=True``, returns the intersection (ids in every store).
        """
        if not self._stores:
            return []

        id_sets = [set(store.list_ids()) for store in self._stores.values()]

        if self._strict:
            common = id_sets[0]
            for s in id_sets[1:]:
                common &= s
            return sorted(common)
        else:
            union: set[WavID] = set()
            for s in id_sets:
                union |= s
            return sorted(union)

    @property
    def namespaces(self) -> list[str]:
        """Return registered namespace names."""
        return sorted(self._stores)

    def __repr__(self) -> str:
        return f"MetadataLoader(namespaces={self.namespaces}, strict={self._strict})"
