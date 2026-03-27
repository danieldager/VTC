"""Processor registration and discovery.

The :class:`ProcessorRegistry` provides a central catalog of available
:class:`~dataloader.processor.base.FeatureProcessor` implementations. It
supports both explicit registration and class-decorator syntax.

Example
-------
::

    registry = ProcessorRegistry()

    @registry.register
    class VADProcessor(FeatureProcessor):
        name = "vad"
        ...

    proc = registry.get("vad")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dataloader.processor.base import FeatureProcessor

log = logging.getLogger(__name__)


class ProcessorRegistry:
    """Thread-safe registry mapping processor names to classes.

    Attributes
    ----------
    _processors : dict[str, type[FeatureProcessor]]
        Internal mapping from :attr:`FeatureProcessor.name` to class.
    """

    def __init__(self) -> None:
        self._processors: dict[str, type[FeatureProcessor]] = {}

    # ── Registration ──────────────────────────────────────────────────────
    def register(self, cls: type[FeatureProcessor]) -> type[FeatureProcessor]:
        """Register a processor class.

        Can be used as a decorator::

            @registry.register
            class MyProcessor(FeatureProcessor): ...

        Parameters
        ----------
        cls:
            A :class:`FeatureProcessor` subclass with a ``name`` attribute.

        Returns
        -------
        type[FeatureProcessor]
            The same class, unmodified (allows decorator usage).

        Raises
        ------
        ValueError
            If a processor with the same ``name`` is already registered.
        """
        name = cls.name
        if name in self._processors:
            raise ValueError(
                f"Processor {name!r} already registered by "
                f"{self._processors[name].__name__}; cannot register "
                f"{cls.__name__}"
            )
        self._processors[name] = cls
        log.debug("Registered processor %r → %s", name, cls.__name__)
        return cls

    # ── Lookup ────────────────────────────────────────────────────────────
    def get(self, name: str) -> type[FeatureProcessor]:
        """Return the class registered under *name*.

        Raises
        ------
        KeyError
            If no processor is registered with that name.
        """
        try:
            return self._processors[name]
        except KeyError:
            available = ", ".join(sorted(self._processors)) or "(none)"
            raise KeyError(
                f"No processor registered as {name!r}. "
                f"Available: {available}"
            ) from None

    def list(self) -> list[str]:
        """Return sorted list of registered processor names."""
        return sorted(self._processors)


# ── Default registry with built-in adapters ───────────────────────────────

def default_registry() -> ProcessorRegistry:
    """Create a :class:`ProcessorRegistry` pre-loaded with pipeline adapters.

    Registers :class:`VADAdapter`, :class:`VTCAdapter`, :class:`SNRAdapter`,
    and :class:`ESCAdapter` so that ``registry.get("vad")`` etc. work
    immediately.
    """
    from dataloader.adapters import ESCAdapter, SNRAdapter, VADAdapter, VTCAdapter

    reg = ProcessorRegistry()
    reg.register(VADAdapter)
    reg.register(VTCAdapter)
    reg.register(SNRAdapter)
    reg.register(ESCAdapter)
    return reg

    def __contains__(self, name: str) -> bool:
        return name in self._processors

    def __len__(self) -> int:
        return len(self._processors)

    def __repr__(self) -> str:
        names = ", ".join(self.list())
        return f"ProcessorRegistry([{names}])"
