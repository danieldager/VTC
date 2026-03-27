"""Pipeline adapters — thin wrappers exposing ``src/pipeline/*`` as FeatureProcessors.

Each adapter delegates to the existing pipeline code without modifying it.
The key normalization rule: ``file_id`` / ``uid`` from the pipeline become
``wav_id`` at the adapter boundary.
"""

from dataloader.adapters.esc import ESCAdapter
from dataloader.adapters.snr import SNRAdapter
from dataloader.adapters.vad import VADAdapter
from dataloader.adapters.vtc import VTCAdapter

__all__ = ["ESCAdapter", "SNRAdapter", "VADAdapter", "VTCAdapter"]
