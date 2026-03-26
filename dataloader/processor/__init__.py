"""Feature Processor abstractions for offline metadata extraction."""

from dataloader.processor.base import FeatureProcessor
from dataloader.processor.registry import ProcessorRegistry

__all__ = ["FeatureProcessor", "ProcessorRegistry"]
