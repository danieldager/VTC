"""Label transforms: encoding and mask generation.

These transforms operate on the metadata dict, converting string labels
to integer IDs and generating attention/prediction masks suitable for
model training.
"""

from __future__ import annotations

import torch

from dataloader.transform.base import DataProcessor
from dataloader.types import MetadataDict, SampleRate, SegmentList, Waveform


class LabelEncoder(DataProcessor):
    """Encode string segment labels to integer IDs.

    Reads a :data:`SegmentList` from metadata, replaces each segment's
    ``"label"`` field with an integer ``"label_id"``, and stores the
    encoded list back into metadata.

    Parameters
    ----------
    labels:
        Ordered list of label strings. The index becomes the integer ID.
    segments_key:
        Metadata key containing the :data:`SegmentList`.
    output_key:
        Metadata key for the encoded segment list.
    unknown_id:
        Integer ID assigned to labels not in *labels*.
        If ``None``, unknown labels raise :exc:`ValueError`.
    """

    def __init__(
        self,
        labels: list[str],
        segments_key: str = "vtc_segments",
        output_key: str = "vtc_segments_encoded",
        unknown_id: int | None = None,
    ) -> None:
        self._label_to_id = {label: i for i, label in enumerate(labels)}
        self._labels = list(labels)
        self._segments_key = segments_key
        self._output_key = output_key
        self._unknown_id = unknown_id

    def __call__(
        self,
        waveform: Waveform,
        sample_rate: SampleRate,
        metadata: MetadataDict,
    ) -> tuple[Waveform, SampleRate, MetadataDict]:
        segments: SegmentList = metadata.get(self._segments_key, [])  # type: ignore[assignment]

        encoded: SegmentList = []
        for seg in segments:
            label = seg.get("label")
            if label is None:
                label_id = self._unknown_id if self._unknown_id is not None else -1
            elif label in self._label_to_id:
                label_id = self._label_to_id[label]
            elif self._unknown_id is not None:
                label_id = self._unknown_id
            else:
                raise ValueError(
                    f"Unknown label {label!r}. "
                    f"Known labels: {self._labels}"
                )
            encoded.append({**seg, "label_id": label_id})

        metadata = {**metadata, self._output_key: encoded}
        # Store the label vocabulary for downstream decoding.
        metadata["label_vocab"] = self._labels
        return waveform, sample_rate, metadata

    @property
    def num_labels(self) -> int:
        return len(self._labels)

    def decode(self, label_id: int) -> str:
        """Convert an integer ID back to its string label."""
        return self._labels[label_id]

    def __repr__(self) -> str:
        return f"LabelEncoder(labels={self._labels})"


class MaskGenerator(DataProcessor):
    """Generate frame-level attention and label masks.

    Produces boolean tensors indicating which frames contain speech
    (attention mask) and which speaker label is active at each frame
    (label mask). These are stored in metadata for use by the collator.

    Parameters
    ----------
    segments_key:
        Metadata key containing the :data:`SegmentList` with ``label_id``.
    frame_shift_s:
        Duration of each frame in seconds (e.g. 0.02 for 20 ms frames).
    num_labels:
        Total number of label classes (for multi-hot encoding).
    attention_key:
        Metadata key for the output attention mask.
    label_key:
        Metadata key for the output label mask.
    """

    def __init__(
        self,
        segments_key: str = "vtc_segments_encoded",
        frame_shift_s: float = 0.02,
        num_labels: int = 4,
        attention_key: str = "attention_mask",
        label_key: str = "label_mask",
    ) -> None:
        self._segments_key = segments_key
        self._frame_shift_s = frame_shift_s
        self._num_labels = num_labels
        self._attention_key = attention_key
        self._label_key = label_key

    def __call__(
        self,
        waveform: Waveform,
        sample_rate: SampleRate,
        metadata: MetadataDict,
    ) -> tuple[Waveform, SampleRate, MetadataDict]:
        duration_s = waveform.shape[-1] / sample_rate
        n_frames = int(duration_s / self._frame_shift_s)

        attention_mask = torch.zeros(n_frames, dtype=torch.bool)
        label_mask = torch.zeros(n_frames, self._num_labels, dtype=torch.bool)

        segments: SegmentList = metadata.get(self._segments_key, [])  # type: ignore[assignment]
        for seg in segments:
            start_frame = int(seg["onset"] / self._frame_shift_s)
            end_frame = min(int(seg["offset"] / self._frame_shift_s), n_frames)
            attention_mask[start_frame:end_frame] = True

            label_id = seg.get("label_id")
            if label_id is not None and isinstance(label_id, int) and 0 <= label_id < self._num_labels:
                label_mask[start_frame:end_frame, label_id] = True

        metadata = {
            **metadata,
            self._attention_key: attention_mask,
            self._label_key: label_mask,
        }
        return waveform, sample_rate, metadata

    def __repr__(self) -> str:
        return (
            f"MaskGenerator(frame_shift_s={self._frame_shift_s}, "
            f"num_labels={self._num_labels})"
        )
