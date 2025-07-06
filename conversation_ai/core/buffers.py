from __future__ import annotations

from collections import deque
from typing import Deque, Tuple

import numpy as np


class BufferManager:
    """Holds the audio/text buffers for the current user turn."""

    def __init__(self, sample_rate: int = 16_000):
        self.sample_rate = sample_rate
        self._audio_chunks: Deque[np.ndarray] = deque()
        self._total_samples: int = 0

    # ------------------------------------------------------------------
    def append_audio(self, pcm: np.ndarray):
        """Append raw PCM chunk (1-D float32 array)."""
        if pcm.ndim != 1:
            raise ValueError("PCM must be mono 1-D")
        self._audio_chunks.append(pcm)
        self._total_samples += len(pcm)

    def append_silence(self, duration_sec: float):
        """Append synthetic silence of given duration to the buffer."""
        n_samples = int(duration_sec * self.sample_rate)
        if n_samples <= 0:
            return
        silence = np.zeros(n_samples, dtype=np.float32)
        self.append_audio(silence)

    def get_audio(self) -> np.ndarray:
        if not self._audio_chunks:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(list(self._audio_chunks), axis=0)

    def duration_sec(self) -> float:
        return self._total_samples / self.sample_rate

    def clear(self):
        self._audio_chunks.clear()
        self._total_samples = 0

    def is_empty(self) -> bool:
        return len(self._audio_chunks) == 0

    # ------------------------------------------------------------------
    # Trimming helpers
    # ------------------------------------------------------------------
    def pop_left(self):
        """Remove the oldest chunk from the buffer."""
        if self._audio_chunks:
            chunk = self._audio_chunks.popleft()
            self._total_samples -= len(chunk)

    def trim_to_duration(self, max_seconds: float):
        """Ensure total duration ≤ max_seconds by dropping oldest chunks."""
        max_samples = int(max_seconds * self.sample_rate)
        while self._total_samples > max_samples and len(self._audio_chunks) > 1:
            self.pop_left()

        # If we still exceed limit and only one chunk remains, slice it from the front
        if self._total_samples > max_samples and self._audio_chunks:
            excess = self._total_samples - max_samples
            chunk = self._audio_chunks.popleft()
            chunk = chunk[excess:]
            self._audio_chunks.appendleft(chunk)
            self._total_samples = max_samples 