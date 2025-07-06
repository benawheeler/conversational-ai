import asyncio
from collections import deque
from typing import AsyncGenerator, Deque, Optional

import numpy as np
import sounddevice as sd


class RingBuffer:
    """Fixed-length FIFO buffer that stores the most recent audio frames.

    The buffer holds *frames*, not individual samples, so that we can keep
    per-block timing intact. Frames are expected to be 1-D NumPy arrays of
    PCM samples (float32, 16 kHz, mono).
    """

    def __init__(self, max_frames: int):
        self._max_frames = max_frames
        self._buf: Deque[np.ndarray] = deque(maxlen=max_frames)

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def extend(self, frame: np.ndarray) -> None:
        """Append a frame to the buffer (makes a copy to avoid aliasing)."""
        self._buf.append(frame.copy())

    def get(self, n_frames: Optional[int] = None) -> np.ndarray:
        """Return the most recent *n_frames* concatenated as a 1-D array.

        If *n_frames* is None or exceeds the buffer length, the whole buffer
        is returned.
        """
        if not self._buf:
            return np.empty((0,), dtype=np.float32)

        if n_frames is None or n_frames > len(self._buf):
            n_frames = len(self._buf)
        frames = list(self._buf)[-n_frames:]
        return np.concatenate(frames, axis=0)

    def clear(self) -> None:
        self._buf.clear()

    # ---------------------------------------------------------------------
    # Convenience properties
    # ---------------------------------------------------------------------
    @property
    def num_frames(self) -> int:
        return len(self._buf)


class AudioStream:
    """Asynchronous microphone reader that yields raw PCM blocks.

    Example
    -------
    >>> async with AudioStream() as mic:
    ...     async for block in mic.frames():
    ...         process(block)
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        block_size: int = 512,
        channels: int = 1,
        dtype: str = "float32",
        ring_buffer_seconds: float = 1.0,
        queue_maxsize: int = 128,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.channels = channels
        self.dtype = dtype

        # Playback cancellation (filled by callback)
        self._queue: "asyncio.Queue[np.ndarray]" = asyncio.Queue(maxsize=queue_maxsize)

        # Ring buffer for rewind (stores *blocks*, not samples)
        max_frames = int((ring_buffer_seconds * sample_rate) / block_size)
        self.ring_buffer = RingBuffer(max_frames=max_frames)

        # Underlying sounddevice stream (created in __aenter__)
        self._sd_stream: Optional[sd.InputStream] = None

    # ------------------------------------------------------------------
    # Context-manager helpers so callers can `async with AudioStream()`
    # ------------------------------------------------------------------
    async def __aenter__(self) -> "AudioStream":
        loop = asyncio.get_running_loop()

        def _callback(indata: np.ndarray, frames: int, time, status) -> None:  # noqa: D401
            if status:
                # Non-fatal warnings are forwarded to stderr by sounddevice, so
                # we only need to handle fatal errors.
                print("Sounddevice status:", status, flush=True)

            # Flatten to 1-D mono float32 array.
            mono = indata.copy().reshape(-1)
            # Write to rewind buffer first.
            self.ring_buffer.extend(mono)

            # Try to push to async queue without blocking audio callback.
            def _safe_put(q, item):
                try:
                    q.put_nowait(item)
                except asyncio.QueueFull:
                    # Drop block if queue full
                    pass

            loop.call_soon_threadsafe(_safe_put, self._queue, mono)

        self._sd_stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=self.channels,
            dtype=self.dtype,
            callback=_callback,
        )
        self._sd_stream.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._sd_stream is not None:
            self._sd_stream.stop()
            self._sd_stream.close()
            self._sd_stream = None

    # ------------------------------------------------------------------
    # Public async generator
    # ------------------------------------------------------------------
    async def frames(self) -> AsyncGenerator[np.ndarray, None]:
        """Yield raw PCM frames from the microphone as they arrive."""
        while True:
            frame = await self._queue.get()
            yield frame 