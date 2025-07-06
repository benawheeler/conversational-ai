import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

# -----------------------------------------------------------------------------
# NOTE: PyTorch Hub will download the model weights (~5 MB) the first time. We
#       use trust_repo=True for scripted model.
# -----------------------------------------------------------------------------

@dataclass
class SpeechEvent:
    """Metadata for one detected speech segment saved to disk."""

    wav_path: Path
    start_time_ms: float  # timestamp relative to detector start
    end_time_ms: float
    rewind_samples: int  # how many leading samples are overlap
    vad_end_time: float   # time.time() when VAD segment ended


class SileroVAD:
    """Streaming wrapper around Silero voice activity detector.

    The detector processes incoming PCM frames (float32, 16 kHz, mono) and emits
    a `SpeechEvent` once a speech segment has *ended*. The saved WAV file
    contains:
      • `pre_roll_sec` seconds preceding the detected speech start (to avoid
        clipping the first phoneme).
      • The full speech up until `speech_end` is confirmed.

    Parameters
    ----------
    pre_roll_sec : float, default 1.0
        Seconds of audio to prepend before the detected speech start.
    sample_rate : int, default 16000
        PCM sampling rate expected by the Silero model.
    speech_threshold : float, default 0.5
        Probability above which we treat the model output as "speech".
    min_speech_frames : int, default 5
        Minimum consecutive speech frames required to open a segment (helps
        filter single-frame glitches).
    silence_hold_frames : int, default 10
        Number of consecutive *non-speech* frames that will close an open
        segment.
    block_size : int, default 512
        Block size for frame processing.
    output_dir : Path | str, default "segments"
        Directory where WAV snippets are written for inspection.
    """

    def __init__(
        self,
        *,
        pre_roll_sec: float = 1.0,
        sample_rate: int = 16_000,
        speech_threshold: float = 0.5,
        min_speech_frames: int = 5,
        silence_hold_frames: int = 10,
        block_size: int = 512,
        output_dir: str | os.PathLike = "segments",
    ) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.pre_roll_sec = pre_roll_sec
        self.speech_threshold = speech_threshold
        self.min_speech_frames = min_speech_frames
        self.silence_hold_frames = silence_hold_frames

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        self._model = self._load_model()
        self._model.eval()
        self._model.to(self.device)

        # Rolling buffer of the last `pre_roll_sec` seconds for rewind
        self._pre_roll_frames = int(pre_roll_sec * sample_rate)
        self._rewind_buf: Deque[np.ndarray] = deque(maxlen=self._frames_to_blocks(self._pre_roll_frames, self.block_size))

        # State
        self._speech_active = False
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._segment_frames: List[np.ndarray] = []  # list of PCM frames

        # Timing
        self._total_frames_processed = 0
        self._start_time = time.time()  # wall clock start for timestamps

        # IO
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public streaming API
    # ------------------------------------------------------------------
    def feed(self, frame: np.ndarray) -> List[SpeechEvent]:
        """Feed a single PCM frame into the detector.

        Returns a list of `SpeechEvent`s emitted **after** this frame (usually
        0 or 1). We return a list so the caller can handle multiple events in
        pathological cases where segments end back-to-back.
        """
        assert frame.ndim == 1, "frame must be mono, 1-D array"
        assert frame.dtype == np.float32, "frame must be float32"

        events: List[SpeechEvent] = []

        # Keep copy in rewind buffer
        self._rewind_buf.append(frame)

        # Convert frame to torch tensor in expected shape (1, T)
        with torch.no_grad():
            waveform = torch.from_numpy(frame).unsqueeze(0).to(self.device)
            probs: torch.Tensor = self._model(waveform, self.sample_rate)
            speech_prob = float(probs.item())

        self._total_frames_processed += len(frame)

        # State machine -----------------------------------------------------------------
        if self._speech_active:
            if speech_prob >= self.speech_threshold:
                # Still speech
                self._segment_frames.append(frame)
                self._silence_frame_count = 0
            else:
                # Silence frame
                self._silence_frame_count += 1
                self._segment_frames.append(frame)

                if self._silence_frame_count >= self.silence_hold_frames:
                    # Close segment
                    evt = self._close_segment()
                    events.append(evt)
        else:
            if speech_prob >= self.speech_threshold:
                # Potential start
                self._speech_frame_count += 1
                if self._speech_frame_count >= self.min_speech_frames:
                    self._open_segment()
                    # Record current frame too
                    self._segment_frames.append(frame)
            else:
                # reset counter if not consecutive
                self._speech_frame_count = 0

        return events

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_model(self):
        # Using torch.hub to fetch the ONNX-scripted model (~5 MB).
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        # We only need the model for streaming predictions; utilities are ignored.
        return model

    @staticmethod
    def _frames_to_blocks(num_samples: int, block_size: int) -> int:
        return max(1, (num_samples + block_size - 1) // block_size)

    def _open_segment(self):
        self._speech_active = True
        self._silence_frame_count = 0
        self._segment_frames.clear()

        start_sec = self._total_frames_processed / self.sample_rate
        print(f"[SileroVAD] Speech started at {start_sec:.2f}s")

        # Determine how much rewind is actually needed based on gap since last segment
        if hasattr(self, "_last_segment_end_frame") and self._last_segment_end_frame is not None:
            gap_frames = self._total_frames_processed - self._last_segment_end_frame
        else:
            gap_frames = self._pre_roll_frames  # first segment => take full cap

        rewind_frames = min(gap_frames, self._pre_roll_frames)

        if rewind_frames > 0 and self._rewind_buf:
            # Concatenate enough recent blocks to cover rewind_frames
            buf_audio = np.concatenate(list(self._rewind_buf), axis=0)
            pre_audio = buf_audio[-rewind_frames:]
            self._segment_frames.append(pre_audio)
        else:
            self._segment_frames.append(np.zeros(rewind_frames, dtype=np.float32))

        # Store for use when closing segment
        self._current_rewind_frames = rewind_frames

        # Timestamp when segment began (approx.) — deduct pre-roll
        self._segment_start_frames = self._total_frames_processed - self._pre_roll_frames

    def _close_segment(self) -> SpeechEvent:
        self._speech_active = False
        self._speech_frame_count = 0
        self._silence_frame_count = 0

        # Concatenate all collected PCM
        pcm = np.concatenate(self._segment_frames, axis=0)
        rewind_samples = getattr(self, "_current_rewind_frames", 0)

        # Build WAV filename
        start_ms = (self._segment_start_frames / self.sample_rate) * 1000.0
        end_frames = self._segment_start_frames + len(pcm)
        end_ms = (end_frames / self.sample_rate) * 1000.0
        ts_wall = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        filename = f"seg_{ts_wall}_{int(start_ms):07d}-{int(end_ms):07d}.wav"
        wav_path = self.output_dir / filename

        # Write using soundfile
        sf.write(wav_path, pcm, self.sample_rate)

        duration_sec = len(pcm) / self.sample_rate
        print(f"[SileroVAD] Speech ended → duration {duration_sec:.2f}s saved to {wav_path}")

        # Clear buffers keeping only last pre_roll seconds (already there)
        self._segment_frames.clear()
        # update last segment end frame index
        self._last_segment_end_frame = end_frames

        return SpeechEvent(
            wav_path=wav_path,
            start_time_ms=start_ms,
            end_time_ms=end_ms,
            rewind_samples=rewind_samples,
            vad_end_time=time.time(),
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def reset(self):
        """Reset internal state (does not clear saved WAVs)."""
        self._speech_active = False
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._segment_frames.clear()
        self._rewind_buf.clear()
        self._total_frames_processed = 0
        self._start_time = time.time() 

    # Public helper
    def is_active(self) -> bool:
        """Return True if currently inside a speech segment."""
        return self._speech_active 