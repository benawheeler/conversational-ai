from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Deque
from collections import deque

import numpy as np
import soundfile as sf
import colorama

from conversation_ai.core.buffers import BufferManager
from conversation_ai.vad.semantic import SemanticTurnDetector

colorama.init(autoreset=True)


class TurnManager:
    """Aggregates speech segments into user turns using a semantic detector.

    Usage
    -----
        tm = TurnManager(detector)
        tm.handle_segment(event)  # for each SpeechEvent from SileroVAD
        tm.tick()                 # call regularly to handle timeouts
    """

    def __init__(
        self,
        detector: SemanticTurnDetector,
        *,
        sample_rate: int = 16_000,
        grace_window_sec: float = 0.5,
        soft_timeout_sec: float | None = None,
        hard_timeout_sec: float | None = None,
        context_max_sec: float = 6.0,
        turns_dir: str | os.PathLike = "turns",
        debug_dir: str | None = "context_inputs",
        transcriber=None,
    ) -> None:
        self.detector = detector
        self.sample_rate = sample_rate
        self.grace_window = grace_window_sec
        self.soft_timeout = soft_timeout_sec
        self.hard_timeout = hard_timeout_sec
        self.context_max_sec = context_max_sec
        self.turns_dir = Path(turns_dir)
        self.turns_dir.mkdir(parents=True, exist_ok=True)

        self.full_buffer = BufferManager(sample_rate)
        self.context_buffer = BufferManager(sample_rate)
        self.context_max_sec = context_max_sec

        # Single probability threshold for completion
        self.prob_threshold = 0.05

        # Decay threshold parameters
        self.decay_window_sec = 3.0  # total silence window
        self.min_threshold = 0.04
        self._last_prob: float | None = None
        self._last_printed_sec: int = 0
        self._asr_check_performed: bool = False
        self._last_vad_end_time: float | None = None

        # silence timer (disabled)
        self.silence_window_sec = 0.0
        self._silence_timer_due: Optional[float] = None

        # Timing
        self._last_segment_end_wall: Optional[float] = None  # wall clock
        self._soft_triggered = False

        # Debug directory
        self.debug_dir = Path(debug_dir) if debug_dir else None
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        # Store only speech chunks for ASR construction
        self._speech_segments: Deque[np.ndarray] = deque()

        # Whether we are currently in a speech segment (VAD active)
        self._in_speech: bool = False

        self.transcriber = transcriber

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def handle_segment(self, event):
        """Consume a SpeechEvent and run semantic detection pipeline."""
        # Load audio from saved WAV path
        pcm, sr = sf.read(event.wav_path, dtype="float32")
        if pcm.ndim > 1:
            pcm = pcm.reshape(-1)
        assert sr == self.sample_rate, "Unexpected sample rate from segment wav"

        now = time.time()
        self._last_vad_end_time = event.vad_end_time # Store timestamp from the latest segment
        if self._last_segment_end_wall is not None:
            gap = now - self._last_segment_end_wall
            if gap > self.grace_window:
                # Insert silence gap
                silence = np.zeros(int(gap * self.sample_rate), dtype=np.float32)
                self.full_buffer.append_audio(silence)
                self.context_buffer.append_audio(silence)
        # Append actual speech to buffers
        self.full_buffer.append_audio(pcm)
        self.context_buffer.append_audio(pcm)

        # Determine overlap to trim for ASR (dynamic rewind)
        relax_pad = 8000  # 500 ms at 16kHz, makes trimming much more relaxed
        if self._last_segment_end_wall is None:
            pcm_for_asr = pcm  # first segment, keep full rewind
            print(f"[ASR] First segment: no trimming (full pre-roll: {getattr(event, 'rewind_samples', 0)/self.sample_rate*1000:.1f} ms)")
        else:
            # Drop the actual rewind that Silero included, but be a bit more relaxed
            dynamic_ms = getattr(event, 'rewind_samples', 0) / self.sample_rate * 1000
            fixed_ms = relax_pad / self.sample_rate * 1000
            drop_samples = max(0, event.rewind_samples - relax_pad)
            drop_samples = min(drop_samples, len(pcm))
            print(f"[ASR] Segment: dynamic rewind={dynamic_ms:.1f} ms, relax_pad={fixed_ms:.1f} ms, drop={drop_samples/self.sample_rate*1000:.1f} ms")
            pcm_for_asr = pcm[drop_samples:]

        self._speech_segments.append(pcm_for_asr)
        # Trim context buffer to cap
        self.context_buffer.trim_to_duration(self.context_max_sec)
        self._last_segment_end_wall = now
        self._soft_triggered = False  # reset for new segment

        # Reset decay state for new speech
        self._last_prob = None
        self._last_printed_sec = 0
        self._in_speech = False  # speech just ended

        # Only run Smart-Turn once per segment
        self._run_detector()

    def tick(self):
        """Check timeouts; call frequently (e.g., each event loop iteration)."""
        if self._in_speech:
            return  # do not run decay while user is talking

        if self.full_buffer.is_empty() or self._last_segment_end_wall is None:
            return
        now = time.time()
        silence_dur = now - self._last_segment_end_wall

        # Tier 1: Delayed ASR Punctuation Check (at 1.5s)
        asr_check_time = 1.5
        if silence_dur >= asr_check_time and not self._asr_check_performed:
            self._asr_check_performed = True  # Ensure this only runs once

            print(f"[TurnManager] {asr_check_time}s silence reached. Checking ASR for punctuation...")

            asr_audio = self._build_asr_audio()
            if self.transcriber and asr_audio.size > 0:
                text = self.transcriber.transcribe(asr_audio)
                print(f"[ASR] Mid-turn transcription: '{text}'")

                if self._has_conclusive_punctuation(text):
                    print("[TurnManager] ASR punctuation found. Overriding VAD and finalizing turn.")
                    self._finalize_turn()
                    return  # The turn is over, no need to continue the tick

        # Tier 2: Decay threshold logic ---------------------------------------
        if silence_dur > 0:
            # Compute dynamic threshold
            frac = min(silence_dur / self.decay_window_sec, 1.0)
            dyn_threshold = max(
                self.min_threshold,
                self.prob_threshold * (1.0 - frac),
            )

            # Print once per elapsed whole second
            if int(silence_dur) > self._last_printed_sec and int(silence_dur) <= self.decay_window_sec:
                self._last_printed_sec = int(silence_dur)
                print(
                    f"[TurnManager] silence {self._last_printed_sec}s -> dynamic threshold {dyn_threshold:.2f}"
                )

            # Auto-finalize after 4 seconds of silence
            if silence_dur >= self.decay_window_sec:
                print(f"[TurnManager] {self.decay_window_sec}s silence – forcing turn completion")
                self._finalize_turn()

            # Finalize if stored prob meets the decayed threshold
            if self._last_prob is not None and self._last_prob >= dyn_threshold:
                self._finalize_turn()

        if self.hard_timeout is not None and silence_dur >= self.hard_timeout:
            print("[TurnManager] Hard timeout reached; forcing turn completion")
            self._finalize_turn()
        elif self.soft_timeout is not None and silence_dur >= self.soft_timeout and not self._soft_triggered:
            self._soft_triggered = True
            print("[TurnManager] Soft timeout reached; re-evaluating…")
            audio = self.context_buffer.get_audio()
            if audio.size == 0:
                return  # nothing to evaluate yet

            dur = len(audio) / self.sample_rate
            print(f"[TurnManager] Running Semantic VAD on {dur:.2f}s audio ({len(audio)} samples)")
            done, prob = self.detector.predict(audio)
            print(f"[TurnManager] Semantic VAD after soft timeout prob={prob:.2f} done={done}")
            if done:
                self._finalize_turn()

        # self._last_prob is updated in _run_detector

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_detector(self):
        audio = self.context_buffer.get_audio()
        if audio.size == 0:
            return  # nothing to evaluate yet

        dur = len(audio) / self.sample_rate
        print(f"[TurnManager] Running Semantic VAD on {dur:.2f}s audio ({len(audio)} samples)")
        done, prob = self.detector.predict(audio)
        print(f"[TurnManager] Semantic VAD prob={prob:.2f} done={done}")
        self._last_prob = prob
        if prob >= self.prob_threshold:
            self._finalize_turn()

        # Save debug wav
        if self.debug_dir:
            ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            fname = f"ctx_{ts}_{int(dur*1000):06d}ms.wav"
            sf.write(self.debug_dir / fname, audio, self.sample_rate)

    def _finalize_turn(self):
        audio = self.full_buffer.get_audio()
        if audio.size == 0:
            self.full_buffer.clear()
            self.context_buffer.clear()
            self._last_segment_end_wall = None
            self._in_speech = False
            return

        ts_wall = time.strftime("%Y%m%d-%H%M%S", time.localtime())

        # Save raw turn
        path_raw = self.turns_dir / f"turn_{ts_wall}.wav"
        sf.write(path_raw, audio, self.sample_rate)

        # Build ASR-friendly clip (speech segments + 0.3s gap)
        asr_audio = self._build_asr_audio()

        asr_dir = self.turns_dir / "asr"
        asr_dir.mkdir(exist_ok=True, parents=True)
        path_asr = asr_dir / f"asr_turn_{ts_wall}.wav"
        sf.write(path_asr, asr_audio, self.sample_rate)

        if self.transcriber is not None:
            text = self.transcriber.transcribe(asr_audio)
            print(f"[ASR] Transcription: {text}")

        latency_ms = (time.time() - self._last_vad_end_time) * 1000 if self._last_vad_end_time else 0

        print(
            f"{colorama.Fore.GREEN}[TurnManager] Turn finalized (latency: {latency_ms:.0f} ms) | raw={len(audio)/self.sample_rate:.2f}s → {path_raw} | "
            f"asr={len(asr_audio)/self.sample_rate:.2f}s → {path_asr}"
        )

        # Reset
        self.full_buffer.clear()
        self.context_buffer.clear()
        self._last_segment_end_wall = None
        self._soft_triggered = False
        self._in_speech = False
        self._speech_segments.clear()
        self._last_prob = None
        self._last_printed_sec = 0
        self._asr_check_performed = False
        self._last_vad_end_time = None

    def _build_asr_audio(self) -> np.ndarray:
        """Constructs the ASR-ready audio clip from speech segments with fixed gaps."""
        if not self._speech_segments:
            return np.empty((0,), dtype=np.float32)

        gap_len = int(0.3 * self.sample_rate)
        gap = np.zeros(gap_len, dtype=np.float32)
        asr_chunks: list[np.ndarray] = []
        for i, seg in enumerate(self._speech_segments):
            asr_chunks.append(seg)
            if i != len(self._speech_segments) - 1:
                asr_chunks.append(gap)
        return np.concatenate(asr_chunks, axis=0)

    def _has_conclusive_punctuation(self, text: str) -> bool:
        """Checks for strong sentence-ending punctuation."""
        text = text.strip()
        if not text:
            return False

        # Check for conclusive endings
        if text.endswith('!') or text.endswith('?'):
            return True

        # Check for a single period, but not "..."
        if text.endswith('.') and not text.endswith('..'):
            return True

        return False

    # ------------------------------------------------------------------
    def reset_decay(self):
        """Called when speech resumes to cancel silence decay timer."""
        if self._last_prob is not None:
            print("[TurnManager] Speech resumed – dynamic timer reset")
        self._last_prob = None
        self._last_printed_sec = 0
        self._in_speech = True
        self._asr_check_performed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def handle_segment(self, event):
        """Consume a SpeechEvent and run semantic detection pipeline."""
        # Load audio from saved WAV path
        pcm, sr = sf.read(event.wav_path, dtype="float32")
        if pcm.ndim > 1:
            pcm = pcm.reshape(-1)
        assert sr == self.sample_rate, "Unexpected sample rate from segment wav"

        now = time.time()
        self._last_vad_end_time = event.vad_end_time # Store timestamp from the latest segment
        if self._last_segment_end_wall is not None:
            gap = now - self._last_segment_end_wall
            if gap > self.grace_window:
                # Insert silence gap
                silence = np.zeros(int(gap * self.sample_rate), dtype=np.float32)
                self.full_buffer.append_audio(silence)
                self.context_buffer.append_audio(silence)
        # Append actual speech to buffers
        self.full_buffer.append_audio(pcm)
        self.context_buffer.append_audio(pcm)

        # Determine overlap to trim for ASR (dynamic rewind)
        relax_pad = 8000  # 500 ms at 16kHz, makes trimming much more relaxed
        if self._last_segment_end_wall is None:
            pcm_for_asr = pcm  # first segment, keep full rewind
            print(f"[ASR] First segment: no trimming (full pre-roll: {getattr(event, 'rewind_samples', 0)/self.sample_rate*1000:.1f} ms)")
        else:
            # Drop the actual rewind that Silero included, but be a bit more relaxed
            dynamic_ms = getattr(event, 'rewind_samples', 0) / self.sample_rate * 1000
            fixed_ms = relax_pad / self.sample_rate * 1000
            drop_samples = max(0, event.rewind_samples - relax_pad)
            drop_samples = min(drop_samples, len(pcm))
            print(f"[ASR] Segment: dynamic rewind={dynamic_ms:.1f} ms, relax_pad={fixed_ms:.1f} ms, drop={drop_samples/self.sample_rate*1000:.1f} ms")
            pcm_for_asr = pcm[drop_samples:]

        self._speech_segments.append(pcm_for_asr)
        # Trim context buffer to cap
        self.context_buffer.trim_to_duration(self.context_max_sec)
        self._last_segment_end_wall = now
        self._soft_triggered = False  # reset for new segment

        # Reset decay state for new speech
        self._last_prob = None
        self._last_printed_sec = 0
        self._in_speech = False  # speech just ended

        # Only run Smart-Turn once per segment
        self._run_detector() 