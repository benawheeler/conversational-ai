import torch
import whisper
import soundfile as sf
import torchaudio.functional as F
import numpy as np


class WhisperTranscriber:
    """Wrapper around OpenAI Whisper for offline transcription of turn WAVs."""

    def __init__(self, model_name: str = "base") -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_name, device=device)
        self.device = device
        print(f"[ASR] Whisper '{model_name}' model loaded on {device}")

    def _load_audio(self, path: str, target_sr: int = 16000):
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # convert to mono
        if sr != target_sr:
            audio = torch.from_numpy(audio)
            audio = F.resample(audio, sr, target_sr).numpy()
        return audio

    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe a NumPy array of audio data and return the text."""
        # Ensure audio is float32, which whisper expects
        audio_float32 = audio_data.astype(np.float32)
        result = self.model.transcribe(audio_float32, fp16=self.device == "cuda")
        text = result.get("text", "").strip()
        return text 