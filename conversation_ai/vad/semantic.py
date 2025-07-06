import torch
import numpy as np
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor


class SemanticTurnDetector:
    """Wrapper around the pipecat-ai/smart-turn model (HuggingFace).

    Parameters
    ----------
    model_id : str, default "pipecat-ai/smart-turn"
        HF model id to load. Must be a sequence classification checkpoint that
        outputs 2 logits (class 0: incomplete, class 1: complete).
    device : str | torch.device, optional
        Device on which to run inference. Defaults to 'cuda' if available.
    use_half : bool, optional
        Whether to use half precision for inference. Defaults to False.
    """

    def __init__(self, model_id: str = "pipecat-ai/smart-turn", device: str | torch.device | None = None, use_half: bool = False):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load feature extractor (no tokenizer needed for pure audio classification)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModelForAudioClassification.from_pretrained(model_id)
        # Do not use half precision by default; some layers (e.g. LayerNorm) require float32
        # If you want to enable half precision, ensure all model layers support it.
        self.model.to(self.device)
        print(f"[SemanticTurnDetector] Model loaded on {self.device} (half={use_half and self.device.type=='cuda'})")
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, audio: np.ndarray, sampling_rate: int = 16_000) -> tuple[bool, float]:
        """Return (is_done, probability_of_done).

        Notes
        -----
        • *audio* should be mono PCM float32 in range [-1, 1].
        • The model expects max_length ≤ 800 (in frames ~ 5 sec). Longer audio
          will be truncated by the processor; we leave padding + truncation at
          defaults as in smart-turn inference.py.
        """
        assert audio.ndim == 1, "audio must be mono 1-D array"

        inputs = self.feature_extractor(
            audio,
            sampling_rate=sampling_rate,
            padding="max_length",
            truncation=True,
            max_length=800,
            return_attention_mask=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        completion_prob = probs[0, 1].item()
        is_done = completion_prob > 0.5
        return is_done, completion_prob 