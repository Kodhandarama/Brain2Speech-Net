"""Brain2Speech-Net: neural activity -> contextual phoneme representations -> speech, in one pass."""
from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import torch
from torch import nn

from .aligner import AlignerOutput, DeepHMMAligner
from .content_encoder import BrainContentEncoder
from .phonemes import N_CTC_CLASSES


class Brain2SpeechNet(nn.Module):
    """Brain content encoder + deep-HMM aligner/PPSP + auxiliary CTC head.

    The TTS backbone (VITS) is frozen and lives outside this module
    (:class:`brain2speech_net.tts.PretrainedVITS`); this module maps neural features to the
    CPRs that VITS turns into speech.
    """

    def __init__(self, encoder: Optional[dict] = None, aligner: Optional[dict] = None):
        super().__init__()
        self.content_encoder = BrainContentEncoder(**(encoder or {}))
        aligner = dict(aligner or {})
        aligner.setdefault("d_in", self.content_encoder.n_outputs)
        self.aligner = DeepHMMAligner(**aligner)
        # Linear CTC head over the phoneme-aware representations; keeps the bottleneck linguistic.
        self.ctc_head = nn.Linear(self.content_encoder.n_outputs, N_CTC_CLASSES)

    @property
    def config(self) -> Dict[str, dict]:
        return {"encoder": self.content_encoder.config, "aligner": self.aligner.config}

    def forward(
        self,
        neural: torch.Tensor,
        day_idx: torch.Tensor,
        frame_lengths: torch.Tensor,
        cpr: Optional[torch.Tensor] = None,
        n_segments: Optional[torch.Tensor] = None,
    ) -> Dict[str, Union[torch.Tensor, AlignerOutput]]:
        """
        Args:
            neural: (B, T, 256) z-scored neural features (20 ms bins).
            day_idx: (B,) recording-session indices.
            frame_lengths: (B,) valid encoder frames I per utterance.
            cpr / n_segments: see :meth:`DeepHMMAligner.forward`.
        """
        n = self.content_encoder(neural, day_idx)
        out = self.aligner(n, frame_lengths, cpr=cpr, n_segments=n_segments)
        with torch.autocast(device_type="cuda" if n.is_cuda else "cpu", enabled=False):
            ctc_logits = self.ctc_head(n.float())
        return {"n": n, "aligner": out, "ctc_logits": ctc_logits}

    @torch.inference_mode()
    def predict_cpr(
        self, neural: Union[np.ndarray, torch.Tensor], day_idx: int, n_segments: Optional[int] = None
    ) -> torch.Tensor:
        """Single utterance (T, 256) -> predicted CPRs (192, N).

        The segment count N comes from the segment count predictor unless given; emissions use
        the learned prototypes, so no transcript or text is involved.
        """
        device = next(self.parameters()).device
        x = torch.as_tensor(np.asarray(neural, dtype=np.float32) if isinstance(neural, np.ndarray) else neural)
        x = x.float().unsqueeze(0).to(device)
        n = self.content_encoder(x, torch.tensor([int(day_idx)], device=device))
        lengths = torch.tensor([n.shape[1]], device=device)
        n_seg = None if n_segments is None else torch.tensor([int(n_segments)], device=device)
        out = self.aligner(n, lengths, cpr=None, n_segments=n_seg)
        return out.cpr[0, :, : int(out.n_segments[0])]

    # ------------------------------------------------------------------ I/O
    def save(self, path: str, **extra) -> None:
        torch.save({"config": self.config, "state_dict": self.state_dict(), **extra}, path)

    @classmethod
    def from_pretrained(cls, path: str, map_location="cpu") -> "Brain2SpeechNet":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model.eval()
