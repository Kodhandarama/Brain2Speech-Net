"""Frozen, pretrained VITS (LJSpeech) used by Brain2Speech-Net.

* Training targets: the VITS text encoder turns a prompt into contextual phoneme
  representations (CPRs) z in R^{192 x N}, N = length of the blank-interspersed phoneme
  string. Brain2Speech-Net learns to predict these from neural activity.
* Synthesis: predicted CPRs go through the VITS prior projection, duration predictor,
  flow and HiFi-GAN decoder ("Projection", "Duration predictor", "Flow + speech decoder"
  in Figure 1), so no text is ever decoded at inference.
* Pseudo-target speech: the participant cannot speak, so reference audio for evaluation
  is VITS speech synthesised from the prompt.
"""
from __future__ import annotations

import json
import os
import re
from typing import List, Optional, Sequence

import numpy as np
import torch

from .vits import commons
from .vits.models import SynthesizerTrn
from .vits.text.symbols import symbols

DEFAULT_VITS_CONFIG = os.path.join(os.path.dirname(__file__), "vits", "configs", "ljs_base.json")


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, HParams(**v) if isinstance(v, dict) else v)

    def __getitem__(self, key):
        return getattr(self, key)

    def keys(self):
        return self.__dict__.keys()


def normalize_text_for_cpr(text: str) -> str:
    """Normalisation applied to a prompt before CPR extraction (punctuation is kept)."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def clean_text_for_speech(text: str) -> str:
    """Normalisation applied to a prompt before synthesising pseudo-target speech."""
    return re.sub(r"[^a-zA-Z\- ']", "", str(text)).replace("--", "").lower()


class PretrainedVITS:
    """Thin wrapper around a frozen VITS ``SynthesizerTrn``."""

    def __init__(self, checkpoint: str, config: Optional[str] = None, device: str = "cpu"):
        with open(config or DEFAULT_VITS_CONFIG) as f:
            self.hps = HParams(**json.load(f))
        hps = self.hps
        self.device = device
        self.model = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **{k: hps.model[k] for k in hps.model.keys()},
        ).to(device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = state.get("model", state)
        own = self.model.state_dict()
        missing = [k for k in own if k not in state]
        own.update({k: v for k, v in state.items() if k in own})
        self.model.load_state_dict(own)
        if missing:
            print(f"[vits] {len(missing)} parameters not in checkpoint (kept at init): {missing[:5]}...")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def sampling_rate(self) -> int:
        return int(self.hps.data.sampling_rate)

    @property
    def cpr_dim(self) -> int:
        return int(self.hps.model.hidden_channels)

    # ------------------------------------------------------------------ text side
    def phonemize(self, texts: Sequence[str]) -> List[str]:
        """english_cleaners2 for a batch of prompts with a single espeak call."""
        from phonemizer import phonemize

        from .vits.text.cleaners import collapse_whitespace, convert_to_ascii, expand_abbreviations, lowercase

        cleaned = [expand_abbreviations(lowercase(convert_to_ascii(t))) for t in texts]
        phones = phonemize(cleaned, language="en-us", backend="espeak", strip=True, preserve_punctuation=True, with_stress=True)
        return [collapse_whitespace(p) for p in phones]

    def _symbol_ids(self, phonemized: str) -> torch.LongTensor:
        from .vits.text import cleaned_text_to_sequence

        ids = cleaned_text_to_sequence(phonemized)
        if self.hps.data.add_blank:
            ids = commons.intersperse(ids, 0)
        return torch.LongTensor(ids)

    @torch.no_grad()
    def texts_to_cpr(self, texts: Sequence[str]) -> List[np.ndarray]:
        """Prompts -> list of (N_i, 192) CPR arrays (the training targets)."""
        out = []
        for ph in self.phonemize([normalize_text_for_cpr(t) for t in texts]):
            ids = self._symbol_ids(ph).to(self.device)
            x, _, _, _ = self.model.enc_p(ids.unsqueeze(0), torch.LongTensor([ids.numel()]).to(self.device))
            out.append(x[0].T.cpu().numpy().astype(np.float32))
        return out

    @torch.no_grad()
    def texts_to_speech(self, texts: Sequence[str], noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0) -> List[np.ndarray]:
        """Prompts -> pseudo-target waveforms (float32 in [-1, 1])."""
        out = []
        for ph in self.phonemize([clean_text_for_speech(t) for t in texts]):
            ids = self._symbol_ids(ph).to(self.device)
            audio = self.model.infer(
                ids.unsqueeze(0), torch.LongTensor([ids.numel()]).to(self.device),
                noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=length_scale,
            )[0][0, 0]
            out.append(np.clip(audio.float().cpu().numpy(), -1.0, 1.0))
        return out

    # ------------------------------------------------------------------ CPR -> speech
    @torch.no_grad()
    def cpr_to_speech(self, cpr: torch.Tensor, noise_scale=0.0, noise_scale_w=0.0, length_scale=1.0) -> np.ndarray:
        """(192, N) or (1, 192, N) CPRs -> waveform, through VITS projection, duration predictor, flow, decoder."""
        m = self.model
        x = cpr.unsqueeze(0) if cpr.dim() == 2 else cpr
        x = x.to(self.device).float()
        x_mask = torch.unsqueeze(commons.sequence_mask(torch.LongTensor([x.size(2)]).to(self.device), x.size(2)), 1).to(x.dtype)
        m_p, logs_p = torch.split(m.enc_p.proj(x) * x_mask, m.enc_p.out_channels, dim=1)
        if m.use_sdp:
            logw = m.dp(x, x_mask, g=None, reverse=True, noise_scale=float(noise_scale_w))
        else:
            logw = m.dp(x, x_mask, g=None)
        w_ceil = torch.ceil(torch.exp(logw) * x_mask * float(length_scale))
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn = commons.generate_path(w_ceil, torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1))
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * float(noise_scale)) * y_mask
        z = m.flow(z_p, y_mask, g=None, reverse=True)
        audio = m.dec(z * y_mask, g=None).squeeze(1).cpu().numpy().reshape(-1).astype(np.float32)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak
        return np.clip(audio, -1.0, 1.0)


def save_wav(path: str, audio: np.ndarray, sampling_rate: int) -> None:
    from scipy.io import wavfile

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    wavfile.write(path, sampling_rate, (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16))
