"""Brain content encoder (Phase 1 of Brain2Speech-Net).

A session-conditioned GRU over windowed neural features that emits frame-level phoneme
logits (the "phoneme-aware representations" n_i in R^41 of the paper). The architecture is
the GRU baseline of Willett et al. (2023) / ``neural_seq_decoder``; parameter names are kept
identical so checkpoints written by that code load without conversion.

The session embedding of the paper is the learned per-session affine input layer
(``dayWeights``, ``dayBias``) applied before the GRU.
"""
from __future__ import annotations

import math
import numbers
import os
import pickle
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .phonemes import N_PHONEMES


class GaussianSmoothing(nn.Module):
    """Depthwise 1-D Gaussian smoothing along time, one kernel per channel."""

    def __init__(self, channels: int, kernel_size: int, sigma: float, dim: int = 1):
        super().__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size], indexing="ij")
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel = kernel * (1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-(((mgrid - mean) / std) ** 2) / 2))
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        self.register_buffer("weight", kernel)
        self.groups = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(x, weight=self.weight, groups=self.groups, padding="same")


class BrainContentEncoder(nn.Module):
    """x (B, T, 256) neural features + session index -> (B, I, 41) phoneme logits.

    Pipeline: Gaussian smoothing -> session-specific affine + softsign -> unfold into
    windows of ``kernel_len`` bins with stride ``stride_len`` -> (bi)GRU -> linear head.
    """

    def __init__(
        self,
        neural_dim: int = 256,
        n_classes: int = N_PHONEMES,
        hidden_dim: int = 1024,
        layer_dim: int = 5,
        n_days: int = 24,
        dropout: float = 0.4,
        stride_len: int = 2,
        kernel_len: int = 32,
        gaussian_smooth_width: float = 2.0,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.config = dict(
            neural_dim=neural_dim, n_classes=n_classes, hidden_dim=hidden_dim, layer_dim=layer_dim,
            n_days=n_days, dropout=dropout, stride_len=stride_len, kernel_len=kernel_len,
            gaussian_smooth_width=gaussian_smooth_width, bidirectional=bidirectional,
        )
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = n_days
        self.strideLen = stride_len
        self.kernelLen = kernel_len
        self.bidirectional = bidirectional

        self.inputLayerNonlinearity = nn.Softsign()
        self.unfolder = nn.Unfold((kernel_len, 1), dilation=1, padding=0, stride=stride_len)
        self.gaussianSmoother = GaussianSmoothing(neural_dim, 20, gaussian_smooth_width, dim=1)

        # Session embedding: one affine transform per recording session, initialised to identity.
        self.dayWeights = nn.Parameter(torch.randn(n_days, neural_dim, neural_dim))
        self.dayBias = nn.Parameter(torch.zeros(n_days, 1, neural_dim))
        for d in range(n_days):
            self.dayWeights.data[d] = torch.eye(neural_dim)

        self.gru_decoder = nn.GRU(
            neural_dim * kernel_len, hidden_dim, layer_dim,
            batch_first=True, dropout=dropout, bidirectional=bidirectional,
        )
        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Unused in forward(); kept so that original neural_seq_decoder checkpoints load strictly.
        for d in range(n_days):
            setattr(self, f"inpLayer{d}", nn.Linear(neural_dim, neural_dim))
        for d in range(n_days):
            layer = getattr(self, f"inpLayer{d}")
            layer.weight = nn.Parameter(layer.weight + torch.eye(neural_dim))

        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc_decoder_out = nn.Linear(out_dim, n_classes + 1)  # +1 for the CTC blank

    @property
    def n_outputs(self) -> int:
        return self.n_classes + 1

    def output_length(self, n_bins: torch.Tensor) -> torch.Tensor:
        """Number of frames the unfold produces for ``n_bins`` input bins."""
        return torch.div(n_bins - self.kernelLen, self.strideLen, rounding_mode="floor") + 1

    def ctc_length(self, n_bins: torch.Tensor) -> torch.Tensor:
        """Valid frame count used for CTC and for HMM training (Willett convention, one less than
        :meth:`output_length` when the window tiles exactly)."""
        return ((n_bins - self.kernelLen) / self.strideLen).to(torch.long)

    def forward(self, x: torch.Tensor, day_idx: torch.Tensor) -> torch.Tensor:
        x = self.gaussianSmoother(x.permute(0, 2, 1)).permute(0, 2, 1)
        w = torch.index_select(self.dayWeights, 0, day_idx)
        b = torch.index_select(self.dayBias, 0, day_idx)
        x = self.inputLayerNonlinearity(torch.einsum("btd,bdk->btk", x, w) + b)
        x = self.unfolder(torch.unsqueeze(x.permute(0, 2, 1), 3)).permute(0, 2, 1)
        n_dirs = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.layer_dim * n_dirs, x.size(0), self.hidden_dim, device=x.device, dtype=x.dtype)
        hid, _ = self.gru_decoder(x, h0)
        return self.fc_decoder_out(hid)

    # ------------------------------------------------------------------ I/O
    def save(self, path: str, **extra) -> None:
        torch.save({"config": self.config, "state_dict": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "BrainContentEncoder":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model

    @classmethod
    def from_neural_seq_decoder(cls, model_dir: str, n_days: int = 24, map_location="cpu") -> "BrainContentEncoder":
        """Load a ``neural_seq_decoder``-style directory (``args`` pickle + ``modelWeights``)."""
        with open(os.path.join(model_dir, "args"), "rb") as f:
            args = pickle.load(f)
        model = cls(
            neural_dim=args["nInputFeatures"], n_classes=args["nClasses"], hidden_dim=args["nUnits"],
            layer_dim=args["nLayers"], n_days=n_days, dropout=args["dropout"],
            stride_len=args["strideLen"], kernel_len=args["kernelLen"],
            gaussian_smooth_width=args["gaussianSmoothWidth"], bidirectional=args["bidirectional"],
        )
        state = torch.load(os.path.join(model_dir, "modelWeights"), map_location=map_location, weights_only=False)
        model.load_state_dict(state)
        return model


def build_content_encoder(cfg: Optional[dict] = None) -> BrainContentEncoder:
    return BrainContentEncoder(**(cfg or {}))
