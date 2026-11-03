"""Deep-HMM aligner and Posterior-Pooled Segment Predictor (Phase 2 of Brain2Speech-Net).

Maps the I phoneme-aware frames n_i of the brain content encoder onto the N contextual
phoneme representations (CPRs) of the VITS text encoder with a left-to-right, no-skip HMM
whose states are CPR *segment positions*, then pools frames into segments with the HMM
posteriors and predicts the CPRs.

Paper notation -> attribute (names match the original research code so its checkpoints load):

    BrainFrameEnc_theta            frame_enc        Linear(41, 256) + 3 residual MLP blocks
    segment count predictor        frame_enc_len -> attention_pool -> length_net
    A    (frame projection)        emit_proj        256 -> 192
    W_tgt (CPR projection)         tgt_proj         192 -> 192
    P    (emission prototypes)     emit_proto       K_max x 192, set by k-means
    v_k  (state embeddings)        state_emb
    h_theta (transition network)   trans_net        MLP 512 -> 256 -> (advance, stop)
    PPSP_phi                       seg_head -> step_context (4-layer Transformer) -> refine_head
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

NEG_INF = -1e30


class ResBlock(nn.Module):
    """Pre-LayerNorm residual MLP block."""

    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d * 2)
        self.fc2 = nn.Linear(d * 2, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.fc2(F.gelu(self.fc1(self.ln(x)))))


class AttentionPooling(nn.Module):
    """Masked attention pooling with a single learned query: (B, T, D) -> (B, D)."""

    def __init__(self, d: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)

    def forward(self, e: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        T = e.shape[1]
        scores = torch.matmul(self.query, e.transpose(1, 2))
        pad = torch.arange(T, device=e.device)[None, :] >= lengths[:, None]
        scores = scores.masked_fill(pad.unsqueeze(1), float("-inf"))
        return torch.matmul(F.softmax(scores, dim=-1), e).squeeze(1)


class LengthNet(nn.Module):
    """Segment count classifier: pooled summary -> logits over N in {1, ..., max_segments}."""

    def __init__(self, d_in: int, max_segments: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, max_segments),
        )

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.mlp(p)


def _frame_encoder(d_in: int, d_frame: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, d_frame),
        ResBlock(d_frame, dropout=0.1),
        ResBlock(d_frame, dropout=0.1),
        ResBlock(d_frame, dropout=0.1),
    )


@dataclass
class AlignerOutput:
    cpr: torch.Tensor               # (B, d_cpr, N_max) predicted CPRs, zero past each utterance's N
    n_segments: torch.Tensor        # (B,) segment count used (given, or predicted at inference)
    seg_count_logits: torch.Tensor  # (B, max_segments) segment count predictor logits
    gamma: torch.Tensor             # (B, I, N_max) posterior state occupancies (soft alignment)
    log_z: torch.Tensor             # (B,) HMM log-likelihood


class DeepHMMAligner(nn.Module):
    """Deep-HMM aligner + PPSP.

    Args:
        d_in: dimension of the phoneme-aware representations (41).
        d_frame: brain frame embedding dimension d_f (256).
        d_cpr: CPR dimension of the VITS text encoder (192).
        d_emit: emission space dimension, the paper's d_state (192).
        d_state_emb: dimension of the state embeddings v_k fed to the transition network.
        max_states: size of the state embedding and prototype tables (K_max).
        max_segments: number of classes of the segment count predictor (longest CPR sequence in the data).
        clamp_eps: probability floor used for transition probabilities.
    """

    def __init__(
        self,
        d_in: int = 41,
        d_frame: int = 256,
        d_cpr: int = 192,
        d_emit: int = 192,
        d_state_emb: int = 128,
        max_states: int = 512,
        max_segments: int = 512,
        clamp_eps: float = 1e-3,
    ):
        super().__init__()
        self.config = dict(
            d_in=d_in, d_frame=d_frame, d_cpr=d_cpr, d_emit=d_emit, d_state_emb=d_state_emb,
            max_states=max_states, max_segments=max_segments, clamp_eps=clamp_eps,
        )
        self.d_in = d_in
        self.d_frame = d_frame
        self.d_cpr = d_cpr
        self.d_state_emb = d_state_emb
        self.max_states = max_states
        self.max_segments = max_segments
        self.clamp_eps = clamp_eps

        self.frame_enc = _frame_encoder(d_in, d_frame)

        # Segment count predictor: its own frame encoder over *detached* encoder outputs, so its
        # cross-entropy loss is optimised independently of the alignment objective.
        self.frame_enc_len = _frame_encoder(d_in, d_frame)
        self.attention_pool = AttentionPooling(d_frame)
        self.length_net = LengthNet(d_frame, max_segments)

        self.state_emb = nn.Embedding(max_states, d_state_emb)
        self.emit_proj = nn.Linear(d_frame, d_emit, bias=False)
        self.tgt_proj = nn.Linear(d_cpr, d_emit, bias=False)
        self.emit_proto = nn.Embedding(max_states, d_emit)
        self.eos_proto = nn.Parameter(torch.randn(d_emit) * 0.02)
        self.trans_net = nn.Sequential(
            nn.Linear(d_frame + d_state_emb, 512), nn.Tanh(),
            nn.Linear(512, 256), nn.Tanh(),
            nn.Linear(256, 2),
        )

        # Posterior-Pooled Segment Predictor
        step_layer = nn.TransformerEncoderLayer(d_model=d_cpr, nhead=4, dim_feedforward=768, batch_first=True, dropout=0.1)
        self.step_context = nn.TransformerEncoder(step_layer, num_layers=4)
        self.seg_head = nn.Sequential(
            nn.Linear(d_frame, d_frame), nn.GELU(), nn.LayerNorm(d_frame), nn.Dropout(0.1),
            nn.Linear(d_frame, d_frame), nn.GELU(), nn.LayerNorm(d_frame), nn.Dropout(0.1),
            nn.Linear(d_frame, d_cpr),
        )
        self.refine_head = nn.Sequential(
            nn.Linear(d_cpr, d_cpr), nn.GELU(), nn.LayerNorm(d_cpr), nn.Linear(d_cpr, d_cpr),
        )

    @staticmethod
    def length_branch_parameter_names() -> Tuple[str, ...]:
        """Prefixes of the segment count predictor's parameters (trained by L_len only)."""
        return ("length_net", "frame_enc_len", "attention_pool")

    # ------------------------------------------------------------------ HMM pieces
    @staticmethod
    def _reachability_mask(T_max, K_max, lengths_T, n_seg, device) -> torch.Tensor:
        """(B, T, K) mask of states reachable at frame t while still able to finish at N-1."""
        B = lengths_T.shape[0]
        K_b = n_seg.clamp(max=K_max).long().view(B, 1, 1)
        T_b = lengths_T.long().view(B, 1, 1)
        t = torch.arange(T_max, device=device).view(1, T_max, 1)
        k = torch.arange(K_max, device=device).view(1, 1, K_max)
        min_k = ((K_b - 1) - ((T_b - 1) - t)).clamp(min=0)
        max_k = torch.minimum(t, K_b - 1)
        return (k >= min_k) & (k <= max_k) & (t < T_b)

    def _emissions(self, e, K_max, cpr, n_seg, lengths_T) -> torch.Tensor:
        """log E_i(k) = -||A e_i - c_k||^2 with c_k = W_tgt z_k (training) or P[k] (inference).

        Returns (B, T, K_max + 1); the last column is the EOS state.
        """
        B, T_max, _ = e.shape
        device = e.device
        Ae = self.emit_proj(e)
        if cpr is not None:
            C = self.tgt_proj(cpr.transpose(1, 2))
        else:
            C = self.emit_proto(torch.arange(K_max, device=device)).unsqueeze(0).expand(B, -1, -1)
        # float32 avoids catastrophic cancellation in ||a||^2 + ||c||^2 - 2 a.c under bf16
        Ae, C = Ae.float(), C.float()
        Ae_sq = (Ae * Ae).sum(-1, keepdim=True)
        C_sq = (C * C).sum(-1, keepdim=True).transpose(1, 2)
        log_emit_states = (-(Ae_sq + C_sq - 2 * torch.bmm(Ae, C.transpose(1, 2)))).clamp(min=-1e4)
        eos = self.eos_proto.view(1, 1, -1).float()
        log_emit_eos = (-((Ae - eos).pow(2).sum(-1))).clamp(min=-1e4)
        log_emit = torch.cat([log_emit_states, log_emit_eos.unsqueeze(2)], dim=2)
        time_pad = torch.arange(T_max, device=device)[None, :] >= lengths_T[:, None]
        state_pad = torch.arange(K_max, device=device)[None, :] >= n_seg.clamp(max=K_max)[:, None]
        log_emit[:, :, :K_max] = log_emit[:, :, :K_max].masked_fill(state_pad.unsqueeze(1), NEG_INF)
        return log_emit.masked_fill(time_pad.unsqueeze(2), NEG_INF)

    def _transitions(self, e, K_max, n_seg) -> Tuple[torch.Tensor, torch.Tensor]:
        """tau_adv, tau_stop = h_theta(e_i, v_k), each (B, T-1, K_max).

        Stay / advance / stop come from a 3-way softmax (stay logit fixed to 0). The last
        segment cannot advance (its stop is renormalised against stay) and only it may stop.
        """
        B, T_max, _ = e.shape
        device, dtype = e.device, e.dtype
        if T_max < 2:
            z = torch.zeros((B, 0, K_max), device=device, dtype=dtype)
            return z, z
        v = self.state_emb(torch.arange(K_max, device=device))
        # First layer on [e_i; v_k] split as W_e e_i + W_v v_k + b: one product per frame and per
        # state instead of one per (frame, state) pair.
        lin1 = self.trans_net[0]
        h_e = F.linear(e[:, :-1, :], lin1.weight[:, : self.d_frame], lin1.bias)
        h_v = F.linear(v, lin1.weight[:, self.d_frame:])
        logits2 = self.trans_net[1:](h_e.unsqueeze(2) + h_v.view(1, 1, K_max, -1))
        logits3 = torch.cat([logits2, torch.zeros(B, T_max - 1, K_max, 1, device=device, dtype=logits2.dtype)], dim=-1)
        probs3 = F.softmax(logits3, dim=-1)
        tau_adv_raw, tau_stop_raw = probs3[..., 0], probs3[..., 1]

        K_b = n_seg.clamp(max=K_max)
        last = (K_b - 1).clamp(min=0)
        k_idx = torch.arange(K_max, device=device).view(1, 1, -1)
        is_last = (k_idx == last.view(B, 1, 1)) & (K_b > 0).view(B, 1, 1).expand(B, T_max - 1, K_max)
        last_idx = last.view(B, 1, 1).expand(B, T_max - 1, 1)
        last_adv = torch.gather(tau_adv_raw, 2, last_idx).squeeze(2)
        last_stop = torch.gather(tau_stop_raw, 2, last_idx).squeeze(2)
        stay_last = (1.0 - last_adv - last_stop).clamp_min(self.clamp_eps)
        stop_last = (last_stop / (last_stop + stay_last).clamp_min(self.clamp_eps)).clamp(self.clamp_eps, 1.0 - self.clamp_eps)
        zero = torch.zeros(1, device=device, dtype=tau_adv_raw.dtype)
        tau_adv = torch.where(is_last, zero, tau_adv_raw)
        tau_stop = torch.where(is_last, stop_last.unsqueeze(2).expand_as(tau_stop_raw), tau_stop_raw)
        invalid = k_idx >= K_b.view(B, 1, 1).expand(B, T_max - 1, K_max)
        tau_adv = torch.where(invalid, zero, tau_adv).clamp(self.clamp_eps, 1.0 - self.clamp_eps)
        tau_stop = torch.where(invalid, zero, tau_stop).clamp(self.clamp_eps, 1.0 - self.clamp_eps)
        return tau_adv, tau_stop

    def _forward_backward(self, log_emit, tau_adv, tau_stop, n_seg, lengths_T):
        """Log-space forward-backward over K segment states + a self-absorbing EOS state.

        Returns log_gamma (B, T, K+1) and log Z (B,).
        """
        device, dtype = log_emit.device, log_emit.dtype
        B, T_max, Kp1 = log_emit.shape
        K_max = Kp1 - 1
        EOS = K_max
        neg_inf = torch.full((), NEG_INF, device=device, dtype=dtype)
        eps = self.clamp_eps

        log_stay = torch.log((1.0 - tau_adv - tau_stop).clamp(eps, 1.0 - eps))
        log_adv = torch.log(tau_adv.clamp_min(eps))
        log_stop = torch.log(tau_stop.clamp_min(eps))
        # EOS can only be entered from the final segment N-1
        last = (n_seg.clamp(max=K_max).long() - 1).clamp(min=0)
        allow_stop = torch.arange(K_max, device=device)[None, :] == last[:, None]
        log_stop = torch.where(allow_stop.unsqueeze(1).expand_as(log_stop), log_stop, neg_inf)
        reach = self._reachability_mask(T_max, K_max, lengths_T, n_seg, device)
        time_pad = torch.arange(T_max, device=device)[None, :] >= lengths_T[:, None]
        k_all = torch.arange(K_max + 1, device=device)

        # ---- forward (alpha) ----
        prev = torch.cat([log_emit[:, 0, 0:1], torch.full((B, K_max), NEG_INF, device=device, dtype=dtype)], dim=1)
        alpha_rows = [prev]
        for t in range(1, T_max):
            stay = prev[:, :K_max] + log_stay[:, t - 1, :]
            if K_max > 1:
                adv = torch.cat([neg_inf.expand(B, 1), prev[:, : K_max - 1] + log_adv[:, t - 1, : K_max - 1]], dim=-1)
            else:
                adv = neg_inf.expand(B, K_max)
            non_eos = torch.logaddexp(stay, adv)
            eos = torch.logaddexp(torch.logsumexp(prev[:, :K_max] + log_stop[:, t - 1, :], dim=1), prev[:, EOS])
            cur = torch.cat([non_eos, eos.unsqueeze(1)], dim=-1) + log_emit[:, t, :]
            cur = torch.cat([torch.where(reach[:, t, :], cur[:, :K_max], neg_inf), cur[:, EOS:]], dim=-1)
            cur = torch.where(time_pad[:, t].unsqueeze(1), prev, cur)
            alpha_rows.append(cur)
            prev = cur
        log_alpha = torch.stack(alpha_rows, dim=1)
        log_z = log_alpha[torch.arange(B, device=device), lengths_T - 1, EOS]

        # ---- backward (beta) ----
        ends_last = (lengths_T - 1 == T_max - 1).unsqueeze(1) & (k_all == EOS).unsqueeze(0)
        beta_rows: List[Optional[torch.Tensor]] = [None] * T_max
        beta_rows[T_max - 1] = torch.where(
            ends_last, torch.zeros(1, device=device, dtype=dtype), torch.full((B, K_max + 1), NEG_INF, device=device, dtype=dtype)
        )
        for t in range(T_max - 2, -1, -1):
            nb, ne = beta_rows[t + 1], log_emit[:, t + 1, :]
            stay = log_stay[:, t, :] + ne[:, :K_max] + nb[:, :K_max]
            if K_max > 1:
                adv = torch.cat([log_adv[:, t, : K_max - 1] + ne[:, 1:K_max] + nb[:, 1:K_max], neg_inf.expand(B, 1)], dim=-1)
            else:
                adv = neg_inf.expand(B, K_max)
            stop = log_stop[:, t, :] + ne[:, EOS:EOS + 1] + nb[:, EOS:EOS + 1]
            beta_non = torch.where(reach[:, t, :], torch.logaddexp(torch.logaddexp(stay, adv), stop), neg_inf)
            beta_t = torch.cat([beta_non, (ne[:, EOS] + nb[:, EOS]).unsqueeze(1)], dim=-1)
            beta_t = torch.where(time_pad[:, t].unsqueeze(1), nb, beta_t)
            ends_here = (lengths_T - 1 == t).unsqueeze(1) & (k_all == EOS).unsqueeze(0)
            beta_rows[t] = torch.where(ends_here, torch.zeros(1, device=device, dtype=dtype), beta_t)
        log_beta = torch.stack(beta_rows, dim=1)

        log_gamma = log_alpha + log_beta - log_z.view(B, 1, 1)
        return log_gamma, log_z

    # ------------------------------------------------------------------ forward
    def predict_segment_count(self, seg_count_logits: torch.Tensor) -> torch.Tensor:
        return (seg_count_logits.argmax(dim=-1) + 1).clamp(max=self.max_states)

    def forward(
        self,
        n: torch.Tensor,
        lengths: torch.Tensor,
        cpr: Optional[torch.Tensor] = None,
        n_segments: Optional[torch.Tensor] = None,
    ) -> AlignerOutput:
        """
        Args:
            n: (B, I, d_in) phoneme-aware representations from the brain content encoder.
            lengths: (B,) number of valid frames I per utterance.
            cpr: (B, d_cpr, N) ground-truth CPRs (training). If given, emissions use W_tgt z_k.
                 If None, emissions use the prototypes P[k] (inference).
            n_segments: (B,) segment counts N. Defaults to ``cpr.shape[-1]`` when ``cpr`` is
                 given, otherwise to the segment count predictor's argmax.
        """
        B = n.shape[0]
        device = n.device
        e = self.frame_enc(n)
        p = self.attention_pool(self.frame_enc_len(n.detach()), lengths)
        seg_count_logits = self.length_net(p)

        if cpr is not None:
            K_max = cpr.shape[-1]
            if n_segments is None:
                n_segments = torch.full((B,), K_max, device=device, dtype=torch.long)
        elif n_segments is not None:
            K_max = int(n_segments.max().item())
        else:
            n_segments = self.predict_segment_count(seg_count_logits)
            K_max = int(n_segments.max().item())
        K_max = min(K_max, self.max_states)
        n_seg = n_segments.clamp(max=K_max).long()
        cpr_pred, gamma, log_z = self.predict_segments(e, lengths, n_seg, K_max, cpr)
        return AlignerOutput(cpr=cpr_pred, n_segments=n_seg, seg_count_logits=seg_count_logits, gamma=gamma, log_z=log_z)

    def predict_segments(
        self, e: torch.Tensor, lengths: torch.Tensor, n_seg: torch.Tensor, K_max: int, cpr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """HMM alignment, posterior pooling and PPSP for frame embeddings e (B, I, d_frame).

        Shapes depend only on (I, K_max), with no host synchronisation, so the call can be
        captured in a CUDA graph (see :mod:`brain2speech_net.cuda_graphs`).
        Returns predicted CPRs (B, d_cpr, K_max), posteriors gamma (B, I, K_max) and log Z (B,).
        """
        device = e.device
        log_emit = self._emissions(e, K_max, cpr, n_seg, lengths)
        tau_adv, tau_stop = self._transitions(e, K_max, n_seg)
        # The forward-backward recursion runs in float32: bf16 error accumulates over I steps.
        log_gamma, log_z = self._forward_backward(log_emit.float(), tau_adv.float(), tau_stop.float(), n_seg, lengths)

        # Posterior pooling: z_k = sum_i gamma_{i,k} e_i / sum_i gamma_{i,k}
        T_dim = log_gamma.shape[1]
        gamma = torch.exp(log_gamma).clamp_min(1e-12)[:, :, :K_max]
        time_pad = torch.arange(T_dim, device=device)[None, :, None] >= lengths[:, None, None]
        seg_pad = torch.arange(K_max, device=device)[None, None, :] >= n_seg[:, None, None]
        gamma = gamma.masked_fill(time_pad | seg_pad, 0.0)
        z = torch.bmm(gamma.transpose(1, 2), e) / (gamma.sum(dim=1).unsqueeze(2) + 1e-8)

        # Segment prediction
        z = self.seg_head(z)
        seg_padding = torch.arange(K_max, device=device)[None, :] >= n_seg[:, None]
        z = self.step_context(z, src_key_padding_mask=seg_padding)
        z = z + self.refine_head(z)
        cpr_pred = z.transpose(1, 2).masked_fill(seg_padding.unsqueeze(1), 0.0)
        if cpr is not None and cpr_pred.shape[2] < cpr.shape[2]:
            cpr_pred = F.pad(cpr_pred, (0, cpr.shape[2] - cpr_pred.shape[2]))
        return cpr_pred, gamma, log_z


# ---------------------------------------------------------------------- losses / prototypes
def masked_cpr_mse(pred: torch.Tensor, target: torch.Tensor, n_segments: torch.Tensor) -> torch.Tensor:
    """L_MSE: squared error summed over CPR channels, averaged over valid segments. (B, d, N) inputs."""
    mask = (torch.arange(pred.shape[2], device=pred.device)[None, :] < n_segments[:, None]).unsqueeze(1).float()
    return ((pred - target) * mask).pow(2).sum() / mask.sum().clamp(min=1)


def segment_count_loss(seg_count_logits: torch.Tensor, n_segments: torch.Tensor) -> torch.Tensor:
    """L_len = -log p(N | n): cross-entropy over N in {1, ..., max_segments}."""
    max_segments = seg_count_logits.shape[-1]
    return F.cross_entropy(seg_count_logits.float(), (n_segments.clamp(1, max_segments) - 1).long())


@torch.no_grad()
def init_prototypes_kmeans(
    aligner: DeepHMMAligner, cpr_list: List[np.ndarray], n_init: int = 10, seed: int = 0, verbose: bool = True
) -> None:
    """Set the emission prototypes P to k-means centres (k = K_max) of {W_tgt z_k} over training CPRs.

    ``cpr_list`` holds one (N_i, d_cpr) array per training utterance.
    """
    from sklearn.cluster import KMeans

    segments = np.vstack(cpr_list).astype(np.float32)
    if segments.shape[0] == 0:
        return
    device = aligner.tgt_proj.weight.device
    projected = aligner.tgt_proj(torch.from_numpy(segments).to(device)).float().cpu().numpy()
    k = min(aligner.max_states, projected.shape[0])
    centers = KMeans(n_clusters=k, random_state=seed, n_init=n_init).fit(projected).cluster_centers_.astype(np.float32)
    weight = aligner.emit_proto.weight
    weight.data[:k].copy_(torch.from_numpy(centers).to(device))
    if k < aligner.max_states:
        weight.data[k:].copy_(weight.data[k - 1:k])
    if verbose:
        print(f"[prototypes] k-means: k={k} over {projected.shape[0]} CPR segments")
