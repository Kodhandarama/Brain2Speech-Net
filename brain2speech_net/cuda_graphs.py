"""CUDA-graph inference for Brain2Speech-Net (GPU, one utterance at a time).

At batch size 1 the deep-HMM aligner is bound by kernel launches, not arithmetic: the
forward-backward recursion issues a few dozen tiny kernels per encoder frame, about 10,000 per
utterance. :class:`GraphedBrain2Speech` captures the aligner (frame encoder -> emissions and
transitions -> forward-backward -> posterior pooling -> PPSP) once per (frame, segment) bucket
and replays it. Inputs are zero-padded to the bucket; the aligner masks frames past I and states
past N, so the predicted CPRs are those of the eager model.

The brain content encoder (cuDNN GRU), the segment count predictor and VITS stay eager: they
issue few, large kernels, and padding would change a bidirectional GRU's output.
"""
from __future__ import annotations

import contextlib
import time
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from .model import Brain2SpeechNet


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m


@contextlib.contextmanager
def _mha_fastpath_disabled():
    """The Transformer fast path turns padded inputs into nested tensors (host sync, dynamic
    shapes), which cannot be captured. The standard path computes the same function."""
    mha = getattr(torch.backends, "mha", None)
    if mha is None or not hasattr(mha, "set_fastpath_enabled"):
        yield
        return
    previous = mha.get_fastpath_enabled()
    mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        mha.set_fastpath_enabled(previous)


class GraphedBrain2Speech:
    """Drop-in for :meth:`Brain2SpeechNet.predict_cpr` that replays the aligner as CUDA graphs.

    Args:
        model: a Brain2SpeechNet on a CUDA device.
        frame_bucket: encoder frames are zero-padded up to a multiple of this.
        segment_bucket: segment counts are padded up to a multiple of this.

    A graph is captured the first time a bucket is used; ``n_graphs`` and ``capture_seconds``
    record that one-off cost.
    """

    def __init__(self, model: Brain2SpeechNet, frame_bucket: int = 16, segment_bucket: int = 32):
        device = next(model.parameters()).device
        if device.type != "cuda":
            raise ValueError("CUDA graphs need the model on a CUDA device")
        self.model = model.eval()
        self.device = device
        self.frame_bucket = frame_bucket
        self.segment_bucket = segment_bucket
        self.pool = torch.cuda.graph_pool_handle()  # shared: graphs never run concurrently
        self.graphs: Dict[Tuple[int, int], tuple] = {}
        self.capture_seconds = 0.0

    @property
    def n_graphs(self) -> int:
        return len(self.graphs)

    def _capture(self, n_frames: int, n_states: int) -> tuple:
        t0 = time.perf_counter()
        al = self.model.aligner
        n_buf = torch.zeros(1, n_frames, al.d_in, device=self.device)
        len_buf = torch.full((1,), n_frames, device=self.device, dtype=torch.long)
        nseg_buf = torch.full((1,), n_states, device=self.device, dtype=torch.long)

        def body():
            return al.predict_segments(al.frame_enc(n_buf), len_buf, nseg_buf, n_states)[0]

        with _mha_fastpath_disabled():
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):  # warm-up outside the graph (cuBLAS handles, workspaces)
                    body()
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                out = body()
        torch.cuda.synchronize()
        self.capture_seconds += time.perf_counter() - t0
        entry = (graph, n_buf, len_buf, nseg_buf, out)
        self.graphs[(n_frames, n_states)] = entry
        return entry

    @torch.inference_mode()
    def predict_cpr(
        self, neural: Union[np.ndarray, torch.Tensor], day_idx: int, n_segments: Optional[int] = None
    ) -> torch.Tensor:
        """Single utterance (T, 256) -> predicted CPRs (192, N), as :meth:`Brain2SpeechNet.predict_cpr`."""
        model, al, device = self.model, self.model.aligner, self.device
        x = torch.as_tensor(np.asarray(neural, dtype=np.float32) if isinstance(neural, np.ndarray) else neural)
        x = x.float().unsqueeze(0).to(device)
        n = model.content_encoder(x, torch.tensor([int(day_idx)], device=device))
        n_frames = n.shape[1]
        if n_segments is None:
            p = al.attention_pool(al.frame_enc_len(n), torch.tensor([n_frames], device=device))
            n_segments = al.predict_segment_count(al.length_net(p))[0]
        n_seg = min(int(n_segments), al.max_states)

        key = (_round_up(n_frames, self.frame_bucket), min(_round_up(n_seg, self.segment_bucket), al.max_states))
        graph, n_buf, len_buf, nseg_buf, out = self.graphs.get(key) or self._capture(*key)
        n_buf[:, :n_frames].copy_(n)
        n_buf[:, n_frames:].zero_()
        len_buf.fill_(n_frames)
        nseg_buf.fill_(n_seg)
        graph.replay()
        return out[0, :, :n_seg].clone()  # the next replay overwrites the graph's output
