"""CPU smoke tests on random data: ``python -m pytest tests`` (no data or checkpoints needed)."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain2speech_net.aligner import init_prototypes_kmeans, masked_cpr_mse, segment_count_loss  # noqa: E402
from brain2speech_net.data import collate, load_session  # noqa: E402
from brain2speech_net.model import Brain2SpeechNet  # noqa: E402
from brain2speech_net.phonemes import ids_to_phonemes, phonemes_to_ids  # noqa: E402

ENC = dict(neural_dim=16, hidden_dim=32, layer_dim=2, n_days=3, dropout=0.0, kernel_len=8, stride_len=2)
ALN = dict(d_frame=32, d_cpr=192, d_emit=192, d_state_emb=16, max_states=24, max_segments=20)


def tiny_batch(B=3, seed=0):
    rng = np.random.default_rng(seed)
    items = []
    for b in range(B):
        T, N, P = int(rng.integers(60, 90)), int(rng.integers(5, 15)), int(rng.integers(3, 8))
        items.append({
            "index": b, "neural": rng.standard_normal((T, 16)).astype(np.float32), "day_idx": b % 3,
            "cpr": rng.standard_normal((N, 192)).astype(np.float32),
            "phoneme": rng.integers(1, 41, size=P).astype(np.int64),
        })
    return items, collate(items)


def test_training_step_and_gradients():
    torch.manual_seed(0)
    model = Brain2SpeechNet(encoder=ENC, aligner=ALN)
    _, batch = tiny_batch()
    lengths = model.content_encoder.ctc_length(batch["n_bins"])
    out = model(batch["neural"], batch["day_idx"], lengths, cpr=batch["cpr"], n_segments=batch["n_segments"])
    al = out["aligner"]
    assert al.cpr.shape == batch["cpr"].shape
    assert torch.isfinite(al.log_z).all()
    # posteriors: every valid frame belongs to exactly one state (segments + EOS)
    l_mse = masked_cpr_mse(al.cpr, batch["cpr"], batch["n_segments"])
    l_len = segment_count_loss(al.seg_count_logits, batch["n_segments"])
    l_mse.backward()
    assert model.content_encoder.gru_decoder.weight_ih_l0.grad is not None
    assert model.aligner.frame_enc_len[0].weight.grad is None  # length branch is detached from L_mse
    l_len.backward()
    assert model.aligner.length_net.mlp[0].weight.grad is not None
    assert model.content_encoder.gru_decoder.weight_ih_l0.grad.abs().sum() > 0


def test_posteriors_are_monotonic_alignment():
    torch.manual_seed(1)
    model = Brain2SpeechNet(encoder=ENC, aligner=ALN).eval()
    _, batch = tiny_batch(B=1)
    lengths = model.content_encoder.ctc_length(batch["n_bins"])
    with torch.no_grad():
        al = model(batch["neural"], batch["day_idx"], lengths, cpr=batch["cpr"])["aligner"]
    g = al.gamma[0, : int(lengths[0])]
    N = int(batch["n_segments"][0])
    # each segment receives mass, and over the frames that sit in a segment state (not yet in
    # the absorbing EOS state) the expected segment index never decreases
    assert (g[:, :N].sum(0) > 0).all()
    in_segment = g[:, :N].sum(1) > 0.5
    expected_state = ((g[:, :N] * torch.arange(N)).sum(1) / g[:, :N].sum(1).clamp_min(1e-8))[in_segment]
    assert in_segment.sum() >= N
    assert (expected_state[1:] - expected_state[:-1] > -1e-3).all()


def test_text_free_inference_and_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = Brain2SpeechNet(encoder=ENC, aligner=ALN).eval()
    items, _ = tiny_batch(B=1)
    init_prototypes_kmeans(model.aligner, [it["cpr"] for it in items] * 4, n_init=1, verbose=False)
    cpr = model.predict_cpr(items[0]["neural"], items[0]["day_idx"])
    assert cpr.shape[0] == 192 and 1 <= cpr.shape[1] <= ALN["max_segments"]
    cpr_n = model.predict_cpr(items[0]["neural"], items[0]["day_idx"], n_segments=7)
    assert cpr_n.shape == (192, 7)
    path = tmp_path / "m.pt"
    model.save(str(path))
    reloaded = Brain2SpeechNet.from_pretrained(str(path))
    assert torch.allclose(reloaded.predict_cpr(items[0]["neural"], items[0]["day_idx"], n_segments=7), cpr_n, atol=1e-5)


def test_bucket_padding_leaves_cprs_unchanged():
    """The CUDA-graph path zero-pads frames and segments to a bucket; outputs must not change."""
    from brain2speech_net.cuda_graphs import _mha_fastpath_disabled

    torch.manual_seed(0)
    model = Brain2SpeechNet(encoder=ENC, aligner=ALN).eval()
    items, _ = tiny_batch(B=1)
    al, N = model.aligner, 9
    with torch.inference_mode():
        ref = model.predict_cpr(items[0]["neural"], items[0]["day_idx"], n_segments=N)
        n = model.content_encoder(torch.as_tensor(items[0]["neural"])[None], torch.tensor([items[0]["day_idx"]]))
        I = n.shape[1]
        n_pad = torch.cat([n, torch.zeros(1, 13, n.shape[2])], dim=1)
        with _mha_fastpath_disabled():
            out = al.predict_segments(al.frame_enc(n_pad), torch.tensor([I]), torch.tensor([N]), ALN["max_states"])[0]
    assert torch.allclose(out[0, :, :N], ref, atol=1e-5)
    assert torch.all(out[0, :, N:] == 0)


def test_phoneme_ids_roundtrip():
    phones = ["HH", "AH", "L", "OW", "SIL"]
    assert ids_to_phonemes(phonemes_to_ids(phones)) == phones
    assert phonemes_to_ids(["AA", "SIL"]) == [1, 40]


def test_competition_session_loader(tmp_path):
    scipy_io = pytest.importorskip("scipy.io")
    rng = np.random.default_rng(0)
    n = 4
    tx1, sp = np.empty((1, n), dtype=object), np.empty((1, n), dtype=object)
    for i in range(n):
        T = int(rng.integers(30, 50))
        tx1[0, i] = rng.poisson(2.0, size=(T, 256)).astype(np.float64)
        sp[0, i] = rng.gamma(2.0, size=(T, 256))
    path = tmp_path / "t12.2022.04.28.mat"
    scipy_io.savemat(str(path), {"sentenceText": np.array(["Hello there.", "How are you?", "Fine.", "Ok"]),
                                 "tx1": tx1, "spikePow": sp, "blockIdx": np.array([[1], [1], [2], [2]])})
    sess = load_session(str(path))
    assert sess["transcripts"][1] == "How are you?"
    assert sess["features"][0].shape[1] == 256
    block1 = np.concatenate(sess["features"][:2])
    assert np.allclose(block1.mean(0), 0, atol=1e-5) and np.allclose(block1.std(0), 1, atol=1e-3)


def test_vits_cpr_to_speech_random_weights(tmp_path):
    from brain2speech_net.tts import PretrainedVITS
    from brain2speech_net.vits.models import SynthesizerTrn
    from brain2speech_net.vits.text.symbols import symbols
    import json
    from brain2speech_net.tts import DEFAULT_VITS_CONFIG

    with open(DEFAULT_VITS_CONFIG) as f:
        h = json.load(f)
    torch.manual_seed(0)
    net = SynthesizerTrn(len(symbols), h["data"]["filter_length"] // 2 + 1, h["train"]["segment_size"] // h["data"]["hop_length"], **h["model"])
    ckpt = tmp_path / "vits_random.pth"
    torch.save({"model": net.state_dict()}, str(ckpt))
    vits = PretrainedVITS(str(ckpt))
    wav = vits.cpr_to_speech(torch.randn(192, 6))
    assert wav.ndim == 1 and wav.size > 0 and np.abs(wav).max() <= 1.0
