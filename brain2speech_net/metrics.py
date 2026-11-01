"""Objective metrics used in the paper.

* WER  - Whisper-medium transcription of the generated speech vs. the prompt; lowercase,
         punctuation stripped; corpus-level (total word errors / total reference words).
* PER  - wav2vec 2.0-Large-LV60 phoneme recogniser; phonemes decoded from the generated wav
         vs. phonemes decoded from the reference (pseudo-target) wav; mean over utterances.
* MCD  - mel-cepstral distortion (dB) on WORLD mel-cepstra (order 39, alpha 0.466, 5 ms
         shift), silent frames removed, DTW-aligned; mean over utterances.
* RTF  - measured in ``scripts/synthesize.py`` (generation time / audio duration).
"""
from __future__ import annotations

import string
from typing import List, Optional, Sequence, Tuple

import numpy as np

PHONEME_MODEL = "excalibur12/wav2vec2-large-lv60_phoneme-timit_english_timit-4k_simplified"
WHISPER_MODEL = "medium"


def edit_distance(a: Sequence, b: Sequence) -> int:
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(dp[j], dp[j - 1], prev)
            prev = temp
    return dp[m]


def normalize_for_wer(text: str) -> str:
    return text.lower().translate(str.maketrans("", "", string.punctuation)).strip()


def corpus_wer(hypotheses: Sequence[str], references: Sequence[str]) -> Tuple[float, int, int]:
    errors = words = 0
    for hyp, ref in zip(hypotheses, references):
        r = normalize_for_wer(ref).split()
        errors += edit_distance(normalize_for_wer(hyp).split(), r)
        words += len(r)
    return (errors / words if words else 0.0), errors, words


# ---------------------------------------------------------------------- WER
def whisper_transcribe(wav_paths: Sequence[str], model_name: str = WHISPER_MODEL, device: Optional[str] = None) -> List[str]:
    import whisper
    from tqdm import tqdm

    model = whisper.load_model(model_name, device=device)
    # fp16=False as in the paper's evaluation (also on GPU), so numbers do not depend on the device
    return [model.transcribe(p, fp16=False, language="en").get("text", "").strip() for p in tqdm(wav_paths, desc="Whisper")]


# ---------------------------------------------------------------------- PER
def _load_16k(path: str, max_sec: float = 20.0):
    import torch
    import torchaudio

    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.squeeze(0)[: int(max_sec * 16000)]
    if wav.numel() == 0:
        raise ValueError(f"empty audio: {path}")
    return wav


def _greedy_ctc(log_probs, blank_id: int) -> List[int]:
    out, prev = [], None
    for p in log_probs.argmax(dim=-1).tolist():
        if p != prev and p != blank_id:
            out.append(int(p))
        prev = p
    return out


def phoneme_error_rates(gen_paths: Sequence[str], ref_paths: Sequence[str], model_name: str = PHONEME_MODEL) -> List[float]:
    """Per-utterance pseudo-PER of generated vs. reference speech (NaN where a file fails)."""
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    from transformers import AutoModelForCTC, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCTC.from_pretrained(model_name).eval()
    blank_id = int(model.config.pad_token_id) if model.config.pad_token_id is not None else 0

    def phones(path):
        inputs = processor(_load_16k(path).numpy(), sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            logits = model(inputs["input_values"], attention_mask=inputs.get("attention_mask")).logits[0].float()
        return _greedy_ctc(F.log_softmax(logits, dim=-1), blank_id)

    pers = []
    for g, r in tqdm(list(zip(gen_paths, ref_paths)), desc="PER"):
        try:
            ref = phones(r)
            pers.append(edit_distance(ref, phones(g)) / len(ref) if ref else float("nan"))
        except Exception as exc:  # keep going; report as NaN like the paper's evaluation
            print(f"[PER] {g}: {exc}")
            pers.append(float("nan"))
    return pers


# ---------------------------------------------------------------------- MCD
MCEP_DIM, MCEP_ALPHA, MCEP_SHIFT_MS, MCEP_FFTL = 39, 0.466, 5, 1024


def _world_mcep(x: np.ndarray, fs: int, f0min: float = 80, f0max: float = 7600):
    import pysptk
    import pyworld as pw
    from scipy.signal import firwin, lfilter

    x = np.asarray(x * np.iinfo(np.int16).max, dtype=np.float64)
    x = lfilter(firwin(255, 70 / (fs // 2), pass_zero=False), 1, x)  # 70 Hz low-cut
    f0, t = pw.harvest(x, fs, f0_floor=f0min, f0_ceil=f0max, frame_period=MCEP_SHIFT_MS)
    sp = pw.cheaptrick(x, f0, t, fs, fft_size=MCEP_FFTL)
    mcep = pysptk.sp2mc(sp, MCEP_DIM, MCEP_ALPHA)
    fftl2 = sp.shape[1] - 1
    power = (sp[:, 0] + sp[:, fftl2] + 2.0 * sp[:, 1:fftl2].sum(axis=1)) / (fftl2 * 2)
    npow = 10.0 * np.log10(power / power.mean())
    return mcep, npow


def mel_cepstral_distortion(ref: np.ndarray, gen: np.ndarray, fs: int, power_threshold: float = -20) -> float:
    import scipy.spatial
    from fastdtw import fastdtw

    ref_mc, ref_pow = _world_mcep(ref, fs)
    gen_mc, gen_pow = _world_mcep(gen, fs)
    ref_mc, gen_mc = ref_mc[ref_pow > power_threshold], gen_mc[gen_pow > power_threshold]
    _, path = fastdtw(gen_mc, ref_mc, dist=scipy.spatial.distance.euclidean)
    path = np.array(path).T
    diff2 = np.sum((gen_mc[path[0]] - ref_mc[path[1]]) ** 2, axis=1)
    return float(np.mean((10.0 / np.log(10.0)) * np.sqrt(2.0 * diff2)))


def mcd_scores(gen_paths: Sequence[str], ref_paths: Sequence[str]) -> List[float]:
    import librosa
    from tqdm import tqdm

    scores = []
    for g, r in tqdm(list(zip(gen_paths, ref_paths)), desc="MCD"):
        try:
            gen, fs = librosa.load(g, sr=None)
            ref, rs = librosa.load(r, sr=None)
            if rs != fs:
                ref = librosa.resample(ref, orig_sr=rs, target_sr=fs)
            scores.append(mel_cepstral_distortion(ref, gen, fs))
        except Exception as exc:
            print(f"[MCD] {g}: {exc}")
            scores.append(float("nan"))
    return scores
