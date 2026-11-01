"""Data loading for Brain2Speech-Net.

Raw data: the ``competitionData`` release of Willett et al. (2023), one ``.mat`` file per
recording session and split. We keep the 256 area-6v channels (128 threshold crossings +
128 spike-band power), z-score them within each recording block, and label each sentence
with phonemes exactly as in the original release's preprocessing.

Prepared data (``scripts/prepare_data.py``) is one ``.npz`` per split with object arrays:

    neural_sequence  (T_i, 256) float32   20 ms bins
    day_idx          int                  session index 0..23
    session, trial   str, int             provenance
    transcript       str                  prompted sentence
    phoneme          (P_i,) int           ids in 1..40 (see ``phonemes.py``)
    cpr              (N_i, 192) float32   VITS text-encoder CPR targets
"""
from __future__ import annotations

import csv
import os
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .phonemes import phonemes_to_ids, text_to_phonemes

SESSIONS: List[str] = [
    "t12.2022.04.28", "t12.2022.05.05", "t12.2022.05.17", "t12.2022.05.19",
    "t12.2022.05.24", "t12.2022.05.26", "t12.2022.06.02", "t12.2022.06.07",
    "t12.2022.06.14", "t12.2022.06.16", "t12.2022.06.21", "t12.2022.06.23",
    "t12.2022.06.28", "t12.2022.07.05", "t12.2022.07.14", "t12.2022.07.21",
    "t12.2022.07.27", "t12.2022.07.29", "t12.2022.08.02", "t12.2022.08.11",
    "t12.2022.08.13", "t12.2022.08.18", "t12.2022.08.23", "t12.2022.08.25",
]


# ---------------------------------------------------------------------- raw data
def load_session(mat_path: str) -> Dict[str, list]:
    """One competitionData session file -> block-z-scored features and transcripts."""
    import scipy.io

    dat = scipy.io.loadmat(mat_path)
    feats, texts = [], []
    for i in range(dat["sentenceText"].shape[0]):
        # first 128 columns of each feature type = area 6v
        feats.append(np.concatenate([dat["tx1"][0, i][:, 0:128], dat["spikePow"][0, i][:, 0:128]], axis=1).astype(np.float64))
        texts.append(str(dat["sentenceText"][i]).strip())
    blocks = np.squeeze(dat["blockIdx"])
    for b in np.unique(blocks):
        idx = np.argwhere(blocks == b)[:, 0].astype(np.int32)
        stacked = np.concatenate(feats[idx[0]: idx[-1] + 1], axis=0)
        mean, std = stacked.mean(axis=0, keepdims=True), stacked.std(axis=0, keepdims=True)
        for i in idx:
            feats[i] = (feats[i] - mean) / (std + 1e-8)
    return {"features": [f.astype(np.float32) for f in feats], "transcripts": texts}


def load_competition_split(data_dir: str, split: str, g2p=None) -> List[Dict]:
    """All sessions of one split ("train" or "test"), in session order."""
    if g2p is None:
        from g2p_en import G2p

        g2p = G2p()
    items = []
    for day_idx, session in enumerate(SESSIONS):
        path = os.path.join(data_dir, split, f"{session}.mat")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        sess = load_session(path)
        for trial, (x, text) in enumerate(zip(sess["features"], sess["transcripts"])):
            items.append({
                "neural_sequence": x, "day_idx": day_idx, "session": session, "trial": trial,
                "transcript": text, "phoneme": np.asarray(phonemes_to_ids(text_to_phonemes(text, g2p)), dtype=np.int64),
            })
    return items


def save_npz(path: str, items: Sequence[Dict]) -> None:
    def obj(key):
        arr = np.empty(len(items), dtype=object)
        for i, it in enumerate(items):
            arr[i] = it[key]
        return arr

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out = {
        "neural_sequence": obj("neural_sequence"),
        "day_idx": np.asarray([it["day_idx"] for it in items], dtype=np.int64),
        "session": np.asarray([it["session"] for it in items]),
        "trial": np.asarray([it["trial"] for it in items], dtype=np.int64),
        "transcript": np.asarray([it["transcript"] for it in items]),
        "phoneme": obj("phoneme"),
    }
    if all("cpr" in it for it in items):
        out["cpr"] = obj("cpr")
    np.savez(path, **out)


def read_subset(tsv_path: str) -> List[tuple]:
    """(session, trial) pairs from a split file such as ``splits/seen880.tsv``."""
    with open(tsv_path) as f:
        return [(r["session"], int(r["trial"])) for r in csv.DictReader(f, delimiter="\t")]


# ---------------------------------------------------------------------- dataset
class Brain2SpeechDataset(Dataset):
    """Prepared split. Also reads the original research NPZs (``x_src`` instead of ``cpr``)."""

    def __init__(self, npz_path: str, require_cpr: bool = True, subset: Optional[Sequence[tuple]] = None):
        data = np.load(npz_path, allow_pickle=True)
        self.neural = [np.asarray(a, dtype=np.float32) for a in data["neural_sequence"]]
        self.day_idx = np.asarray(data["day_idx"], dtype=np.int64).reshape(-1)
        n = len(self.neural)
        self.transcript = [str(t) for t in data["transcript"]] if "transcript" in data else [""] * n
        self.session = [str(s) for s in data["session"]] if "session" in data else [SESSIONS[d] for d in self.day_idx]
        if "trial" in data:
            self.trial = np.asarray(data["trial"], dtype=np.int64)
        else:  # session-contiguous order -> index within session
            first = {}
            self.trial = np.asarray([i - first.setdefault(d, i) for i, d in enumerate(self.day_idx)], dtype=np.int64)
        self.phoneme = None
        if "phoneme" in data:
            lens = np.asarray(data["phoneme_len"]) if "phoneme_len" in data else None
            self.phoneme = [
                np.asarray(p, dtype=np.int64)[: int(lens[i])] if lens is not None else np.asarray(p, dtype=np.int64)
                for i, p in enumerate(data["phoneme"])
            ]
        key = "cpr" if "cpr" in data else ("x_src" if "x_src" in data else None)
        self.cpr = [np.asarray(a, dtype=np.float32) for a in data[key]] if key else None
        if require_cpr and self.cpr is None:
            raise KeyError(f"{npz_path} has no CPR targets; run scripts/prepare_data.py with a VITS checkpoint")
        self.indices = list(range(n))
        if subset is not None:
            where = {(s, int(t)): i for i, (s, t) in enumerate(zip(self.session, self.trial))}
            missing = [k for k in subset if tuple(k) not in where]
            if missing:
                raise KeyError(f"{len(missing)} subset entries not found in {npz_path}, e.g. {missing[:3]}")
            self.indices = [where[tuple(k)] for k in subset]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict:
        j = self.indices[i]
        item = {"index": j, "neural": self.neural[j], "day_idx": int(self.day_idx[j]), "transcript": self.transcript[j],
                "session": self.session[j], "trial": int(self.trial[j])}
        if self.phoneme is not None:
            item["phoneme"] = self.phoneme[j]
        if self.cpr is not None:
            item["cpr"] = self.cpr[j]
        return item

    def n_bins(self, i: int) -> int:
        return int(self.neural[self.indices[i]].shape[0])

    def n_segments(self, i: int) -> int:
        return int(self.cpr[self.indices[i]].shape[0]) if self.cpr is not None else 0


def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Pad to the batch maximum. CPRs come out channel-first (B, 192, N) as VITS expects."""
    B = len(batch)
    T = max(b["neural"].shape[0] for b in batch)
    neural = torch.zeros(B, T, batch[0]["neural"].shape[1])
    for i, b in enumerate(batch):
        neural[i, : b["neural"].shape[0]] = torch.from_numpy(b["neural"])
    out = {
        "neural": neural,
        "n_bins": torch.tensor([b["neural"].shape[0] for b in batch], dtype=torch.long),
        "day_idx": torch.tensor([b["day_idx"] for b in batch], dtype=torch.long),
        "index": torch.tensor([b["index"] for b in batch], dtype=torch.long),
    }
    if "cpr" in batch[0]:
        N = max(b["cpr"].shape[0] for b in batch)
        cpr = torch.zeros(B, batch[0]["cpr"].shape[1], N)
        for i, b in enumerate(batch):
            cpr[i, :, : b["cpr"].shape[0]] = torch.from_numpy(b["cpr"].T)
        out["cpr"] = cpr
        out["n_segments"] = torch.tensor([b["cpr"].shape[0] for b in batch], dtype=torch.long)
    if "phoneme" in batch[0]:
        P = max(len(b["phoneme"]) for b in batch)
        ph = torch.zeros(B, P, dtype=torch.long)
        for i, b in enumerate(batch):
            ph[i, : len(b["phoneme"])] = torch.from_numpy(b["phoneme"])
        out["phoneme"] = ph
        out["phoneme_len"] = torch.tensor([len(b["phoneme"]) for b in batch], dtype=torch.long)
    return out


class LengthPoolBatchSampler(Sampler[List[int]]):
    """Length-sorted batching: sort by (frames + segments), shuffle inside pools of
    ``pool_mult * batch_size``, cut into batches and shuffle the batch order every epoch."""

    def __init__(self, lengths: Sequence[int], batch_size: int, pool_mult: int = 2, seed: int = 0):
        self.order = np.argsort(np.asarray(lengths), kind="stable")
        self.batch_size = batch_size
        self.pool = batch_size * pool_mult
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return (len(self.order) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[List[int]]:
        shuffled = np.concatenate([self.rng.permutation(self.order[s: s + self.pool]) for s in range(0, len(self.order), self.pool)])
        batches = [shuffled[b * self.batch_size: (b + 1) * self.batch_size].tolist() for b in range(len(self))]
        for b in self.rng.permutation(len(batches)):
            yield batches[b]
