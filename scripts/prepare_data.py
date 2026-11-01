#!/usr/bin/env python3
"""Build Brain2Speech-Net inputs from the public Willett et al. (2023) ``competitionData``.

Writes
    <out_dir>/train.npz, <out_dir>/test.npz     neural features, phonemes, transcripts, CPR targets
    <out_dir>/reference_wavs/{test,seen880}/    (with --references) VITS pseudo-target speech used
                                                as the reference for PER / MCD and the ground-truth row

Example:
    python scripts/prepare_data.py --competition_data /path/to/competitionData \
        --vits_checkpoint pretrained/pretrained_ljs.pth --out_dir data --references
"""
import argparse
import csv
import os

import _bootstrap  # noqa: F401
import numpy as np
from tqdm import tqdm

from brain2speech_net.data import Brain2SpeechDataset, load_competition_split, read_subset, save_npz
from brain2speech_net.tts import PretrainedVITS, save_wav

ROOT = _bootstrap.ROOT


def write_references(vits, ds, out_dir, batch=32):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for start in tqdm(range(0, len(ds), batch), desc=f"references -> {out_dir}"):
        items = [ds[i] for i in range(start, min(start + batch, len(ds)))]
        for it, wav in zip(items, vits.texts_to_speech([it["transcript"] for it in items])):
            name = f"{it['session']}_{it['trial']}.wav"
            save_wav(os.path.join(out_dir, name), wav, vits.sampling_rate)
            rows.append({"file": name, "session": it["session"], "trial": it["trial"], "transcript": it["transcript"]})
    with open(os.path.join(out_dir, "manifest.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--competition_data", required=True, help="directory with train/ and test/ session .mat files")
    ap.add_argument("--vits_checkpoint", required=True, help="pretrained LJSpeech VITS generator (pretrained_ljs.pth)")
    ap.add_argument("--vits_config", default=None, help="defaults to the bundled ljs_base.json")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--references", action="store_true", help="also synthesise reference speech for evaluation")
    ap.add_argument("--seen_subset", default=os.path.join(ROOT, "splits", "seen880.tsv"))
    args = ap.parse_args()

    from g2p_en import G2p

    g2p = G2p()
    vits = PretrainedVITS(args.vits_checkpoint, args.vits_config, device=args.device)
    for split in ("train", "test"):
        items = load_competition_split(args.competition_data, split, g2p=g2p)
        print(f"[{split}] {len(items)} utterances from {len(set(it['session'] for it in items))} sessions")
        texts = [it["transcript"] for it in items]
        cprs = []
        for s in tqdm(range(0, len(texts), 256), desc=f"[{split}] CPR targets"):
            cprs.extend(vits.texts_to_cpr(texts[s: s + 256]))
        for it, z in zip(items, cprs):
            it["cpr"] = z
        lens = np.array([z.shape[0] for z in cprs])
        print(f"[{split}] CPR segments per utterance: min {lens.min()} / median {int(np.median(lens))} / max {lens.max()}")
        save_npz(os.path.join(args.out_dir, f"{split}.npz"), items)

    if args.references:
        test = Brain2SpeechDataset(os.path.join(args.out_dir, "test.npz"), require_cpr=False)
        write_references(vits, test, os.path.join(args.out_dir, "reference_wavs", "test"))
        seen = Brain2SpeechDataset(os.path.join(args.out_dir, "train.npz"), require_cpr=False, subset=read_subset(args.seen_subset))
        write_references(vits, seen, os.path.join(args.out_dir, "reference_wavs", "seen880"))


if __name__ == "__main__":
    main()
