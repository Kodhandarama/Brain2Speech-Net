#!/usr/bin/env python3
"""Generate speech from neural recordings with a trained Brain2Speech-Net.

Single pass per utterance: neural features -> brain content encoder -> deep-HMM / PPSP
(segment count predicted, prototype emissions) -> CPRs -> frozen VITS -> waveform.
No transcript is used. Real-time factor = generation time / audio duration, timed from the
neural input to the waveform (file I/O excluded).

Open set (official test split, Table 1):
    python scripts/synthesize.py --checkpoint pretrained/brain2speech_net_t12.pt \
        --vits_checkpoint pretrained/pretrained_ljs.pth --data data/test.npz --out_dir outputs/open_set
Training fit (880 seen utterances, Table 2):
    python scripts/synthesize.py ... --data data/train.npz --subset splits/seen880.tsv --out_dir outputs/seen880
"""
import argparse
import csv
import json
import os
import time

import _bootstrap  # noqa: F401
import numpy as np
import torch
from tqdm import tqdm

from brain2speech_net.data import Brain2SpeechDataset, read_subset
from brain2speech_net.model import Brain2SpeechNet
from brain2speech_net.tts import PretrainedVITS, save_wav


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="Brain2Speech-Net checkpoint (.pt)")
    ap.add_argument("--vits_checkpoint", required=True)
    ap.add_argument("--vits_config", default=None)
    ap.add_argument("--data", required=True, help="prepared .npz (train.npz / test.npz)")
    ap.add_argument("--subset", default=None, help="optional (session, trial) TSV, e.g. splits/seen880.tsv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cpu", help="the paper's RTFs were measured on CPU")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: all cores)")
    ap.add_argument("--max_utterances", type=int, default=None)
    ap.add_argument("--oracle_length", action="store_true", help="use the true segment count N (diagnostic only)")
    ap.add_argument("--noise_scale", type=float, default=0.0)
    ap.add_argument("--noise_scale_w", type=float, default=0.0)
    ap.add_argument("--length_scale", type=float, default=1.0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    model = Brain2SpeechNet.from_pretrained(args.checkpoint).to(args.device)
    vits = PretrainedVITS(args.vits_checkpoint, args.vits_config, device=args.device)
    ds = Brain2SpeechDataset(args.data, require_cpr=args.oracle_length, subset=read_subset(args.subset) if args.subset else None)
    n = len(ds) if args.max_utterances is None else min(len(ds), args.max_utterances)
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    with torch.inference_mode():
        for i in tqdm(range(n), desc="synthesising"):
            it = ds[i]
            n_true = int(it["cpr"].shape[0]) if "cpr" in it else None
            t0 = time.perf_counter()
            cpr = model.predict_cpr(it["neural"], it["day_idx"], n_segments=n_true if args.oracle_length else None)
            wav = vits.cpr_to_speech(cpr, args.noise_scale, args.noise_scale_w, args.length_scale)
            elapsed = time.perf_counter() - t0
            dur = len(wav) / vits.sampling_rate
            name = f"{it['session']}_{it['trial']}.wav"
            save_wav(os.path.join(args.out_dir, name), wav, vits.sampling_rate)
            rows.append({
                "file": name, "session": it["session"], "trial": it["trial"], "transcript": it["transcript"],
                "n_segments_pred": int(cpr.shape[1]), "n_segments_true": n_true if n_true is not None else "",
                "time_s": round(elapsed, 6), "duration_s": round(dur, 6), "rtf": round(elapsed / dur, 6) if dur > 0 else "",
            })

    with open(os.path.join(args.out_dir, "manifest.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    rtfs = [r["rtf"] for r in rows if r["rtf"] != ""]
    summary = {"checkpoint": args.checkpoint, "data": args.data, "subset": args.subset, "n_utterances": len(rows),
               "device": args.device, "threads": torch.get_num_threads(), "mean_rtf": float(np.mean(rtfs))}
    with open(os.path.join(args.out_dir, "synthesis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
