#!/usr/bin/env python3
"""Score generated speech: WER (Whisper-medium), PER (wav2vec2 phonemes vs. reference speech), MCD.

Both directories need a ``manifest.tsv`` (written by synthesize.py / prepare_data.py --references);
files are paired by name.

    python scripts/evaluate.py --generated outputs/open_set --references data/reference_wavs/test
    # ground-truth speech reference row (WER only):
    python scripts/evaluate.py --generated data/reference_wavs/test --metrics wer
"""
import argparse
import csv
import json
import os

import _bootstrap  # noqa: F401
import numpy as np

from brain2speech_net import metrics


def read_manifest(d):
    with open(os.path.join(d, "manifest.tsv")) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generated", required=True)
    ap.add_argument("--references", default=None, help="reference (pseudo-target) speech; needed for PER and MCD")
    ap.add_argument("--metrics", default="wer,per,mcd")
    ap.add_argument("--whisper_device", default=None)
    ap.add_argument("--out", default=None, help="defaults to <generated>/evaluation.json")
    args = ap.parse_args()
    wanted = set(args.metrics.split(","))
    if wanted & {"per", "mcd"} and not args.references:
        ap.error("--references is required for PER and MCD")

    rows = read_manifest(args.generated)
    gen = [os.path.join(args.generated, r["file"]) for r in rows]
    refs = [os.path.join(args.references, r["file"]) for r in rows] if args.references else None
    if refs:
        missing = [p for p in refs if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"{len(missing)} reference wavs missing, e.g. {missing[:3]}")

    result = {"generated": args.generated, "references": args.references, "n_utterances": len(rows)}
    if "wer" in wanted:
        hyps = metrics.whisper_transcribe(gen, device=args.whisper_device)
        wer, errors, words = metrics.corpus_wer(hyps, [r["transcript"] for r in rows])
        result.update(wer=wer, word_errors=errors, reference_words=words)
        for r, h in zip(rows, hyps):
            r["whisper"] = h
    if "per" in wanted:
        pers = metrics.phoneme_error_rates(gen, refs)
        result.update(per=float(np.nanmean(pers)), per_failed=int(np.isnan(pers).sum()))
        for r, v in zip(rows, pers):
            r["per"] = v
    if "mcd" in wanted:
        mcds = metrics.mcd_scores(gen, refs)
        result.update(mcd=float(np.nanmean(mcds)), mcd_failed=int(np.isnan(mcds).sum()))
        for r, v in zip(rows, mcds):
            r["mcd"] = v
    summary = os.path.join(args.generated, "synthesis_summary.json")
    if os.path.exists(summary):
        with open(summary) as f:
            result["rtf"] = json.load(f).get("mean_rtf")

    out = args.out or os.path.join(args.generated, "evaluation.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.splitext(out)[0] + "_per_utterance.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
