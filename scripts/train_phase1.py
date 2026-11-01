#!/usr/bin/env python3
"""Phase 1: pre-train the brain content encoder with CTC on phoneme sequences.

    python scripts/train_phase1.py --config configs/phase1_content_encoder.yaml [key=value ...]
"""
import argparse
import os
import time

import _bootstrap  # noqa: F401
import numpy as np
import torch
from torch.utils.data import DataLoader

from brain2speech_net.config import dump_config, load_config
from brain2speech_net.content_encoder import BrainContentEncoder
from brain2speech_net.data import Brain2SpeechDataset, collate
from brain2speech_net.metrics import edit_distance
from brain2speech_net.phonemes import BLANK_ID


def ctc_loss(model, logits, batch, criterion):
    lengths = model.ctc_length(batch["n_bins"].to(logits.device))
    return criterion(logits.log_softmax(2).permute(1, 0, 2), batch["phoneme"].to(logits.device), lengths, batch["phoneme_len"].to(logits.device))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Mean CTC loss and greedy phoneme error rate."""
    model.eval()
    losses, errors, total = [], 0, 0
    for batch in loader:
        logits = model(batch["neural"].to(device), batch["day_idx"].to(device))
        losses.append(ctc_loss(model, logits, batch, criterion).item())
        lengths = model.ctc_length(batch["n_bins"])
        for i in range(logits.shape[0]):
            pred = torch.unique_consecutive(logits[i, : lengths[i]].argmax(-1)).cpu().numpy()
            pred = [int(p) for p in pred if p != BLANK_ID]
            ref = batch["phoneme"][i, : batch["phoneme_len"][i]].tolist()
            errors += edit_distance(ref, pred)
            total += len(ref)
    model.train()
    return float(np.mean(losses)), errors / max(total, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase1_content_encoder.yaml")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("overrides", nargs="*", help="config overrides, e.g. train.n_batches=1000")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    tcfg, device = cfg["train"], args.device
    os.makedirs(cfg["output_dir"], exist_ok=True)
    dump_config(cfg, os.path.join(cfg["output_dir"], "config.yaml"))
    torch.manual_seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])

    train_ds = Brain2SpeechDataset(cfg["data"]["train"], require_cpr=False)
    val_ds = Brain2SpeechDataset(cfg["data"]["val"], require_cpr=False)
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False, collate_fn=collate)

    model = BrainContentEncoder(**cfg["encoder"]).to(device)
    criterion = torch.nn.CTCLoss(blank=BLANK_ID, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr_start"], betas=(0.9, 0.999), eps=tcfg["adam_eps"], weight_decay=tcfg["l2_decay"])
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=tcfg["lr_end"] / tcfg["lr_start"], total_iters=tcfg["n_batches"])

    best_per, t0, it = float("inf"), time.time(), iter(train_loader)
    for step in range(tcfg["n_batches"]):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        x = batch["neural"].to(device)
        if tcfg["white_noise_sd"] > 0:
            x = x + torch.randn_like(x) * tcfg["white_noise_sd"]
        if tcfg["constant_offset_sd"] > 0:
            x = x + torch.randn(x.shape[0], 1, x.shape[2], device=device) * tcfg["constant_offset_sd"]
        loss = ctc_loss(model, model(x, batch["day_idx"].to(device)), batch, criterion)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % tcfg["eval_every"] == 0:
            val_loss, per = evaluate(model, val_loader, criterion, device)
            print(f"step {step:>7d}  train ctc {loss.item():.4f}  val ctc {val_loss:.4f}  val PER {per:.4f}  "
                  f"({(time.time() - t0) / tcfg['eval_every']:.3f} s/step)", flush=True)
            t0 = time.time()
            if per < best_per:
                best_per = per
                model.save(os.path.join(cfg["output_dir"], "content_encoder.pt"), step=step, val_per=per)
    print(f"done; best val greedy PER {best_per:.4f} -> {cfg['output_dir']}/content_encoder.pt")


if __name__ == "__main__":
    main()
