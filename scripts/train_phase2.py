#!/usr/bin/env python3
"""Phase 2: train the deep-HMM aligner + PPSP and fine-tune the brain content encoder.

Losses (paper, Sec. 4):
    L_MSE  CPR regression through the soft HMM alignment   -> encoder, frame encoder, HMM, PPSP
    L_CTC  linear CTC head on the phoneme-aware frames      -> encoder, CTC head  (weight lambda_ctc)
    L_len  segment count cross-entropy on detached frames   -> segment count predictor only
Each parameter group has its own Adam optimiser with linear warm-up + cosine decay.

    python scripts/train_phase2.py --config configs/phase2_brain2speech_net.yaml [key=value ...]
    python scripts/train_phase2.py --config ... --resume checkpoints/phase2/last.pt
"""
import argparse
import math
import os
import time
from contextlib import nullcontext

import _bootstrap  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from brain2speech_net.aligner import init_prototypes_kmeans, masked_cpr_mse, segment_count_loss
from brain2speech_net.config import dump_config, load_config
from brain2speech_net.content_encoder import BrainContentEncoder
from brain2speech_net.data import Brain2SpeechDataset, LengthPoolBatchSampler, collate
from brain2speech_net.model import Brain2SpeechNet
from brain2speech_net.phonemes import BLANK_ID


def compute_losses(model, batch, device, amp, ctc_criterion):
    neural = batch["neural"].to(device, non_blocking=True)
    day_idx = batch["day_idx"].to(device)
    cpr = batch["cpr"].to(device, non_blocking=True)
    n_seg = batch["n_segments"].to(device)
    frame_lengths = model.content_encoder.ctc_length(batch["n_bins"]).to(device)
    with amp():
        out = model(neural, day_idx, frame_lengths, cpr=cpr, n_segments=n_seg)
    al = out["aligner"]
    losses = {
        "mse": masked_cpr_mse(al.cpr.float(), cpr, n_seg),
        "len": segment_count_loss(al.seg_count_logits, n_seg),
    }
    if "phoneme" in batch:
        log_probs = F.log_softmax(out["ctc_logits"].float(), dim=-1).transpose(0, 1)
        plen = batch["phoneme_len"]
        targets = torch.cat([batch["phoneme"][i, : plen[i]] for i in range(len(plen))]).to(device)
        losses["ctc"] = ctc_criterion(log_probs, targets, frame_lengths, plen.to(device))
    with torch.no_grad():
        pred_n = al.seg_count_logits.argmax(-1) + 1
        losses["len_acc"] = (pred_n == n_seg).float().mean()
        losses["len_mae"] = (pred_n - n_seg).abs().float().mean()
    return losses


@torch.no_grad()
def validate(model, loader, device, amp, ctc_criterion):
    model.eval()
    sums, n = {}, 0
    for batch in loader:
        for k, v in compute_losses(model, batch, device, amp, ctc_criterion).items():
            sums[k] = sums.get(k, 0.0) + float(v)
        n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in sums.items()}


def finite(t):
    return bool(torch.isfinite(t).all())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase2_brain2speech_net.yaml")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", default=None, help="last.pt / epoch_*.pt from a previous run")
    ap.add_argument("--wandb_project", default=None, help="log to Weights & Biases if set")
    ap.add_argument("overrides", nargs="*", help="config overrides, e.g. train.epochs=10")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    tcfg, pcfg, device = cfg["train"], cfg["prototypes"], args.device
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    dump_config(cfg, os.path.join(out_dir, "config.yaml"))
    torch.manual_seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])
    use_amp = bool(tcfg.get("amp_bf16", True)) and device.startswith("cuda")
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_amp else nullcontext
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ---------------- data
    full_train = Brain2SpeechDataset(cfg["data"]["train"])
    n_hold = int(cfg["data"].get("val_from_train") or 0)
    if n_hold > 0:
        # seeded permutation of the training split: first N - n_hold train, last n_hold validation
        perm = np.random.default_rng(cfg["data"].get("split_seed", 42)).permutation(len(full_train))
        train_idx, val_idx = perm[: len(full_train) - n_hold].tolist(), perm[len(full_train) - n_hold:].tolist()
        train_ds, val_ds = Subset(full_train, train_idx), Subset(full_train, val_idx)
    else:
        train_ds, val_ds = full_train, Brain2SpeechDataset(cfg["data"]["val"])
        train_idx = list(range(len(full_train)))
    train_cprs = [full_train.cpr[i] for i in train_idx]
    val_base = val_ds.dataset if isinstance(val_ds, Subset) else val_ds
    max_seg_data = max(max(z.shape[0] for z in full_train.cpr), max(z.shape[0] for z in val_base.cpr))
    print(f"train {len(train_ds)} / val {len(val_ds)} utterances; longest CPR sequence {max_seg_data}")

    # ---------------- model
    aligner_cfg = dict(cfg["aligner"])
    if aligner_cfg.get("max_segments") is None:
        aligner_cfg["max_segments"] = max_seg_data
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model = Brain2SpeechNet(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
    else:
        encoder = BrainContentEncoder.load(cfg["encoder_init"])
        model = Brain2SpeechNet(encoder=encoder.config, aligner=aligner_cfg)
        model.content_encoder.load_state_dict(encoder.state_dict())
    model.to(device).train()
    if max_seg_data > model.aligner.max_segments or max_seg_data > model.aligner.max_states:
        raise ValueError(f"longest CPR sequence ({max_seg_data}) exceeds max_segments/max_states")

    # length-sorted batches: similar (frames + segments) per batch keeps padding and the HMM small
    n_frames = model.content_encoder.ctc_length(torch.tensor([full_train.n_bins(i) for i in train_idx]))
    lengths = (n_frames + torch.tensor([full_train.n_segments(i) for i in train_idx])).tolist()
    sampler = LengthPoolBatchSampler(lengths, tcfg["batch_size"], seed=tcfg["seed"])
    train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate, num_workers=tcfg["num_workers"], pin_memory=device.startswith("cuda"))
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False, collate_fn=collate, num_workers=tcfg["num_workers"])

    # ---------------- optimisers (one per parameter group, as in the paper's training code)
    length_prefixes = tuple(f"aligner.{p}" for p in model.aligner.length_branch_parameter_names())
    groups = {"encoder": [], "aligner": [], "length": [], "ctc": []}
    for name, p in model.named_parameters():
        if name.startswith("content_encoder."):
            groups["encoder"].append(p)
        elif name.startswith("ctc_head."):
            groups["ctc"].append(p)
        elif name.startswith(length_prefixes):
            groups["length"].append(p)
        else:
            groups["aligner"].append(p)
    lrs = {"encoder": tcfg["encoder_lr"], "aligner": tcfg["lr"], "length": tcfg["lr"], "ctc": tcfg["lr"]}
    optims = {k: torch.optim.Adam(v, lr=lrs[k]) for k, v in groups.items()}
    total_steps = tcfg["epochs"] * len(sampler)
    warmup = tcfg["warmup_steps"]

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total_steps - warmup, 1)))

    scheds = {k: torch.optim.lr_scheduler.LambdaLR(o, lr_lambda) for k, o in optims.items()}
    start_epoch, step = 0, 0
    if args.resume:
        for k in optims:
            optims[k].load_state_dict(ckpt["optimizers"][k])
            scheds[k].load_state_dict(ckpt["schedulers"][k])
        start_epoch, step = ckpt["epoch"] + 1, ckpt["step"]
        sampler.rng = np.random.default_rng(tcfg["seed"] + start_epoch)
    elif pcfg.get("kmeans_init", True):
        init_prototypes_kmeans(model.aligner, subsample(train_cprs, pcfg), n_init=pcfg["n_init"])

    ctc_criterion = torch.nn.CTCLoss(blank=BLANK_ID, zero_infinity=True, reduction="mean")
    wandb = None
    if args.wandb_project:
        import wandb as _wandb

        wandb = _wandb
        wandb.init(project=args.wandb_project, config=cfg)

    def save(path, epoch):
        torch.save({
            "config": model.config, "state_dict": model.state_dict(), "epoch": epoch, "step": step,
            "optimizers": {k: o.state_dict() for k, o in optims.items()},
            "schedulers": {k: s.state_dict() for k, s in scheds.items()},
        }, path)

    # ---------------- training
    for epoch in range(start_epoch, tcfg["epochs"]):
        t0 = time.time()
        for batch in train_loader:
            losses = compute_losses(model, batch, device, amp, ctc_criterion)
            if not finite(losses["mse"]):
                print(f"[step {step}] non-finite L_mse, skipping batch")
                step += 1
                continue
            for k in ("encoder", "aligner", "ctc"):
                optims[k].zero_grad(set_to_none=True)
            main_loss = losses["mse"] + (tcfg["lambda_ctc"] * losses["ctc"] if "ctc" in losses else 0.0)
            main_loss.backward()
            norms = [torch.nn.utils.clip_grad_norm_(groups[k], tcfg["grad_clip"]) for k in ("encoder", "aligner", "ctc")]
            if all(finite(n) for n in norms):
                for k in ("encoder", "aligner", "ctc"):
                    optims[k].step()
            # segment count predictor: separate graph (detached input), separate optimiser
            optims["length"].zero_grad(set_to_none=True)
            if finite(losses["len"]):
                losses["len"].backward()
                if finite(torch.nn.utils.clip_grad_norm_(groups["length"], tcfg["grad_clip"])):
                    optims["length"].step()
            for s in scheds.values():
                s.step()

            if step % tcfg["log_every"] == 0:
                msg = {f"train/{k}": float(v) for k, v in losses.items()}
                print(f"epoch {epoch + 1} step {step}  " + "  ".join(f"{k.split('/')[1]} {v:.4f}" for k, v in msg.items()), flush=True)
                if wandb:
                    wandb.log({**msg, "epoch": epoch}, step=step)
            step += 1

        if epoch == 0 or (epoch + 1) % tcfg["val_every"] == 0:
            val = validate(model, val_loader, device, amp, ctc_criterion)
            print(f"epoch {epoch + 1} validation  " + "  ".join(f"{k} {v:.4f}" for k, v in val.items()), flush=True)
            if wandb:
                wandb.log({f"val/{k}": v for k, v in val.items()}, step=step)
        save(os.path.join(out_dir, "last.pt"), epoch)
        if (epoch + 1) % tcfg["save_every"] == 0:
            save(os.path.join(out_dir, f"epoch_{epoch + 1:04d}.pt"), epoch)
        # prototypes are re-clustered after the checkpoints of this epoch are written
        if pcfg.get("recluster_every") and (epoch + 1) % pcfg["recluster_every"] == 0:
            init_prototypes_kmeans(model.aligner, subsample(train_cprs, pcfg, seed=epoch), n_init=pcfg["n_init"])
        print(f"epoch {epoch + 1} done in {time.time() - t0:.0f} s", flush=True)

    # final export with prototypes clustered on the final W_tgt
    if pcfg.get("recluster_every") and tcfg["epochs"] % pcfg["recluster_every"] != 0:
        init_prototypes_kmeans(model.aligner, subsample(train_cprs, pcfg, seed=tcfg["epochs"]), n_init=pcfg["n_init"])
    model.save(os.path.join(out_dir, "brain2speech_net.pt"))
    print(f"saved final model to {out_dir}/brain2speech_net.pt")


def subsample(cprs, pcfg, seed=0):
    """Optionally keep a random subset of CPR segments for k-means (``prototypes.max_points``)."""
    max_points = pcfg.get("max_points")
    if not max_points:
        return cprs
    seg = np.vstack(cprs)
    if seg.shape[0] <= max_points:
        return cprs
    return [seg[np.random.default_rng(seed).choice(seg.shape[0], int(max_points), replace=False)]]


if __name__ == "__main__":
    main()
