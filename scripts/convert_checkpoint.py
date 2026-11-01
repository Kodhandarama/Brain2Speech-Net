#!/usr/bin/env python3
"""Package checkpoints written by the original research code into one Brain2Speech-Net file.

The research code saved the fine-tuned encoder as a ``neural_seq_decoder`` directory
(``args`` + ``modelWeights``) and the aligner / CTC head in a separate ``.pt``
(``alignment_module``, ``ctc_head``, ``len_max_bins``).

    python scripts/convert_checkpoint.py --encoder_dir FT_GRU/<run> \
        --aligner_checkpoint checkpoints/<run>/last.pt --out pretrained/brain2speech_net_t12.pt
"""
import argparse
import os
import pickle

import _bootstrap  # noqa: F401
import torch

from brain2speech_net.content_encoder import BrainContentEncoder
from brain2speech_net.model import Brain2SpeechNet


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder_dir", required=True, help="directory with args + modelWeights")
    ap.add_argument("--aligner_checkpoint", required=True, help=".pt with alignment_module / ctc_head")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_days", type=int, default=24)
    ap.add_argument("--clamp_eps", type=float, default=1e-3, help="transition probability floor used in training")
    args = ap.parse_args()

    with open(os.path.join(args.encoder_dir, "args"), "rb") as f:
        print("encoder args:", pickle.load(f))
    encoder = BrainContentEncoder.from_neural_seq_decoder(args.encoder_dir, n_days=args.n_days)

    ckpt = torch.load(args.aligner_checkpoint, map_location="cpu", weights_only=False)
    aligner_state = {k[len("aligner."):]: v for k, v in ckpt["alignment_module"].items() if k.startswith("aligner.")}
    if ckpt.get("ctc_head") is None:
        raise KeyError("aligner checkpoint has no ctc_head")
    aligner_cfg = dict(
        d_in=encoder.n_outputs,
        d_frame=aligner_state["frame_enc.0.weight"].shape[0],
        d_cpr=aligner_state["tgt_proj.weight"].shape[1],
        d_emit=aligner_state["emit_proto.weight"].shape[1],
        d_state_emb=aligner_state["state_emb.weight"].shape[1],
        max_states=aligner_state["emit_proto.weight"].shape[0],
        max_segments=int(ckpt.get("len_max_bins") or aligner_state["length_net.mlp.6.weight"].shape[0]),
        clamp_eps=args.clamp_eps,
    )
    model = Brain2SpeechNet(encoder=encoder.config, aligner=aligner_cfg)
    model.content_encoder.load_state_dict(encoder.state_dict())
    model.aligner.load_state_dict(aligner_state)  # strict: fails loudly on any architecture mismatch
    model.ctc_head.load_state_dict(ckpt["ctc_head"])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    model.save(args.out, source={"encoder_dir": args.encoder_dir, "aligner_checkpoint": args.aligner_checkpoint,
                                 "epoch": ckpt.get("epoch"), "global_step": ckpt.get("global_step")})
    n_params = sum(p.numel() for p in model.parameters())
    print(f"saved {args.out}: {n_params / 1e6:.1f} M parameters, aligner config {aligner_cfg}, epoch {ckpt.get('epoch')}")


if __name__ == "__main__":
    main()
