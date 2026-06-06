import os
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from utils.constants import get_best_gpu

os.environ["CUDA_VISIBLE_DEVICES"] = str(get_best_gpu())

import argparse
import torch
from transformers import AutoTokenizer
from gector import GECToR
from gector.predict import predict_detect_only
from typing import List, Dict


KEEP_LABEL   = "$KEEP"
ERROR_LABELS = {"$WRONG_CAP", "$WRONG_DECL", "$WRONG_COMP"}

def visualizer(iteration_log: List[List[Dict]]) -> str:
    strs = ""
    for sent_id, sent in enumerate(iteration_log):
        strs += f"=== Sentence {sent_id} ===\n"
        itr = sent[0]
        if itr["tag"] is None:
            tokens = [t for t in itr["src"] if t != "$START"]
            strs += "  " + " ".join(tokens) + "\n"
            strs += "  [all $KEEP — no errors detected]\n"
        else:
            src_str = "|"
            tag_str = "|"
            for tok, tag in zip(itr["src"], itr["tag"]):
                if tok == "$START":
                    continue
                max_len = max(len(tok), len(tag)) + 1
                src_str += tok + " " * (max_len - len(tok)) + "|"
                tag_str += tag + " " * (max_len - len(tag)) + "|"
            strs += src_str + "\n"
            strs += tag_str + "\n"
        strs += "\n"
    return strs

def main(args):
    # ── load model ───────────────────────────
    print(f"Loading model from: {args.restore_dir}")
    model     = GECToR.from_pretrained(args.restore_dir).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.restore_dir)

    if torch.cuda.is_available():
        model.cuda()
        print("Running on GPU")
    else:
        print("Running on CPU")

    print(f"CUDA_VISIBLE_DEVICES : {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Visible devices      : {torch.cuda.device_count()}")
    print(f"Current device       : {torch.cuda.current_device()}")
    print(f"Model device         : {next(model.parameters()).device}")

    # ── load input ───────────────────────────
    srcs = open(args.input, encoding="utf-8").read().rstrip().split("\n")
    print(f"Loaded {len(srcs)} sentences")

    # ── predict ──────────────────────────────
    print("Running detection...")
    label_sequences, iteration_log = predict_detect_only(
        model=model,
        tokenizer=tokenizer,
        srcs=srcs,
        keep_confidence=args.keep_confidence,
        min_error_prob=args.min_error_prob,
        batch_size=args.batch_size,
    )

    # ── save labels ──────────────────────────
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for labels in label_sequences:
            f.write(" ".join(labels) + "\n")
    print(f"Labels saved → {args.out}")

    # ── save visualization ───────────────────
    if args.visualize:
        with open(args.visualize, "w", encoding="utf-8") as f:
            f.write(visualizer(iteration_log))
        print(f"Visualization saved → {args.visualize}")

    # ── summary ──────────────────────────────
    n_tokens      = sum(len(l) for l in label_sequences)
    n_errors      = sum(1 for l in label_sequences for t in l if t in ERROR_LABELS)
    n_error_sents = sum(1 for l in label_sequences if any(t in ERROR_LABELS for t in l))

    print("\n" + "=" * 50)
    print(f"  Sentences         : {len(srcs)}")
    print(f"  Sentences w/ error: {n_error_sents}")
    print(f"  Total tokens      : {n_tokens}")
    print(f"  Error tokens      : {n_errors}  ({100*n_errors/max(n_tokens,1):.1f}%)")
    print("=" * 50)

def get_parser():
    parser = argparse.ArgumentParser(
        description="Detection-only prediction for German GEC"
    )
    parser.add_argument("--input",       required=True)
    parser.add_argument("--restore_dir", required=True)
    parser.add_argument("--out",         default="results/gector/predictions.txt")
    parser.add_argument("--visualize",   default=None)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--keep_confidence", type=float, default=0.0)
    parser.add_argument("--min_error_prob",  type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()
    main(args)