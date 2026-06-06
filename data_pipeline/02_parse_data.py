"""
02_parse_data.py
----------------
Annotates German text data with Stanza (POS, lemma, depparse).

Two modes:
    wiki    — input is a flat JSON list of sentences (FineWeb/Wikipedia)
              no alignment needed, annotate directly
    asr     — input is a CSV with reference + hypothesis columns
              splits into sentences, aligns with DTW + sentence-transformers,
              then annotates each aligned pair

Output format per entry:

    wiki mode:
    {
        "sent_id": "0000001",
        "text":    "...",
        "tokens":  [...]
    }

    asr mode:
    {
        "sent_id":      "0000001",
        "recording_id": "train_eurospeech_00000",
        "sent_index":   0,
        "reference":  {"text": "...", "tokens": [...]},
        "hypothesis": {"text": "...", "tokens": [...]}
    }

Usage:
    python 02_parse_data.py --mode wiki \
        --input  ./data/wiki_fineweb_sent.json \
        --output ./data/wiki_fineweb_annotated.json

    python 02_parse_data.py --mode asr \
        --input  ./data/raw/manifest_train_eurospeech.csv \
        --output ./data/hypotheses_eurospeech_annotated.json
"""

import csv
import json
import os
import argparse
import subprocess
from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from utils.stanza_utils import load_stanza_full, load_stanza_sentence_tokenizer, annotate_sentence, sentence_split
from utils.constants import get_best_gpu

os.environ["CUDA_VISIBLE_DEVICES"] = str(get_best_gpu())


# ─────────────────────────────────────────────
# DTW ALIGNMENT (ASR mode only)
# ─────────────────────────────────────────────

def cosine_similarity_matrix(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    norm_a = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-9)
    norm_b = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-9)
    return norm_a @ norm_b.T


def dtw_align(sim_matrix: np.ndarray) -> list[tuple[list[int], list[int]]]:
    n_ref, n_hyp = sim_matrix.shape
    cost = 1.0 - sim_matrix
    acc  = np.full((n_ref, n_hyp), np.inf)
    acc[0, 0] = cost[0, 0]

    for i in range(1, n_ref):
        acc[i, 0] = acc[i-1, 0] + cost[i, 0]
    for j in range(1, n_hyp):
        acc[0, j] = acc[0, j-1] + cost[0, j]
    for i in range(1, n_ref):
        for j in range(1, n_hyp):
            acc[i, j] = cost[i, j] + min(acc[i-1, j-1], acc[i-1, j], acc[i, j-1])

    path = []
    i, j = n_ref - 1, n_hyp - 1
    while i > 0 or j > 0:
        path.append((i, j))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            move = np.argmin([acc[i-1, j-1], acc[i-1, j], acc[i, j-1]])
            if move == 0:
                i -= 1; j -= 1
            elif move == 1:
                i -= 1
            else:
                j -= 1
    path.append((0, 0))
    path.reverse()

    pairs = []
    k = 0
    while k < len(path):
        ref_i, hyp_j = path[k]
        ref_group = {ref_i}
        hyp_group = {hyp_j}
        while k + 1 < len(path):
            next_ref, next_hyp = path[k + 1]
            if next_ref == ref_i or next_hyp == hyp_j:
                ref_group.add(next_ref)
                hyp_group.add(next_hyp)
                ref_i = next_ref
                hyp_j = next_hyp
                k += 1
            else:
                break
        pairs.append((sorted(ref_group), sorted(hyp_group)))
        k += 1
    return pairs


def align_sentences(sbert, ref_sents: list[str], hyp_sents: list[str]) -> list[tuple[str, str]]:
    if not ref_sents or not hyp_sents:
        return [(" ".join(ref_sents), " ".join(hyp_sents))]
    if len(ref_sents) == 1 and len(hyp_sents) == 1:
        return [(ref_sents[0], hyp_sents[0])]

    ref_embs   = sbert.encode(ref_sents, convert_to_numpy=True, show_progress_bar=False)
    hyp_embs   = sbert.encode(hyp_sents, convert_to_numpy=True, show_progress_bar=False)
    sim_matrix = cosine_similarity_matrix(ref_embs, hyp_embs)
    grouped    = dtw_align(sim_matrix)

    return [
        (" ".join(ref_sents[i] for i in ref_idxs),
         " ".join(hyp_sents[j] for j in hyp_idxs))
        for ref_idxs, hyp_idxs in grouped
    ]


# ─────────────────────────────────────────────
# WIKI MODE
# ─────────────────────────────────────────────

def process_wiki(nlp_full, input_path: Path) -> list[dict]:
    with open(input_path, encoding="utf-8") as f:
        sentences = json.load(f)
    print(f"Loaded {len(sentences)} sentences")

    annotated = []
    for idx, sentence in enumerate(sentences):
        sent_id = f"{idx + 1:07d}"
        try:
            tokens = annotate_sentence(nlp_full, sentence)
            annotated.append({
                "sent_id": sent_id,
                "text":    sentence,
                "tokens":  tokens,
            })
        except Exception as e:
            print(f"  [error] sentence {sent_id}: {e}")
            continue

        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{len(sentences)} done")

    return annotated


# ─────────────────────────────────────────────
# ASR MODE
# ─────────────────────────────────────────────

def process_asr(nlp_full, nlp_split, sbert, input_path: Path) -> list[dict]:
    with open(input_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} rows from CSV")

    annotated       = []
    global_sent_idx = 0
    error_count     = 0

    for rec_idx, row in enumerate(rows):
        recording_id = row["id"]
        reference    = row["reference"].strip()
        hypothesis   = row["hypothesis"].strip()

        try:
            ref_sents = sentence_split(nlp_split, reference)
            hyp_sents = sentence_split(nlp_split, hypothesis)
            pairs     = align_sentences(sbert, ref_sents, hyp_sents)
        except Exception as e:
            print(f"  [error] alignment failed for {recording_id}: {e}")
            pairs = [(reference, hypothesis)]
            error_count += 1

        for sent_i, (ref_text, hyp_text) in enumerate(pairs):
            try:
                ref_tokens = annotate_sentence(nlp_full, ref_text)
                hyp_tokens = annotate_sentence(nlp_full, hyp_text)
            except Exception as e:
                print(f"  [error] annotation failed {recording_id} sent {sent_i}: {e}")
                ref_tokens = []
                hyp_tokens = []

            global_sent_idx += 1
            annotated.append({
                "sent_id":      f"{global_sent_idx:07d}",
                "recording_id": recording_id,
                "sent_index":   sent_i,
                "reference":  {"text": ref_text,  "tokens": ref_tokens},
                "hypothesis": {"text": hyp_text, "tokens": hyp_tokens},
            })

        if (rec_idx + 1) % 10 == 0:
            print(f"  {rec_idx + 1}/{len(rows)} recordings "
                  f"({global_sent_idx} sentences so far)...")

    print(f"  Alignment errors: {error_count}")
    return annotated


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",   required=True, choices=["wiki", "asr"],
                        help="wiki: flat JSON sentences | asr: CSV with reference+hypothesis")
    parser.add_argument("--input",  required=True, help="Input file path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Mode   : {args.mode}")
    print(f"Input  : {input_path}")
    print(f"Output : {output_path}\n")

    print("Loading Stanza full pipeline...")
    nlp_full = load_stanza_full(use_gpu=True)

    if args.mode == "wiki":
        annotated = process_wiki(nlp_full, input_path)

    else:  # asr
        print("Loading Stanza sentence tokenizer...")
        nlp_split = load_stanza_sentence_tokenizer(use_gpu=False)

        print("Loading sentence-transformers model...")
        from sentence_transformers import SentenceTransformer
        sbert = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

        annotated = process_asr(nlp_full, nlp_split, sbert, input_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotated, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Saved {len(annotated)} entries to {output_path}")


if __name__ == "__main__":
    main()