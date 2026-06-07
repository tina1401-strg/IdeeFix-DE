import argparse
import json
import os
import re

from datasets import load_dataset
from tqdm import tqdm
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from utils.stanza_utils import load_stanza_sentence_tokenizer, sentence_split
from utils.constants import get_best_gpu

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
FINEWEB_QUOTA   = 25_000
WIKIPEDIA_QUOTA = 25_000
TOTAL           = FINEWEB_QUOTA + WIKIPEDIA_QUOTA

MIN_LENGTH  = 10    # words
MAX_LENGTH  = 100   # words
CHUNK_SIZE  = 5_000 # max chars per stanza call

# ─────────────────────────────────────────────
# GPU SELECTION
# ─────────────────────────────────────────────


os.environ["CUDA_VISIBLE_DEVICES"] = str(get_best_gpu())


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def clean_and_presplit(text: str) -> list[str]:
    """Split on newlines, drop very short lines."""
    chunks = re.split(r"\n+", text)
    return [c.strip() for c in chunks if len(c.strip()) > 20]


def extract_sentences(nlp, text: str, sentences: list[str], quota: int) -> list[str]:
    """Extract valid sentences from text, stop when quota reached."""
    for chunk in clean_and_presplit(text):
        if len(sentences) >= quota:
            break
        for subchunk in [chunk[i:i + CHUNK_SIZE] for i in range(0, len(chunk), CHUNK_SIZE)]:
            if len(sentences) >= quota:
                break
            try:
                for sent in sentence_split(nlp, subchunk):
                    sent_text = sent.text.strip()
                    if MIN_LENGTH <= len(sent_text.split()) <= MAX_LENGTH:
                        sentences.append(sent_text)
                        if len(sentences) >= quota:
                            break
            except Exception:
                continue
    return sentences


def collect_from_source(nlp, dataset, quota: int, desc: str) -> list[str]:
    """Stream a dataset and collect sentences up to quota."""
    sentences = []
    with tqdm(total=quota, desc=f"  {desc}", unit="sent") as pbar:
        prev = 0
        for doc in dataset:
            if len(sentences) >= quota:
                break
            text = doc.get("text", "")
            sentences = extract_sentences(nlp, text, sentences, quota)
            pbar.update(len(sentences) - prev)
            prev = len(sentences)
    return sentences


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main(args):
    print("Initializing Stanza (CPU, tokenize only)...")
    nlp = load_stanza_sentence_tokenizer(use_gpu=False)
    all_sentences = []

    # ── Source 1: FineWeb (25k) ──────────────
    print(f"\n[1/2] FineWeb — target: {FINEWEB_QUOTA:,} sentences")
    fineweb = load_dataset(
        "HuggingFaceFW/fineweb-2",
        "deu_Latn",
        split="train",
        streaming=True,
    )
    fineweb_sents = collect_from_source(nlp, fineweb, FINEWEB_QUOTA, "FineWeb")
    print(f"  Collected: {len(fineweb_sents):,}")
    all_sentences.extend(fineweb_sents)

    # ── Source 2: Wikipedia (25k) ────────────
    print(f"\n[2/2] Wikipedia — target: {WIKIPEDIA_QUOTA:,} sentences")
    wiki = load_dataset(
        "wikimedia/wikipedia",
        "20231101.de",
        split="train",
        streaming=True,
    )
    wiki_sents = collect_from_source(nlp, wiki, WIKIPEDIA_QUOTA, "Wikipedia")
    print(f"  Collected: {len(wiki_sents):,}")
    all_sentences.extend(wiki_sents)

    # ── Save ─────────────────────────────────
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_sentences, f, ensure_ascii=False, indent=2)

    # ── Summary ──────────────────────────────
    print("\n" + "=" * 50)
    print("DONE")
    print(f"  FineWeb sentences   : {len(fineweb_sents):,}")
    print(f"  Wikipedia sentences : {len(wiki_sents):,}")
    print(f"  Total               : {len(all_sentences):,}")
    print(f"  Output              : {args.output}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect 50k German sentences (25k FineWeb + 25k Wikipedia)"
    )
    parser.add_argument(
        "--output", "-o",
        default="german_sentences.json",
        help="Output JSON file (default: german_sentences.json)"
    )
    args = parser.parse_args()
    main(args)