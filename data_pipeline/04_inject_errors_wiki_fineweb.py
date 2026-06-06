"""
04_inject_error_wiki_fineweb.py
--------------------------------
Injects three types of German ASR errors into clean annotated sentences
from FineWeb/Wikipedia.

Error types:
    $WRONG_CAP   — noun lowercased (text == verb lemma)
    $WRONG_DECL  — adjective given wrong declension ending
    $WRONG_COMP  — compound noun split into parts

Input:  annotated wiki/fineweb JSON (flat list of {sent_id, text, tokens})
Output: labeled JSON with err/cor/labeled fields

Usage:
    python 04_inject_error_wiki_fineweb.py \
        --input_paths ./data/anno/wiki_fineweb_annotated.json \
        --output      ./data/processed/wiki_fineweb_injected.json
"""

import json
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.constants import SEP, VERB_LIST_PATH, RANDOM_SEED
from utils.injection_utils import (
    build_position_map,
    collect_wrong_cap,
    collect_wrong_decl,
    collect_wrong_comp,
    apply_edits,
    build_labeled,
    print_summary,
)

VERB_LIST = Path(VERB_LIST_PATH)

TARGET_COUNTS = {
    "$WRONG_CAP":  8500,
    "$WRONG_DECL": 7000,
    "$WRONG_COMP": 20000,
}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      required=True,
                        help="Input JSON path")
    parser.add_argument("--output",      required=True,
                        help="Output JSON path")
    args = parser.parse_args()

    output      = Path(args.output)

    # ── load data ────────────────────────────
    data = []
    input = Path(args.input)
    if not input.exists():
        raise FileNotFoundError(f"Input not found: {input}")
    with open(input, encoding="utf-8") as f:
        data.extend(json.load(f))
    print(f"Loaded {len(data)} sentences")

    # ── load verb list ────────────────────────
    if not VERB_LIST.exists():
        raise FileNotFoundError(f"Verb list not found: {VERB_LIST}")
    verb_set = set(
        line.strip().lower()
        for line in VERB_LIST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    print(f"Loaded {len(verb_set)} verbs")

    # ── deduplicate ───────────────────────────
    seen_texts  = set()
    unique_data = []
    for sentence in data:
        if sentence["text"] not in seen_texts:
            seen_texts.add(sentence["text"])
            unique_data.append(sentence)
    print(f"After deduplication: {len(unique_data)} sentences "
          f"({len(data) - len(unique_data)} duplicates removed)")
    data = unique_data

    # ── collect candidates ────────────────────
    all_cap_candidates  = []
    all_decl_candidates = []
    all_comp_candidates = []

    for idx, sentence in enumerate(data):
        pos_map = build_position_map(sentence["text"], sentence["tokens"])

        for edit in collect_wrong_cap(sentence["tokens"], pos_map, verb_set):
            all_cap_candidates.append((idx, edit, pos_map))
        for edit in collect_wrong_decl(sentence["tokens"], pos_map):
            all_decl_candidates.append((idx, edit, pos_map))
        for edit in collect_wrong_comp(sentence["tokens"], pos_map):
            all_comp_candidates.append((idx, edit, pos_map))

    print(f"  CAP  candidates : {len(all_cap_candidates)}")
    print(f"  DECL candidates : {len(all_decl_candidates)}")
    print(f"  COMP candidates : {len(all_comp_candidates)}")

    # ── sample to target counts ───────────────
    random.seed(RANDOM_SEED)

    def sample_candidates(candidates, target):
        if len(candidates) <= target:
            print(f"  WARNING: only {len(candidates)} candidates, target was {target}")
            return candidates
        return random.sample(candidates, target)

    selected_cap  = sample_candidates(all_cap_candidates,  TARGET_COUNTS["$WRONG_CAP"])
    selected_decl = sample_candidates(all_decl_candidates, TARGET_COUNTS["$WRONG_DECL"])
    selected_comp = sample_candidates(all_comp_candidates, TARGET_COUNTS["$WRONG_COMP"])

    # ── build per-sentence edit lookup ────────
    sentence_edits = defaultdict(list)
    for idx, edit, _ in selected_cap + selected_decl + selected_comp:
        sentence_edits[idx].append(edit)

    # ── inject errors ─────────────────────────
    results = []

    for idx, sentence in enumerate(data):
        pos_map = build_position_map(sentence["text"], sentence["tokens"])
        edits   = sentence_edits.get(idx, [])
        labeled = build_labeled(sentence["tokens"], edits, pos_map)

        if not edits:
            results.append({
                "sent_id": sentence["sent_id"],
                "cor":     sentence["text"],
                "err":     sentence["text"],
                "labeled": labeled,
            })
            continue

        err_text = apply_edits(sentence["text"], edits)
        results.append({
            "sent_id": sentence["sent_id"],
            "cor":     sentence["text"],
            "err":     err_text,
            "labeled": labeled,
        })

    # ── save ──────────────────────────────────
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ── summary ───────────────────────────────
    label_counts = Counter()
    for entry in results:
        for token_label in entry["labeled"].split(" "):
            if f"{SEP}$" in token_label:
                label = "$" + token_label.split(f"{SEP}$")[1]
                label_counts[label] += 1

    print_summary(results, dict(label_counts), len(data), str(output))

    # ── sanity check ──────────────────────────
    print()
    for entry in results[:3]:
        print(f"SENT: {entry['sent_id']}")
        print(f"COR : {entry['cor']}")
        print(f"ERR : {entry['err']}")
        print(f"LAB : {entry['labeled']}")
        print()


if __name__ == "__main__":
    main()