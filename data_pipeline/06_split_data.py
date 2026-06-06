"""
05_split_data.py
----------------
Takes the full error-injected/detected JSON files (wiki + whisper),
creates stratified train/dev/test splits for GECToR and LLM evaluation,
and writes JSON, GECToR txt, plain txt, and gold txt formats.

Split strategy:
    test_llm    : 10,000 sentences (Whisper only, held out first)
    eval_llm    :    200 sentences (Whisper only, held out)
    remaining   : split stratified by dominant error type
                  dev=10% / test_gector=10% / train=80%

Outputs (all under --output_dir):
    gector/train.json         gector/train.txt
    gector/dev.json           gector/dev.txt
    gector/test_gector.json   gector/test_gector.txt
                              gector/test_plain.txt
                              gector/test_gold.txt
    llm/eval_llm.json
    llm/test_llm.json

"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from utils.constants import SEP, RANDOM_SEED

TEST_LLM_SIZE  = 1000
EVAL_LLM_SIZE  = 200
DEV_RATIO      = 0.1
TEST_GEC_RATIO = 0.1

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_json(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_dominant_error(item: dict) -> str:
    labeled = item.get("labeled", "")
    for error in ("$WRONG_CAP", "$WRONG_DECL", "$WRONG_COMP"):
        if error in labeled:
            return error
    return "$KEEP"


def stratified_split(items: list[dict], dev_r: float, test_r: float) -> tuple:
    groups = defaultdict(list)
    for item in items:
        groups[get_dominant_error(item)].append(item)

    train, dev, test = [], [], []
    for error_type, group in groups.items():
        random.shuffle(group)
        n      = len(group)
        n_dev  = max(1, int(n * dev_r))
        n_test = max(1, int(n * test_r))
        dev.extend(group[:n_dev])
        test.extend(group[n_dev:n_dev + n_test])
        train.extend(group[n_dev + n_test:])
        print(f"    {error_type:15s}: total={n:5d}  "
              f"train={n - n_dev - n_test:5d}  "
              f"dev={n_dev:4d}  test={n_test:4d}")

    return train, dev, test


def print_label_dist(items: list[dict], name: str):
    counts    = defaultdict(int)
    err_sents = sum(1 for i in items if i["cor"] != i["err"])
    for item in items:
        for token_label in item["labeled"].split(" "):
            if f"{SEP}$" in token_label:
                label = "$" + token_label.split(f"{SEP}$")[-1]
                counts[label] += 1
    print(f"\n  {name} ({len(items)} sentences, {err_sents} with errors):")
    for label in ("$KEEP", "$WRONG_CAP", "$WRONG_COMP", "$WRONG_DECL"):
        print(f"    {label:20s}: {counts[label]:7d}")


def write_gector(items: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            line = item["labeled"]
            if line.strip():
                f.write(line + "\n")
    print(f"    GECToR → {path.name:<40} ({len(items):6d} sentences)")

def labeled_to_plain_tok(labeled_str: str) -> str:
    """
    Extract just the tokens from labeled string, space-joined.
    'Land$SEPL|||SEPR$WRONG_COMP wirtschaft$SEPL|||SEPR$KEEP'
    → 'Land wirtschaft'
    """
    tokens = []
    for chunk in labeled_str.strip().split(" "):
        if SEP in chunk:
            token, _ = chunk.split(SEP, 1)
            tokens.append(token)
    return " ".join(tokens)


def write_json(items: list[dict], path: Path):
    """Write JSON — adds plain_tok field to each item."""
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = []
    for item in items:
        enriched.append({
            "sent_id":   item["sent_id"],
            "cor":       item["cor"],
            "err":       item["err"],
            "plain_tok": labeled_to_plain_tok(item["labeled"]),
            "labeled":   item["labeled"],
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    print(f"    JSON   → {path.name:<40} ({len(items):6d} sentences)")


def write_plain_and_gold(items: list[dict], plain_path: Path,
                         gold_path: Path, plain_tok_path: Path):
    """
    plain.txt     ← err sentences (raw text, input for plain LLM)
    plain_tok.txt ← stanza-tokenized tokens space-joined (input for GECToR)
    gold.txt      ← label sequences (for evaluation)
    """
    plain_lines     = []
    plain_tok_lines = []
    gold_lines      = []

    for item in items:
        plain_lines.append(item["err"].strip())
        plain_tok_lines.append(labeled_to_plain_tok(item["labeled"]))
        labels = []
        for chunk in item["labeled"].strip().split(" "):
            if SEP in chunk:
                _, label = chunk.rsplit(SEP, 1)
                labels.append(label)
        gold_lines.append(" ".join(labels))

    plain_path.parent.mkdir(parents=True, exist_ok=True)
    plain_path.write_text(    "\n".join(plain_lines),     encoding="utf-8")
    plain_tok_path.write_text("\n".join(plain_tok_lines), encoding="utf-8")
    gold_path.write_text(     "\n".join(gold_lines),      encoding="utf-8")

    print(f"    Plain     → {plain_path.name:<40} ({len(plain_lines):6d} sentences)")
    print(f"    Plain tok → {plain_tok_path.name:<40} ({len(plain_tok_lines):6d} sentences)")
    print(f"    Gold      → {gold_path.name:<40} ({len(gold_lines):6d} sentences)")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper_path", required=True,
                        help="ASR detected+injected JSON")
    parser.add_argument("--wiki_path",    required=True,
                        help="Wiki/FineWeb injected JSON")
    parser.add_argument("--output_dir",   required=True,
                        help="Root output directory (e.g. ./data/splits)")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)

    output_dir  = Path(args.output_dir)
    gector_dir  = output_dir / "encoder_detection"
    llm_dir     = output_dir / "llm_correction"

    # ── load ─────────────────────────────────
    print("Loading data...")
    whisper = load_json(Path(args.whisper_path))
    wiki    = load_json(Path(args.wiki_path))
    print(f"  Whisper : {len(whisper):6d} sentences")
    print(f"  Wiki    : {len(wiki):6d} sentences")

    # ── deduplicate by err sentence ───────────
    print("\nDeduplicating by err sentence...")
    seen_err      = set()
    whisper_dedup = []
    for item in whisper:
        if item["err"] not in seen_err:
            seen_err.add(item["err"])
            whisper_dedup.append(item)
    wiki_dedup = []
    for item in wiki:
        if item["err"] not in seen_err:
            seen_err.add(item["err"])
            wiki_dedup.append(item)
    print(f"  Whisper : {len(whisper):6d} → {len(whisper_dedup):6d} "
          f"({len(whisper) - len(whisper_dedup):4d} removed)")
    print(f"  Wiki    : {len(wiki):6d} → {len(wiki_dedup):6d} "
          f"({len(wiki) - len(wiki_dedup):4d} removed)")
    whisper = whisper_dedup
    wiki    = wiki_dedup

    # ── hold out test_llm (Whisper only) ─────
    print(f"\nHolding out test_llm ({TEST_LLM_SIZE} Whisper sentences)...")
    random.shuffle(whisper)
    test_llm          = whisper[:TEST_LLM_SIZE]
    whisper_remaining = whisper[TEST_LLM_SIZE:]
    print(f"  test_llm          : {len(test_llm)}")
    print(f"  Whisper remaining : {len(whisper_remaining)}")

    # ── hold out eval_llm (Whisper only) ─────
    print(f"\nHolding out eval_llm ({EVAL_LLM_SIZE} sentences)...")
    random.shuffle(whisper_remaining)
    eval_llm          = whisper_remaining[:EVAL_LLM_SIZE]
    whisper_remaining = whisper_remaining[EVAL_LLM_SIZE:]
    print(f"  eval_llm          : {len(eval_llm)}")
    print(f"  Whisper remaining : {len(whisper_remaining)}")

    # ── stratified split of remaining pool ───
    pool = whisper_remaining + wiki
    print(f"\nStratified split of {len(pool)} sentences "
          f"(dev={DEV_RATIO:.0%} / test={TEST_GEC_RATIO:.0%} / train=rest)...")
    train, dev, test_gector = stratified_split(pool, DEV_RATIO, TEST_GEC_RATIO)

    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test_gector)

    # ── label distributions ───────────────────
    print_label_dist(train,       "train")
    print_label_dist(dev,         "dev")
    print_label_dist(test_gector, "test_gector")
    print_label_dist(eval_llm,    "eval_llm")
    print_label_dist(test_llm,    "test_llm")

# ── write GECToR splits ───────────────────
    print("\nWriting GECToR splits...")
    write_json(train,         gector_dir / "train.json")
    write_json(dev,           gector_dir / "dev.json")
    write_json(test_gector,   gector_dir / "test_gector.json")
    write_gector(train,       gector_dir / "train.txt")
    write_gector(dev,         gector_dir / "dev.txt")
    write_gector(test_gector, gector_dir / "test_gector.txt")
    write_plain_and_gold(
        test_gector,
        plain_path     = gector_dir / "test_plain.txt",
        gold_path      = gector_dir / "test_gold.txt",
        plain_tok_path = gector_dir / "test_plain_tok.txt",
    )

    # ── write LLM splits ──────────────────────
    print("\nWriting LLM splits...")
    write_json(eval_llm, llm_dir / "eval_llm.json")
    write_json(test_llm, llm_dir / "test_llm.json")
    write_gector(test_llm, llm_dir / "test_llm.txt")
    write_plain_and_gold(
        test_llm,
        plain_path     = llm_dir / "test_plain.txt",
        gold_path      = llm_dir / "test_gold.txt",
        plain_tok_path = llm_dir / "test_plain_tok.txt",
    )

    # ── summary ───────────────────────────────
    total = len(train) + len(dev) + len(test_gector) + len(eval_llm) + len(test_llm)
    print("\n" + "=" * 55)
    print("DONE")
    print(f"  {'train':<15}: {len(train):6d}  → gector/")
    print(f"  {'dev':<15}: {len(dev):6d}  → gector/")
    print(f"  {'test_gector':<15}: {len(test_gector):6d}  → gector/")
    print(f"  {'eval_llm':<15}: {len(eval_llm):6d}  → llm/")
    print(f"  {'test_llm':<15}: {len(test_llm):6d}  → llm/")
    print(f"  {'total':<15}: {total:6d}")
    print(f"  Output dir : {output_dir}")
    print("=" * 55)


if __name__ == "__main__":
    main()