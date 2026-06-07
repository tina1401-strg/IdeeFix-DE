import json
import argparse
from collections import defaultdict
from pathlib import Path


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

KEEP_LABEL   = "$KEEP"
ERROR_LABELS = {"$WRONG_CAP", "$WRONG_DECL", "$WRONG_COMP"}
DELIM        = "SEPL|||SEPR"


# ─────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────

def load_gold_json(path: Path) -> list[dict]:
    """Load gold JSON — returns list of items with labeled field."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_labeled_string(labeled_str: str) -> list[str]:
    """Extract label sequence from labeled string."""
    labels = []
    for pair in labeled_str.strip().split(" "):
        if DELIM in pair:
            _, label = pair.rsplit(DELIM, 1)
            labels.append(label)
    return labels


def load_pred_labels(path: Path) -> list[list[str]]:
    """Load predicted label sequences from txt file."""
    lines  = path.read_text(encoding="utf-8").strip().splitlines()
    result = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # handle both plain and delimited format
        if DELIM in line:
            result.append(parse_labeled_string(line))
        else:
            result.append(line.split())
    return result


# ─────────────────────────────────────────────
# LABELED STRING BUILDERS
# ─────────────────────────────────────────────

def build_gold_labeled(labeled_str: str) -> str:
    """
    Convert labeled string to compact format without DELIM.
    'Land$SEPL|||SEPR$WRONG_COMP wirtschaft$SEPL|||SEPR$KEEP'
    → 'Land$WRONG_COMP wirtschaft$KEEP'
    """
    parts = []
    for pair in labeled_str.strip().split(" "):
        if DELIM in pair:
            token, label = pair.rsplit(DELIM, 1)
            parts.append(f"{token}{label}")
    return " ".join(parts)


def build_pred_labeled(plain_tok: str, pred_labels: list[str]) -> str:
    """
    Build compact labeled string from plain_tok tokens + predicted labels.
    plain_tok:   'Land wirtschaft ist'
    pred_labels: ['$KEEP', '$KEEP', '$KEEP']
    → 'Land$KEEP wirtschaft$KEEP ist$KEEP'
    """
    tokens = plain_tok.strip().split()
    return " ".join(
        f"{tok}{lab}"
        for tok, lab in zip(tokens, pred_labels)
    )


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1  = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    f05 = (
        1.25 * precision * recall / (0.25 * precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return {
        "TP":        tp,
        "FP":        fp,
        "FN":        fn,
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "F1":        round(f1,        4),
        "F0.5":      round(f05,       4),
    }


# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────

def evaluate(
    gold_seqs: list[list[str]],
    pred_seqs: list[list[str]],
) -> dict:
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    n_sentences          = 0
    n_correct_sentences  = 0
    n_error_sentences    = 0
    n_detected_sentences = 0
    n_both_sentences     = 0
    length_mismatches    = 0

    for gold, pred in zip(gold_seqs, pred_seqs):
        n_sentences += 1

        if len(gold) != len(pred):
            length_mismatches += 1
            min_len = min(len(gold), len(pred))
            gold    = gold[:min_len]
            pred    = pred[:min_len]

        gold_has_error = any(g in ERROR_LABELS for g in gold)
        pred_has_error = any(p in ERROR_LABELS for p in pred)

        if gold_has_error:                            n_error_sentences    += 1
        if pred_has_error:                            n_detected_sentences += 1
        if gold_has_error and pred_has_error:         n_both_sentences     += 1
        if not gold_has_error and not pred_has_error: n_correct_sentences  += 1

        for g, p in zip(gold, pred):
            g_is_error = g in ERROR_LABELS
            p_is_error = p in ERROR_LABELS

            if g_is_error and p == g:
                tp[g]         += 1
                tp["overall"] += 1
            elif p_is_error and p != g:
                fp[p]         += 1
                fp["overall"] += 1
                if g_is_error:
                    fn[g]         += 1
                    fn["overall"] += 1
            elif g_is_error and not p_is_error:
                fn[g]         += 1
                fn["overall"] += 1

    metrics = {}
    for label in [*sorted(ERROR_LABELS), "overall"]:
        metrics[label] = prf(
            tp=tp.get(label, 0),
            fp=fp.get(label, 0),
            fn=fn.get(label, 0),
        )

    error_metrics = [metrics[l] for l in sorted(ERROR_LABELS)]
    macro = {
        "precision": round(sum(m["precision"] for m in error_metrics) / len(error_metrics), 4),
        "recall":    round(sum(m["recall"]    for m in error_metrics) / len(error_metrics), 4),
        "F1":        round(sum(m["F1"]        for m in error_metrics) / len(error_metrics), 4),
        "F0.5":      round(sum(m["F0.5"]      for m in error_metrics) / len(error_metrics), 4),
    }

    return {
        "metrics": metrics,
        "macro":   macro,
        "sentences": {
            "total":             n_sentences,
            "correct":           n_correct_sentences,
            "gold_error":        n_error_sentences,
            "pred_error":        n_detected_sentences,
            "both_error":        n_both_sentences,
            "length_mismatches": length_mismatches,
        },
        "raw": {"tp": dict(tp), "fp": dict(fp), "fn": dict(fn)},
    }


# ─────────────────────────────────────────────
# PRINT
# ─────────────────────────────────────────────

def print_results(results: dict, gold_path: str, pred_path: str):
    s = results["sentences"]
    print(f"\nGold : {gold_path}")
    print(f"Pred : {pred_path}")

    print("\n" + "=" * 65)
    print("SENTENCE-LEVEL STATS")
    print("=" * 65)
    print(f"  Total sentences       : {s['total']}")
    print(f"  Gold error sentences  : {s['gold_error']}")
    print(f"  Pred error sentences  : {s['pred_error']}")
    print(f"  Both error            : {s['both_error']}")
    print(f"  Both correct          : {s['correct']}")
    if s["length_mismatches"] > 0:
        print(f"  [warn] Length mismatches: {s['length_mismatches']}")

    print("\n" + "=" * 65)
    print("TOKEN-LEVEL METRICS (per error type)")
    print("=" * 65)
    print(f"  {'Label':<15}  {'TP':>6}  {'FP':>6}  {'FN':>6}  "
          f"{'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'F0.5':>8}")
    print("  " + "-" * 70)

    for label in sorted(ERROR_LABELS):
        m = results["metrics"][label]
        print(f"  {label:<15}  "
              f"TP={m['TP']:>4}  FP={m['FP']:>4}  FN={m['FN']:>4}  "
              f"P={m['precision']:.4f}  "
              f"R={m['recall']:.4f}  "
              f"F1={m['F1']:.4f}  "
              f"F0.5={m['F0.5']:.4f}")

    print("  " + "-" * 70)
    m = results["metrics"]["overall"]
    print(f"  {'OVERALL':<15}  "
          f"TP={m['TP']:>4}  FP={m['FP']:>4}  FN={m['FN']:>4}  "
          f"P={m['precision']:.4f}  "
          f"R={m['recall']:.4f}  "
          f"F1={m['F1']:.4f}  "
          f"F0.5={m['F0.5']:.4f}")

    mac = results["macro"]
    print(f"\n  {'MACRO AVG':<15}  "
          f"{'':>18}  "
          f"P={mac['precision']:.4f}  "
          f"R={mac['recall']:.4f}  "
          f"F1={mac['F1']:.4f}  "
          f"F0.5={mac['F0.5']:.4f}")
    print("=" * 65)


# ─────────────────────────────────────────────
# WRITE OUTPUT JSON
# ─────────────────────────────────────────────

"""
evaluate_gector.py
------------------
Evaluates GECToR detection predictions against gold labels.

Input:
    --gold_json  JSON file with sent_id, cor, err, plain_tok, labeled
    --pred       predictions.txt (one label sequence per line)

Output (printed):
    Per-label and overall precision/recall/F1/F0.5
    Sentence-level statistics

Output (optional --output):
    JSON with sent_id, cor, err, gold_labeled, pred_labeled
    (only sentences with errors or mismatches)

Usage:
    python evaluate_gector.py \
        --gold_json  ../data/splits/gector/test_gector.json \
        --pred       ./results/predictions.txt \
        --output     ./results/error_analysis.json
"""

import json
import argparse
from collections import defaultdict
from pathlib import Path


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

KEEP_LABEL   = "$KEEP"
ERROR_LABELS = {"$WRONG_CAP", "$WRONG_DECL", "$WRONG_COMP"}
DELIM        = "SEPL|||SEPR"


# ─────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────

def load_gold_json(path: Path) -> list[dict]:
    """Load gold JSON — returns list of items with labeled field."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_labeled_string(labeled_str: str) -> list[str]:
    """Extract label sequence from labeled string."""
    labels = []
    for pair in labeled_str.strip().split(" "):
        if DELIM in pair:
            _, label = pair.rsplit(DELIM, 1)
            labels.append(label)
    return labels


def load_pred_labels(path: Path) -> list[list[str]]:
    """Load predicted label sequences from txt file."""
    lines  = path.read_text(encoding="utf-8").strip().splitlines()
    result = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # handle both plain and delimited format
        if DELIM in line:
            result.append(parse_labeled_string(line))
        else:
            result.append(line.split())
    return result


# ─────────────────────────────────────────────
# LABELED STRING BUILDERS
# ─────────────────────────────────────────────

def build_gold_labeled(labeled_str: str) -> str:
    """
    Convert labeled string to compact format without DELIM.
    'Land$SEPL|||SEPR$WRONG_COMP wirtschaft$SEPL|||SEPR$KEEP'
    → 'Land$WRONG_COMP wirtschaft$KEEP'
    """
    parts = []
    for pair in labeled_str.strip().split(" "):
        if DELIM in pair:
            token, label = pair.rsplit(DELIM, 1)
            parts.append(f"{token}{label}")
    return " ".join(parts)


def build_pred_labeled(plain_tok: str, pred_labels: list[str]) -> str:
    """
    Build compact labeled string from plain_tok tokens + predicted labels.
    plain_tok:   'Land wirtschaft ist'
    pred_labels: ['$KEEP', '$KEEP', '$KEEP']
    → 'Land$KEEP wirtschaft$KEEP ist$KEEP'
    """
    tokens = plain_tok.strip().split()
    return " ".join(
        f"{tok}{lab}"
        for tok, lab in zip(tokens, pred_labels)
    )


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1  = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    f05 = (
        1.25 * precision * recall / (0.25 * precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return {
        "TP":        tp,
        "FP":        fp,
        "FN":        fn,
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "F1":        round(f1,        4),
        "F0.5":      round(f05,       4),
    }


# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────

def evaluate(
    gold_seqs: list[list[str]],
    pred_seqs: list[list[str]],
) -> dict:
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    n_sentences          = 0
    n_correct_sentences  = 0
    n_error_sentences    = 0
    n_detected_sentences = 0
    n_both_sentences     = 0
    length_mismatches    = 0

    for gold, pred in zip(gold_seqs, pred_seqs):
        n_sentences += 1

        if len(gold) != len(pred):
            length_mismatches += 1
            min_len = min(len(gold), len(pred))
            gold    = gold[:min_len]
            pred    = pred[:min_len]

        gold_has_error = any(g in ERROR_LABELS for g in gold)
        pred_has_error = any(p in ERROR_LABELS for p in pred)

        if gold_has_error:                            n_error_sentences    += 1
        if pred_has_error:                            n_detected_sentences += 1
        if gold_has_error and pred_has_error:         n_both_sentences     += 1
        if not gold_has_error and not pred_has_error: n_correct_sentences  += 1

        for g, p in zip(gold, pred):
            g_is_error = g in ERROR_LABELS
            p_is_error = p in ERROR_LABELS

            if g_is_error and p == g:
                tp[g]         += 1
                tp["overall"] += 1
            elif p_is_error and p != g:
                fp[p]         += 1
                fp["overall"] += 1
                if g_is_error:
                    fn[g]         += 1
                    fn["overall"] += 1
            elif g_is_error and not p_is_error:
                fn[g]         += 1
                fn["overall"] += 1

    metrics = {}
    for label in [*sorted(ERROR_LABELS), "overall"]:
        metrics[label] = prf(
            tp=tp.get(label, 0),
            fp=fp.get(label, 0),
            fn=fn.get(label, 0),
        )

    error_metrics = [metrics[l] for l in sorted(ERROR_LABELS)]
    macro = {
        "precision": round(sum(m["precision"] for m in error_metrics) / len(error_metrics), 4),
        "recall":    round(sum(m["recall"]    for m in error_metrics) / len(error_metrics), 4),
        "F1":        round(sum(m["F1"]        for m in error_metrics) / len(error_metrics), 4),
        "F0.5":      round(sum(m["F0.5"]      for m in error_metrics) / len(error_metrics), 4),
    }

    return {
        "metrics": metrics,
        "macro":   macro,
        "sentences": {
            "total":             n_sentences,
            "correct":           n_correct_sentences,
            "gold_error":        n_error_sentences,
            "pred_error":        n_detected_sentences,
            "both_error":        n_both_sentences,
            "length_mismatches": length_mismatches,
        },
        "raw": {"tp": dict(tp), "fp": dict(fp), "fn": dict(fn)},
    }


# ─────────────────────────────────────────────
# PRINT
# ─────────────────────────────────────────────

def print_results(results: dict, gold_path: str, pred_path: str):
    s = results["sentences"]
    print(f"\nGold : {gold_path}")
    print(f"Pred : {pred_path}")

    print("\n" + "=" * 65)
    print("SENTENCE-LEVEL STATS")
    print("=" * 65)
    print(f"  Total sentences       : {s['total']}")
    print(f"  Gold error sentences  : {s['gold_error']}")
    print(f"  Pred error sentences  : {s['pred_error']}")
    print(f"  Both error            : {s['both_error']}")
    print(f"  Both correct          : {s['correct']}")
    if s["length_mismatches"] > 0:
        print(f"  [warn] Length mismatches: {s['length_mismatches']}")

    print("\n" + "=" * 65)
    print("TOKEN-LEVEL METRICS (per error type)")
    print("=" * 65)
    print(f"  {'Label':<15}  {'TP':>6}  {'FP':>6}  {'FN':>6}  "
          f"{'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'F0.5':>8}")
    print("  " + "-" * 70)

    for label in sorted(ERROR_LABELS):
        m = results["metrics"][label]
        print(f"  {label:<15}  "
              f"TP={m['TP']:>4}  FP={m['FP']:>4}  FN={m['FN']:>4}  "
              f"P={m['precision']:.4f}  "
              f"R={m['recall']:.4f}  "
              f"F1={m['F1']:.4f}  "
              f"F0.5={m['F0.5']:.4f}")

    print("  " + "-" * 70)
    m = results["metrics"]["overall"]
    print(f"  {'OVERALL':<15}  "
          f"TP={m['TP']:>4}  FP={m['FP']:>4}  FN={m['FN']:>4}  "
          f"P={m['precision']:.4f}  "
          f"R={m['recall']:.4f}  "
          f"F1={m['F1']:.4f}  "
          f"F0.5={m['F0.5']:.4f}")

    mac = results["macro"]
    print(f"\n  {'MACRO AVG':<15}  "
          f"{'':>18}  "
          f"P={mac['precision']:.4f}  "
          f"R={mac['recall']:.4f}  "
          f"F1={mac['F1']:.4f}  "
          f"F0.5={mac['F0.5']:.4f}")
    print("=" * 65)


# ─────────────────────────────────────────────
# WRITE OUTPUT JSON
# ─────────────────────────────────────────────

def write_output_json(
    gold_data:   list[dict],
    pred_seqs:   list[list[str]],
    output_path: Path,
):
    entries = []

    for item, pred_labels in zip(gold_data, pred_seqs):
        gold_labels = parse_labeled_string(item["labeled"])

        is_mismatch = len(gold_labels) != len(pred_labels)
        min_len     = min(len(gold_labels), len(pred_labels))
        is_correct  = (
            not is_mismatch and
            all(g == p for g, p in zip(gold_labels, pred_labels))
        )

        # ← removed the filter, write ALL sentences
        entries.append({
            "sent_id":      item["sent_id"],
            "cor":          item["cor"],
            "err":          item["err"],
            "gold_labeled": build_gold_labeled(item["labeled"]),
            "pred_labeled": build_pred_labeled(item["plain_tok"], pred_labels),
            "correct":      is_correct,
            "mismatch":     is_mismatch,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"\nOutput JSON → {output_path} ({len(entries)} sentences)")
# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GECToR detection predictions"
    )
    parser.add_argument("--input", required=True,
                        help="Gold JSON file (test_gector.json)")
    parser.add_argument("--pred",      required=True,
                        help="Predicted label file (predictions.txt)")
    parser.add_argument("--output",    default=None,
                        help="Output JSON with error sentences (optional)")
    args = parser.parse_args()

    gold_data = load_gold_json(Path(args.input))
    pred_seqs = load_pred_labels(Path(args.pred))
    gold_seqs = [parse_labeled_string(item["labeled"]) for item in gold_data]

    if len(gold_seqs) != len(pred_seqs):
        print(f"[warn] Sentence count mismatch: "
              f"gold={len(gold_seqs)} pred={len(pred_seqs)}")
        min_len   = min(len(gold_seqs), len(pred_seqs))
        gold_data = gold_data[:min_len]
        gold_seqs = gold_seqs[:min_len]
        pred_seqs = pred_seqs[:min_len]

    results = evaluate(gold_seqs, pred_seqs)
    print_results(results, args.input, args.pred)

    if args.output:
        write_output_json(
            gold_data   = gold_data,
            pred_seqs   = pred_seqs,
            output_path = Path(args.output),
        )


if __name__ == "__main__":
    main()
