"""
06_eda.py
---------
Exploratory Data Analysis for any GECToR split JSON.
Analyzes:
  1. Basic stats (sentence count, length distribution)
  2. Error rate (how many sentences actually have errors)
  3. Label distribution (how often each error type appears)
  4. Data quality issues (duplicates, empty sentences)
  5. Vocabulary / domain diversity
"""

import json
import argparse
from pathlib import Path
from collections import Counter
import sys

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from utils.constants import SEP


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def parse_labeled(labeled_str: str) -> list[tuple[str, str]]:
    pairs = []
    for chunk in labeled_str.split(" "):
        if SEP in chunk:
            token, label = chunk.split(SEP, 1)
            pairs.append((token, label))
    return pairs


def analyze(data: list[dict]) -> dict:
    lengths      = []
    label_counts = Counter()
    token_texts  = []

    for entry in data:
        pairs = parse_labeled(entry["labeled"])
        lengths.append(len(pairs))
        for token, label in pairs:
            label_counts[label] += 1
            token_texts.append(token.lower())

    vocab        = Counter(token_texts)
    total_tokens = sum(vocab.values())
    vocab_size   = len(vocab)
    hapax        = sum(1 for c in vocab.values() if c == 1)

    buckets = {"1-5": 0, "6-10": 0, "11-20": 0, "21-35": 0,
               "36-50": 0, "51-150": 0, "150+": 0}
    for n in lengths:
        if   n <= 5:   buckets["1-5"]    += 1
        elif n <= 10:  buckets["6-10"]   += 1
        elif n <= 20:  buckets["11-20"]  += 1
        elif n <= 35:  buckets["21-35"]  += 1
        elif n <= 50:  buckets["36-50"]  += 1
        elif n <= 150: buckets["51-150"] += 1
        else:          buckets["150+"]   += 1

    seen       = set()
    duplicates = 0
    empty      = 0
    for entry in data:
        if not entry["err"].strip():
            empty += 1
        if entry["err"] in seen:
            duplicates += 1
        seen.add(entry["err"])

    erroneous = sum(1 for e in data if e["err"].strip() != e["cor"].strip())

    return {
        "n":            len(data),
        "lengths":      lengths,
        "buckets":      buckets,
        "avg_len":      sum(lengths) / len(lengths),
        "min_len":      min(lengths),
        "max_len":      max(lengths),
        "erroneous":    erroneous,
        "correct":      len(data) - erroneous,
        "label_counts": label_counts,
        "total_tokens": total_tokens,
        "error_tokens": sum(c for l, c in label_counts.items() if l != "$KEEP"),
        "vocab":        vocab,
        "vocab_size":   vocab_size,
        "total_words":  total_tokens,
        "ttr":          vocab_size / total_tokens if total_tokens else 0,
        "hapax":        hapax,
        "hapax_ratio":  hapax / vocab_size if vocab_size else 0,
        "duplicates":   duplicates,
        "empty":        empty,
    }


def build_report(stats: dict, split_name: str) -> str:
    s     = stats
    n     = s["n"]
    lines = [f"EDA Report — {split_name}", "=" * 60, ""]

    lines += [
        "1. Sentence length (token count)",
        f"  Average : {s['avg_len']:.1f} tokens",
        f"  Shortest: {s['min_len']} tokens",
        f"  Longest : {s['max_len']} tokens",
        "",
    ]
    for bucket, count in s["buckets"].items():
        lines.append(f"  {bucket:>8} tokens: {count:>6}")
    lines.append("")

    lines += [
        "2. Error rate",
        f"  Sentences with errors : {s['erroneous']} ({100*s['erroneous']/n:.1f}%)",
        f"  Correct sentences     : {s['correct']} ({100*s['correct']/n:.1f}%)",
        "",
    ]

    lines += [
        "3. Label distribution",
        f"  Total tokens : {s['total_tokens']:,}",
        f"  Error tokens : {s['error_tokens']:,} "
        f"({100*s['error_tokens']/s['total_tokens']:.1f}%)",
        "",
    ]
    for label, count in s["label_counts"].most_common():
        pct = 100 * count / s["total_tokens"]
        lines.append(f"  {label:<25} {count:>7}  ({pct:5.1f}%)")
    lines.append("")

    lines += [
        "4. Data quality issues",
        f"  Duplicates : {s['duplicates']}",
        f"  Empty      : {s['empty']}",
        f"  Clean      : {n - s['duplicates'] - s['empty']}",
        "",
    ]

    lines += [
        "5. Vocabulary & domain diversity",
        f"  Total tokens    : {s['total_words']:,}",
        f"  Vocabulary size : {s['vocab_size']:,}",
        f"  TTR             : {s['ttr']:.4f}  (higher = more diverse)",
        f"  Hapax legomena  : {s['hapax']:,} ({100*s['hapax_ratio']:.1f}% of vocab)",
        "",
        "  Top 20 most frequent tokens:",
    ]
    for token, count in s["vocab"].most_common(20):
        pct = 100 * count / s["total_words"]
        lines.append(f"    {token:<20} {count:>7}  ({pct:5.2f}%)")

    return "\n".join(lines)


def plot_eda(stats: dict, split_name: str, output_dir: Path):
    s       = stats
    lengths = s["lengths"]
    avg     = s["avg_len"]
    buckets = s["buckets"]
    vocab   = s["vocab"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"EDA — {split_name}", fontsize=14)
    axes = axes.flatten()

    # ── subplot 0: length histogram ───────────
    axes[0].hist(lengths, bins=40, color="steelblue", edgecolor="white", linewidth=0.5)
    axes[0].axvline(avg, color="tomato", linestyle="--", linewidth=1.5,
                    label=f"Mean: {avg:.1f}")
    axes[0].set_xlabel("Token count")
    axes[0].set_ylabel("Sentence count")
    axes[0].set_title("Raw length histogram")
    axes[0].legend()

    # ── subplot 1: length buckets ─────────────
    axes[1].bar(buckets.keys(), buckets.values(),
                color="steelblue", edgecolor="white")
    axes[1].set_xlabel("Token count bucket")
    axes[1].set_ylabel("Sentence count")
    axes[1].set_title("Length buckets")
    max_v = max(buckets.values())
    for i, (bucket, count) in enumerate(buckets.items()):
        axes[1].text(i, count + max_v * 0.01, str(count),
                     ha="center", va="bottom", fontsize=9)

    # ── subplot 2: top-30 token frequency ─────
    top30 = vocab.most_common(30)
    top30_tokens, top30_counts = zip(*top30)
    axes[2].barh(top30_tokens[::-1], top30_counts[::-1],
                 color="steelblue", edgecolor="white")
    axes[2].set_xlabel("Frequency")
    axes[2].set_title("Top 30 most frequent tokens")
    axes[2].tick_params(axis="y", labelsize=8)

    # ── subplot 3: diversity summary ──────────
    summary_text = (
        f"Vocabulary size : {s['vocab_size']:,}\n"
        f"Total tokens    : {s['total_words']:,}\n\n"
        f"TTR             : {s['ttr']:.4f}\n"
        f"Hapax ratio     : {s['hapax_ratio']:.4f}\n"
        f"({s['hapax']:,} of {s['vocab_size']:,} types appear once)"
    )
    axes[3].axis("off")
    axes[3].text(0.1, 0.5, summary_text, transform=axes[3].transAxes,
                 fontsize=11, verticalalignment="center", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.8))
    axes[3].set_title("Diversity summary")

    plt.tight_layout()
    plot_path = output_dir / f"eda_{split_name}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved to {plot_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      required=True,
                        help="Path to split JSON (e.g. ./data/splits/gector/train.json)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save report and plots")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_dir  = Path(args.output_dir)
    split_name  = input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path}...")
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} sentences\n")

    stats  = analyze(data)
    report = build_report(stats, split_name)

    print(report)

    report_path = output_dir / f"eda_{split_name}.txt"
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nReport saved to {report_path}")

    plot_eda(stats, split_name, output_dir)


if __name__ == "__main__":
    main()