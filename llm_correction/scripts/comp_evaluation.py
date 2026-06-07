import json
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURE HERE
# ─────────────────────────────────────────────

EXPERIMENTS = {
    "no_label":     "/home/mlt_ml1/IdeeFix-DE/llm_correction/results/unlabeled/gemma_unlabeled_conservative_unlabeled_6_rev_results_evaluated.json",
    "hybrid_label": "/home/mlt_ml1/IdeeFix-DE/llm_correction/results/hybrid/gemma_hybrid_labeled_conservative_labeled_6_rev_results_evaluated.json",
    "gold_label":   "/home/mlt_ml1/IdeeFix-DE/llm_correction/results/gold/gemma_gold_labeled_conservative_labeled_6_rev_results_evaluated.json"
}

OUTPUT_CSV = "../results/comparison.csv"

# ─────────────────────────────────────────────
# ROWS TO EXTRACT
# ─────────────────────────────────────────────

ROWS = [
    ("sentence", "BLEU",            lambda d: round(d["bleu"], 2)),
    ("sentence", "exact_match",     lambda d: d["exact_match"]),
    ("sentence", "no_change_total", lambda d: d["no_change"]["total"]),
    ("sentence", "no_change_clean", lambda d: d["no_change"]["clean"]),
    ("sentence", "no_change_dirty", lambda d: d["no_change"]["dirty"]),
    ("sentence", "rate_clean",      lambda d: round(d["no_change"]["rate_clean"], 4)),
    ("sentence", "rate_dirty",      lambda d: round(d["no_change"]["rate_dirty"], 4)),

    ("overall",  "TP",        lambda d: d["metrics"]["overall"]["TP"]),
    ("overall",  "FP",        lambda d: d["metrics"]["overall"]["FP"]),
    ("overall",  "FN",        lambda d: d["metrics"]["overall"]["FN"]),
    ("overall",  "TN",        lambda d: d["metrics"]["overall"]["TN"]),
    ("overall",  "precision", lambda d: d["metrics"]["overall"]["precision"]),
    ("overall",  "recall",    lambda d: d["metrics"]["overall"]["recall"]),
    ("overall",  "F1",        lambda d: d["metrics"]["overall"]["F1"]),
    ("overall",  "F0.5",      lambda d: d["metrics"]["overall"]["F0.5"]),

    ("CAP",  "TP",        lambda d: d["metrics"]["CAP"]["TP"]),
    ("CAP",  "FP",        lambda d: d["metrics"]["CAP"]["FP"]),
    ("CAP",  "FN",        lambda d: d["metrics"]["CAP"]["FN"]),
    ("CAP",  "precision", lambda d: d["metrics"]["CAP"]["precision"]),
    ("CAP",  "recall",    lambda d: d["metrics"]["CAP"]["recall"]),
    ("CAP",  "F1",        lambda d: d["metrics"]["CAP"]["F1"]),
    ("CAP",  "F0.5",      lambda d: d["metrics"]["CAP"]["F0.5"]),

    ("COMP", "TP",        lambda d: d["metrics"]["COMP"]["TP"]),
    ("COMP", "FP",        lambda d: d["metrics"]["COMP"]["FP"]),
    ("COMP", "FN",        lambda d: d["metrics"]["COMP"]["FN"]),
    ("COMP", "precision", lambda d: d["metrics"]["COMP"]["precision"]),
    ("COMP", "recall",    lambda d: d["metrics"]["COMP"]["recall"]),
    ("COMP", "F1",        lambda d: d["metrics"]["COMP"]["F1"]),
    ("COMP", "F0.5",      lambda d: d["metrics"]["COMP"]["F0.5"]),

    ("DECL", "TP",        lambda d: d["metrics"]["DECL"]["TP"]),
    ("DECL", "FP",        lambda d: d["metrics"]["DECL"]["FP"]),
    ("DECL", "FN",        lambda d: d["metrics"]["DECL"]["FN"]),
    ("DECL", "precision", lambda d: d["metrics"]["DECL"]["precision"]),
    ("DECL", "recall",    lambda d: d["metrics"]["DECL"]["recall"]),
    ("DECL", "F1",        lambda d: d["metrics"]["DECL"]["F1"]),
    ("DECL", "F0.5",      lambda d: d["metrics"]["DECL"]["F0.5"]),

    ("labels", "KEEP_CORR",       lambda d: d["label_counts"].get("$KEEP_CORR", 0)),
    ("labels", "KEEP_SUB",        lambda d: d["label_counts"].get("$KEEP_SUB", 0)),
    ("labels", "KEEP_DEL",        lambda d: d["label_counts"].get("$KEEP_DEL", 0)),
    ("labels", "INS",             lambda d: d["label_counts"].get("$INS", 0)),
    ("labels", "KEEP_PUNCT_CORR", lambda d: d["label_counts"].get("$KEEP_PUNCT_CORR", 0)),
    ("labels", "KEEP_PUNCT_SUB",  lambda d: d["label_counts"].get("$KEEP_PUNCT_SUB", 0)),
    ("labels", "KEEP_PUNCT_DEL",  lambda d: d["label_counts"].get("$KEEP_PUNCT_DEL", 0)),
    ("labels", "PUNCT_INS",       lambda d: d["label_counts"].get("$PUNCT_INS", 0)),
    ("labels", "CAP_CORR",        lambda d: d["label_counts"].get("$CAP_CORR", 0)),
    ("labels", "CAP_KEPT",        lambda d: d["label_counts"].get("$CAP_KEPT", 0)),
    ("labels", "CAP_SUB",         lambda d: d["label_counts"].get("$CAP_SUB", 0)),
    ("labels", "CAP_DEL",         lambda d: d["label_counts"].get("$CAP_DEL", 0)),
    ("labels", "COMP_CORR",       lambda d: d["label_counts"].get("$COMP_CORR", 0)),
    ("labels", "COMP_KEPT",       lambda d: d["label_counts"].get("$COMP_KEPT", 0)),
    ("labels", "COMP_SUB",        lambda d: d["label_counts"].get("$COMP_SUB", 0)),
    ("labels", "COMP_DEL",        lambda d: d["label_counts"].get("$COMP_DEL", 0)),
    ("labels", "DECL_CORR",       lambda d: d["label_counts"].get("$DECL_CORR", 0)),
    ("labels", "DECL_KEPT",       lambda d: d["label_counts"].get("$DECL_KEPT", 0)),
    ("labels", "DECL_SUB",        lambda d: d["label_counts"].get("$DECL_SUB", 0)),
    ("labels", "DECL_DEL",        lambda d: d["label_counts"].get("$DECL_DEL", 0)),
]

# ─────────────────────────────────────────────
# STEP 1: BUILD CSV
# ─────────────────────────────────────────────

def build_csv():
    run_labels = list(EXPERIMENTS.keys())   # ["no_label", "hybrid_label", "gold_label"]
    data = {}
    for label, path in EXPERIMENTS.items():
        try:
            with open(path, encoding="utf-8") as f:
                data[label] = json.load(f)
            print(f"  ✓ {label}")
        except FileNotFoundError:
            print(f"  ✗ {label} — file not found: {path}")
            data[label] = None

    out = Path(OUTPUT_CSV)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "metric"] + run_labels)
        for group, metric, extractor in ROWS:
            row = [group, metric]
            for label in run_labels:
                if data[label] is None:
                    row.append("N/A")
                else:
                    try:
                        row.append(extractor(data[label]))
                    except (KeyError, TypeError):
                        row.append("N/A")
            writer.writerow(row)

    print(f"\nCSV saved → {out}")
    return out

# ─────────────────────────────────────────────
# STEP 2: PLOTS
# ─────────────────────────────────────────────

def build_plots(csv_path: Path):
    df = pd.read_csv(csv_path)
    run_labels = list(EXPERIMENTS.keys())
    out_dir = csv_path.parent
    x = np.arange(len(run_labels))
    w = 0.5

    def get(group, metric):
        row = df[(df["group"] == group) & (df["metric"] == metric)]
        if row.empty:
            return [0] * len(run_labels)
        return [float(row.iloc[0][label]) for label in run_labels]

    # ── Plot 1: Overcorrections ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = [
        ("KEEP_SUB",       "#378ADD"),
        ("KEEP_DEL",       "#85B7EB"),
        ("INS",            "#B5D4F4"),
        ("KEEP_PUNCT_SUB", "#D85A30"),
        ("KEEP_PUNCT_DEL", "#F0997B"),
        ("PUNCT_INS",      "#F5C4B3"),
    ]
    bottoms = np.zeros(len(run_labels))
    for metric, color in metrics:
        vals = np.array(get("labels", metric), dtype=float)
        ax.bar(x, vals, width=w, bottom=bottoms, color=color, label=metric, zorder=3)
        bottoms += vals
    for i, total in enumerate(bottoms):
        ax.text(x[i], total + 3, str(int(total)), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels)
    ax.set_ylabel("count")
    ax.set_title("Overcorrections per LLM run", fontsize=13, fontweight="normal")
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, ncol=2, framealpha=0, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(out_dir / "plot_overcorrections.jpg", dpi=150)
    plt.close()
    print(f"Plot saved → {out_dir / 'plot_overcorrections.jpg'}")

    # ── Plot 2: Correct changes ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    corr_metrics = [
        ("CAP_CORR",  "#1D9E75"),
        ("COMP_CORR", "#5DCAA5"),
        ("DECL_CORR", "#9FE1CB"),
    ]
    bottoms = np.zeros(len(run_labels))
    for metric, color in corr_metrics:
        vals = np.array(get("labels", metric), dtype=float)
        ax.bar(x, vals, width=w, bottom=bottoms, color=color, label=metric, zorder=3)
        bottoms += vals
    for i, total in enumerate(bottoms):
        ax.text(x[i], total + 1, str(int(total)), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels)
    ax.set_ylabel("count")
    ax.set_title("Correct changes per LLM run", fontsize=13, fontweight="normal")
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, framealpha=0, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(out_dir / "plot_correct_changes.jpg", dpi=150)
    plt.close()
    print(f"Plot saved → {out_dir / 'plot_correct_changes.jpg'}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading experiment files...")
    csv_path = build_csv()
    print("\nGenerating plots...")
    build_plots(csv_path)