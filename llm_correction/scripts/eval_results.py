"""
eval/eval_llm.py
----------------
Evaluates LLM predictions on German ASR GEC.

Alignment strategy:
    1. difflib.SequenceMatcher aligns err tokens to pred tokens
       handles: equal, replace, delete, insert naturally
    2. German-specific post-processing on top of difflib:
       - compound joining (wie+viel → wieviel)
       - contraction handling (im ↔ in+dem)
       - fuzzy matching for declension changes
    3. Same alignment runs on err→cor for reference
    4. Labels assigned by comparing err group label + pred alignment

Usage:
    python eval_llm.py --results ./results/qwen_labeled_results.json
    python eval_llm.py --results ./results/qwen_labeled_results.json \
                       --output  ./results/qwen_labeled_evaluated.json
"""

import json
import argparse
import difflib
import editdistance
from pathlib import Path
from collections import Counter
import os
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.constants import (
    get_best_gpu,
    SEP,
    CONTRACTIONS_FORWARD,
    CONTRACTIONS_REVERSE,
)
from utils.stanza_utils import load_stanza_word_tokenizer, word_tokenize

os.environ["CUDA_VISIBLE_DEVICES"] = str(get_best_gpu())

from sacrebleu.metrics import BLEU


# ─────────────────────────────────────────────
# CONTRACTION HELPERS
# ─────────────────────────────────────────────

def is_contraction(text: str) -> bool:
    return text.lower() in CONTRACTIONS_FORWARD


def contraction_parts(text: str) -> tuple[str, str] | None:
    return CONTRACTIONS_FORWARD.get(text.lower())


def contraction_form(a: str, b: str) -> str | None:
    return CONTRACTIONS_REVERSE.get((a.lower(), b.lower()))


# ─────────────────────────────────────────────
# LABELED STRING PARSING
# ─────────────────────────────────────────────
KNOWN_LABELS = ("$KEEP", "$WRONG_DECL", "$WRONG_COMP", "$WRONG_CAP")

def parse_labeled(labeled_str: str) -> list[dict]:
    tokens = []
    for chunk in labeled_str.strip().split(" "):
        if SEP in chunk:
            text, label = chunk.split(SEP, 1)
            tokens.append({"text": text, "label": label})
        else:
            for known in KNOWN_LABELS:
                if known in chunk:
                    text = chunk[:chunk.index(known)]
                    tokens.append({"text": text, "label": known})

    return tokens


def get_compound_groups(labeled_tokens: list[dict]) -> list[list[dict]]:
    """
    Compound group: one or more $WRONG_COMP tokens + final $KEEP tail.
    All other tokens → single-token groups.
    """
    groups = []
    i      = 0
    while i < len(labeled_tokens):
        tok = labeled_tokens[i]
        if tok["label"] == "$WRONG_COMP":
            group = [tok]
            j     = i + 1
            while j < len(labeled_tokens) and labeled_tokens[j]["label"] == "$WRONG_COMP":
                group.append(labeled_tokens[j])
                j += 1
            if j < len(labeled_tokens) and labeled_tokens[j]["label"] == "$KEEP":
                group.append(labeled_tokens[j])
                j += 1
            groups.append(group)
            i = j
        else:
            groups.append([tok])
            i += 1
    return groups


# ─────────────────────────────────────────────
# ALIGNMENT HELPERS
# ─────────────────────────────────────────────

def strip_umlauts(s: str) -> str:
    return (s.replace("ä", "a").replace("ö", "o")
             .replace("ü", "u").replace("ß", "ss"))


def fuzzy_match(a: str, b: str) -> bool:
    al, bl = a.lower(), b.lower()
    if al == bl:
        return True
    if len(al) <= 2 or len(bl) <= 2:
        return False
    max_len = max(len(al), len(bl), 1)
    dist    = editdistance.eval(strip_umlauts(al), strip_umlauts(bl))
    return dist / max_len <= 0.2


def glue_candidates(parts: list[str]) -> set[str]:
    base       = "".join(parts)
    candidates = {base}
    for i in range(1, len(parts)):
        left  = "".join(parts[:i])
        right = "".join(parts[i:])
        candidates.add(left + "s" + right)
        if left.endswith("s"):
            candidates.add(left[:-1] + right)
    return candidates


def is_punct(text: str) -> bool:
    return len(text) > 0 and all(not c.isalnum() for c in text)


# ─────────────────────────────────────────────
# DIFFLIB-BASED ALIGNMENT
# ─────────────────────────────────────────────

def align_with_difflib(
    err_groups: list[list[dict]],
    tgt_tokens: list[dict],
) -> dict:
    """
    Align err groups to tgt tokens using difflib as the backbone.

    Returns alignment dict: g_idx → {tgt_tok, type, tgt_toks}
    tgt_toks: all tgt tokens aligned to this group (for n-to-1 cases)
    """
    # flatten err groups to token texts for difflib
    # keep mapping: flat_idx → g_idx
    flat_err   = []
    flat_to_g  = []
    for g_idx, group in enumerate(err_groups):
        for tok in group:
            flat_err.append(tok["text"].lower())
            flat_to_g.append(g_idx)

    flat_tgt = [t["text"].lower() for t in tgt_tokens]

    matcher = difflib.SequenceMatcher(
        None,
        flat_err,
        flat_tgt,
        autojunk=False,
    )

    # g_idx → list of (tgt_tok, match_type)
    g_to_tgt: dict[int, list] = {g_idx: [] for g_idx in range(len(err_groups))}

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        err_chunk = flat_err[i1:i2]
        tgt_chunk = tgt_tokens[j1:j2]
        g_indices = flat_to_g[i1:i2]

        if tag == "equal":
            # 1-to-1: each err token matches its tgt token exactly
            for k, g_idx in enumerate(g_indices):
                if k < len(tgt_chunk):
                    g_to_tgt[g_idx].append((tgt_chunk[k], "exact"))

        elif tag == "replace":
            # n err tokens → m tgt tokens
            # try to do a smarter alignment within the replace block
            _align_replace_block(
                err_groups, g_indices, tgt_chunk, g_to_tgt
            )

        elif tag == "delete":
            # err tokens deleted by LLM — no tgt token
            for g_idx in g_indices:
                g_to_tgt[g_idx].append((None, "deleted"))

        elif tag == "insert":
            # tgt tokens inserted by LLM — no err token
            # these become $INS — handled separately
            pass

    # build final alignment: g_idx → best single tgt_tok + type
    alignment   = {}
    matched_ids = set()

    for g_idx, matches in g_to_tgt.items():
        if not matches:
            alignment[g_idx] = {"tgt_tok": None, "type": None, "extra_toks": []}
            continue

        # filter out (None, deleted) if there's a real match
        real = [(tok, t) for tok, t in matches if tok is not None]
        if real:
            # take first real match as primary
            tgt_tok, match_type = real[0]
            extra = [tok for tok, _ in real[1:]]
            alignment[g_idx] = {
                "tgt_tok":   tgt_tok,
                "type":      match_type,
                "extra_toks": extra,
            }
            matched_ids.add(tgt_tok["id"])
            for tok in extra:
                matched_ids.add(tok["id"])
        else:
            alignment[g_idx] = {"tgt_tok": None, "type": "deleted", "extra_toks": []}

    return alignment, matched_ids


def _align_replace_block(
    err_groups:  list[list[dict]],
    g_indices:   list[int],
    tgt_chunk:   list[dict],
    g_to_tgt:    dict,
):
    """
    Handle replace blocks with German-aware matching:
    - compound joins (Land+wirtschaft → Landwirtschaft)
    - contractions (in+dem → im)
    - fuzzy matches (großer → großen)
    - deletions within block
    - insertions within block
    """
    if not g_indices or not tgt_chunk:
        # all deleted
        for g_idx in g_indices:
            g_to_tgt[g_idx].append((None, "deleted"))
        return

    # get unique g_indices in order
    seen      = set()
    g_ordered = []
    for g in g_indices:
        if g not in seen:
            seen.add(g)
            g_ordered.append(g)

    err_texts = [
        "".join(t["text"] for t in err_groups[g]).lower()
        for g in g_ordered
    ]
    tgt_texts = [t["text"].lower() for t in tgt_chunk]

    # try to greedily match err tokens to tgt tokens
    tgt_ptr = 0
    for ei, g_idx in enumerate(g_ordered):
        err_text = err_texts[ei]

        if tgt_ptr >= len(tgt_chunk):
            # no more tgt tokens → deleted
            g_to_tgt[g_idx].append((None, "deleted"))
            continue

        cand      = tgt_chunk[tgt_ptr]
        cand_text = cand["text"].lower()

        # exact match
        if cand_text == err_text:
            g_to_tgt[g_idx].append((cand, "exact"))
            tgt_ptr += 1

        # fuzzy match (declension change etc.)
        elif fuzzy_match(cand_text, err_text):
            g_to_tgt[g_idx].append((cand, "fuzzy"))
            tgt_ptr += 1

        # compound join: err="land" next_err="wirtschaft" tgt="landwirtschaft"
        elif (ei + 1 < len(g_ordered)):
            next_err = err_texts[ei + 1]
            candidates = glue_candidates([err_text, next_err])
            if cand_text in candidates:
                g_to_tgt[g_idx].append((cand, "compound_joined"))
                # next err group gets marked as consumed
                g_to_tgt[g_ordered[ei + 1]].append((None, "consumed_by_join"))
                tgt_ptr += 1
                continue

            # try startswith (partial compound)
            if cand_text.startswith(err_text) and len(cand_text) > len(err_text) + 1:
                g_to_tgt[g_idx].append((cand, "compound_startswith"))
                tgt_ptr += 1
                continue

            # contraction merge: err="in" next_err="dem" tgt="im"
            cf = contraction_form(err_text, next_err)
            if cf and cand_text == cf:
                g_to_tgt[g_idx].append((cand, "contraction_merged"))
                g_to_tgt[g_ordered[ei + 1]].append((None, "consumed_by_join"))
                tgt_ptr += 1
                continue

            # contraction split: err="im" tgt=["in","dem"]
            if is_contraction(err_text):
                parts = contraction_parts(err_text)
                if (parts and
                        cand_text == parts[0] and
                        tgt_ptr + 1 < len(tgt_chunk) and
                        tgt_chunk[tgt_ptr + 1]["text"].lower() == parts[1]):
                    g_to_tgt[g_idx].append((cand, "contraction_split"))
                    tgt_ptr += 2
                    continue

            # next tgt matches better → current err deleted
            if tgt_ptr + 1 < len(tgt_chunk):
                next_tgt = tgt_chunk[tgt_ptr + 1]["text"].lower()
                if next_tgt == err_text or fuzzy_match(next_tgt, err_text):
                    # skip current tgt (it's an insertion) → mark err as deleted
                    g_to_tgt[g_idx].append((None, "deleted"))
                    continue

            # fallback substitution
            g_to_tgt[g_idx].append((cand, "sub"))
            tgt_ptr += 1

        else:
            # last err token — check contraction split
            if is_contraction(err_text):
                parts = contraction_parts(err_text)
                if (parts and
                        cand_text == parts[0] and
                        tgt_ptr + 1 < len(tgt_chunk) and
                        tgt_chunk[tgt_ptr + 1]["text"].lower() == parts[1]):
                    g_to_tgt[g_idx].append((cand, "contraction_split"))
                    tgt_ptr += 2
                    continue

            if fuzzy_match(cand_text, err_text):
                g_to_tgt[g_idx].append((cand, "fuzzy"))
            else:
                g_to_tgt[g_idx].append((cand, "sub"))
            tgt_ptr += 1


# ─────────────────────────────────────────────
# SENTENCE EVALUATION
# ─────────────────────────────────────────────

def evaluate_sentence(
    labeled_tokens: list[dict],
    pred_tokens:    list[dict],
    cor_tokens:     list[dict],
    err_text:       str,
    cor_text:       str,
    pred_text:      str,
) -> dict:

    counts      = Counter()
    pred_labels = []

    err_groups = get_compound_groups(labeled_tokens)

    err_to_cor,  cor_matched  = align_with_difflib(err_groups, cor_tokens)
    err_to_pred, pred_matched = align_with_difflib(err_groups, pred_tokens)

    # collect unmatched pred tokens for insertion detection
    all_matched_pred_ids = pred_matched

    for g_idx, group in enumerate(err_groups):
        err_text_g      = "".join(t["text"] for t in group)
        group_label     = group[0]["label"]

        cor_info   = err_to_cor[g_idx]
        pred_info  = err_to_pred[g_idx]

        cor_tok        = cor_info["tgt_tok"]
        pred_tok       = pred_info["tgt_tok"]
        match_type_pred = pred_info["type"]

        cor_text_g  = cor_tok["text"]  if cor_tok  else None
        pred_text_g = pred_tok["text"] if pred_tok else None

        entry = {"err": err_text_g, "cor": cor_text_g, "pred": pred_text_g}

        # ── consumed by join → sub for $KEEP ──────────────────────────────
        if match_type_pred == "consumed_by_join":
            label = "$KEEP_SUB"

        # ── $KEEP ─────────────────────────────────────────────────────────
        elif group_label == "$KEEP":
            if pred_text_g is None:
                label = "$KEEP_PUNCT_DEL" if is_punct(err_text_g) else "$KEEP_DEL"
            elif is_punct(err_text_g):
                label = "$KEEP_PUNCT_CORR" if pred_text_g == err_text_g else "$KEEP_PUNCT_SUB"
            elif pred_text_g == err_text_g:
                label = "$KEEP_CORR"
            elif match_type_pred in ("contraction_split", "contraction_merged"):
                label = "$KEEP_CORR"
            elif is_contraction(err_text_g):
                parts = contraction_parts(err_text_g)
                label = "$KEEP_CORR" if (parts and pred_text_g.lower() == parts[0]) else "$KEEP_SUB"
            elif is_contraction(pred_text_g or ""):
                parts = contraction_parts(pred_text_g)
                label = "$KEEP_CORR" if (parts and err_text_g.lower() == parts[0]) else "$KEEP_SUB"
            else:
                label = "$KEEP_SUB"

        # ── $WRONG_COMP ───────────────────────────────────────────────────
        elif len(group) > 1:
            if pred_text_g is None:
                if match_type_pred == "consumed_by_join":
                    label = "$COMP_SUB"
                else:
                    head_lower = group[0]["text"].lower()
                    near_match = any(
                        tok["id"] not in pred_matched and
                        not is_punct(tok["text"]) and
                        len(tok["text"]) >= 4 and
                        tok["text"].lower()[:4] == head_lower[:4]
                        for tok in pred_tokens
                    )
                    label = "$COMP_SUB" if near_match else "$COMP_DEL"

            elif cor_text_g is not None and pred_text_g == cor_text_g:
                label = "$COMP_CORR"

            elif match_type_pred in ("compound_joined", "compound_startswith"):
                # joined but not matching cor exactly
                label = "$COMP_SUB"

            elif "-" in (pred_text_g or ""):
                label = "$COMP_SUB"

            elif match_type_pred == "sub" and pred_text_g == group[0]["text"]:
                label = "$COMP_KEPT"

            elif pred_text_g == group[0]["text"]:
                label = "$COMP_KEPT"

            else:
                label = "$COMP_SUB"

        # ── $WRONG_CAP ────────────────────────────────────────────────────
        elif group_label == "$WRONG_CAP":
            if pred_text_g is None:
                label = "$CAP_DEL"
            elif cor_text_g is not None and pred_text_g == cor_text_g:
                label = "$CAP_CORR"
            elif pred_text_g == err_text_g:
                label = "$CAP_KEPT"
            else:
                label = "$CAP_SUB"

        # ── $WRONG_DECL ───────────────────────────────────────────────────
        elif group_label == "$WRONG_DECL":
            if pred_text_g is None:
                label = "$DECL_DEL"
            elif cor_text_g is not None and pred_text_g == cor_text_g:
                label = "$DECL_CORR"
            elif pred_text_g == err_text_g:
                label = "$DECL_KEPT"
            else:
                label = "$DECL_SUB"

        # ── fallback ──────────────────────────────────────────────────────
        else:
            label = "$KEEP_CORR" if pred_text_g == err_text_g else "$KEEP_SUB"

        counts[label] += 1
        entry["label"] = label
        pred_labels.append(entry)

    # ── insertions: pred tokens not matched to any err group ──────────────
    for tok in pred_tokens:
        if tok["id"] not in all_matched_pred_ids:
            label = "$PUNCT_INS" if is_punct(tok["text"]) else "$INS"
            counts[label] += 1
            pred_labels.append({"pred": tok["text"], "label": label})

    return {"counts": counts, "pred_labels": pred_labels}


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def compute_metrics(total: Counter) -> dict:

    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    def f_score(p, r, beta=1.0):
        if p + r == 0:
            return 0.0
        return (1 + beta**2) * p * r / (beta**2 * p + r)

    def class_prf(corr, kept, sub, del_):
        tp = corr
        fp = sub
        fn = kept + del_
        p  = safe_div(tp, tp + fp)
        r  = safe_div(tp, tp + fn)
        return {
            "TP": tp, "FP": fp, "FN": fn,
            "precision": round(p,  4),
            "recall":    round(r,  4),
            "F1":        round(f_score(p, r, 1.0), 4),
            "F0.5":      round(f_score(p, r, 0.5), 4),
        }

    TP = (total["$CAP_CORR"]  + total["$COMP_CORR"]  + total["$DECL_CORR"])
    FP = (total["$KEEP_SUB"]  + total["$KEEP_DEL"]   +
          total["$KEEP_PUNCT_SUB"] + total["$KEEP_PUNCT_DEL"] +
          total["$CAP_SUB"]   + total["$COMP_SUB"]   + total["$DECL_SUB"] +
          total["$INS"]       + total["$PUNCT_INS"])
    FN = (total["$CAP_KEPT"]  + total["$CAP_DEL"]    +
          total["$COMP_KEPT"] + total["$COMP_DEL"]   +
          total["$DECL_KEPT"] + total["$DECL_DEL"])
    TN = (total["$KEEP_CORR"] + total["$KEEP_PUNCT_CORR"])

    P  = safe_div(TP, TP + FP)
    R  = safe_div(TP, TP + FN)

    return {
        "overall": {
            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "precision": round(P, 4),
            "recall":    round(R, 4),
            "F1":        round(f_score(P, R, 1.0), 4),
            "F0.5":      round(f_score(P, R, 0.5), 4),
        },
        "CAP":  class_prf(total["$CAP_CORR"],  total["$CAP_KEPT"],
                          total["$CAP_SUB"],   total["$CAP_DEL"]),
        "COMP": class_prf(total["$COMP_CORR"], total["$COMP_KEPT"],
                          total["$COMP_SUB"],  total["$COMP_DEL"]),
        "DECL": class_prf(total["$DECL_CORR"], total["$DECL_KEPT"],
                          total["$DECL_SUB"],  total["$DECL_DEL"]),
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="Path to results JSON from run_llm.py")
    parser.add_argument("--output",  default=None,
                        help="Output path (default: replaces .json with _evaluated.json)")
    args = parser.parse_args()

    print("Loading data...")
    with open(args.results, encoding="utf-8") as f:
        results = json.load(f)
    print(f"  Results : {len(results)} sentences")

    print("Loading stanza...")
    nlp = load_stanza_word_tokenizer(use_gpu=False)

    print("Evaluating...")
    total_counts    = Counter()
    all_entries     = []
    hypotheses      = []
    references      = []
    exact_match     = 0
    no_change_total = 0
    no_change_clean = 0
    no_change_dirty = 0

    n_clean = sum(1 for r in results if r["cor"] == r["err"])
    n_dirty = sum(1 for r in results if r["cor"] != r["err"])

    for i, item in enumerate(results):
        sent_id      = item["sent_id"]
        err_text     = item["err"]
        cor_text     = item["cor"]
        labeled_text = item.get("labeled") or item.get("gold_labeled", "")
        pred_text    = item["predicted"]

        labeled_tokens = parse_labeled(labeled_text)
        pred_tokens    = word_tokenize(nlp, pred_text)
        cor_tokens     = word_tokenize(nlp, cor_text)

        result = evaluate_sentence(
            labeled_tokens = labeled_tokens,
            pred_tokens    = pred_tokens,
            cor_tokens     = cor_tokens,
            err_text       = err_text,
            cor_text       = cor_text,
            pred_text      = pred_text,
        )

        total_counts += result["counts"]

        if pred_text.strip() == err_text.strip():
            no_change_total += 1
            if err_text.strip() == cor_text.strip():
                no_change_clean += 1
            else:
                no_change_dirty += 1

        if pred_text.strip() == cor_text.strip():
            exact_match += 1

        hypotheses.append(pred_text)
        references.append(cor_text)

        all_entries.append({
            "sent_id":     sent_id,
            "err":         err_text,
            "cor":         cor_text,
            "predicted":   pred_text,
            "pred_labels": result["pred_labels"],
            "counts":      dict(result["counts"]),
        })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(results)} done")

    n       = len(all_entries)
    print("total_counts:", dict(total_counts))
    print("TP would be:", total_counts["$CAP_CORR"] + total_counts["$COMP_CORR"] + total_counts["$DECL_CORR"])
    metrics = compute_metrics(total_counts)
    bleu    = BLEU(effective_order=True).corpus_score(hypotheses, [references])

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"\n  Sentences evaluated : {n}")
    print(f"  Exact match         : {exact_match} ({100*exact_match/n:.1f}%)")
    print(f"  BLEU                : {bleu.score:.2f}")

    print(f"\n── No-change rate ──────────────────────────────────────────")
    print(f"  pred == err (total)        : {no_change_total} ({100*no_change_total/n:.1f}%)")
    print(f"  pred == err on clean sents : {no_change_clean}/{n_clean} "
          f"({100*no_change_clean/max(n_clean,1):.1f}%) ← correct laziness")
    print(f"  pred == err on dirty sents : {no_change_dirty}/{n_dirty} "
          f"({100*no_change_dirty/max(n_dirty,1):.1f}%) ← lazy failure")

    print(f"\n── Token label counts ──────────────────────────────────────")
    for label, count in sorted(total_counts.items()):
        print(f"  {label:<22}: {count:>6}")

    print(f"\n── Punctuation ─────────────────────────────────────────────")
    print(f"  $KEEP_PUNCT_CORR : {total_counts['$KEEP_PUNCT_CORR']:>6}  (correctly unchanged)")
    print(f"  $KEEP_PUNCT_SUB  : {total_counts['$KEEP_PUNCT_SUB']:>6}  (incorrectly changed)")
    print(f"  $KEEP_PUNCT_DEL  : {total_counts['$KEEP_PUNCT_DEL']:>6}  (incorrectly deleted)")
    print(f"  $PUNCT_INS       : {total_counts['$PUNCT_INS']:>6}  (incorrectly inserted)")

    print(f"\n── Overall ─────────────────────────────────────────────────")
    ov = metrics["overall"]
    print(f"  TP={ov['TP']}  FP={ov['FP']}  FN={ov['FN']}  TN={ov['TN']}")
    print(f"  Precision : {ov['precision']:.4f}")
    print(f"  Recall    : {ov['recall']:.4f}")
    print(f"  F1        : {ov['F1']:.4f}")
    print(f"  F0.5      : {ov['F0.5']:.4f}")

    for cls in ("CAP", "COMP", "DECL"):
        m = metrics[cls]
        print(f"\n── {cls} ────────────────────────────────────────────────────")
        print(f"  TP={m['TP']}  FP={m['FP']}  FN={m['FN']}")
        print(f"  Precision : {m['precision']:.4f}")
        print(f"  Recall    : {m['recall']:.4f}")
        print(f"  F1        : {m['F1']:.4f}")
        print(f"  F0.5      : {m['F0.5']:.4f}")

    print("\n" + "=" * 60)

    out = args.output or args.results.replace(".json", "_evaluated.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "metrics":      metrics,
            "bleu":         bleu.score,
            "exact_match":  exact_match,
            "no_change": {
                "total":      no_change_total,
                "clean":      no_change_clean,
                "dirty":      no_change_dirty,
                "rate_total": round(no_change_total / n, 4),
                "rate_clean": round(no_change_clean / max(n_clean, 1), 4),
                "rate_dirty": round(no_change_dirty / max(n_dirty, 1), 4),
            },
            "label_counts": dict(total_counts),
            "sentences":    all_entries,
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()