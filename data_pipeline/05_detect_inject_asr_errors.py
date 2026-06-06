"""
03_inject_detect_error_asr.py
------------------------------
Detects real ASR errors in whisper output and injects additional errors
into clean sentences to reach target error counts.

Error types:
    $WRONG_CAP   — nominalized verb written in lowercase
    $WRONG_DECL  — adjective with wrong declension ending
    $WRONG_COMP  — compound word incorrectly split into parts

Input:  annotated ASR hypothesis JSON files (reference + hypothesis)
Output: labeled JSON with err/cor/labeled fields
"""

import sys
import json
import random
import argparse
import editdistance
from pathlib import Path
from collections import defaultdict, Counter

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.constants import SEP, COMPOUND_BLOCKLIST, STOPWORDS, VERB_LIST_PATH
from utils.adj_error_intro import ALL_VALID_ENDINGS
from utils.injection_utils import (
    build_position_map,
    collect_wrong_cap,
    collect_wrong_decl,
    collect_wrong_comp,
    apply_edits,
    print_summary,
)

VERB_LIST = Path(VERB_LIST_PATH)

TARGET_COUNTS = {
    "$WRONG_CAP":  8000,
    "$WRONG_DECL": 8000,
    "$WRONG_COMP": 12000,
}
RANDOM_SEED = 42


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────
def validate_item(item: dict) -> bool:
    ref = item.get("reference", {})
    hyp = item.get("hypothesis", {})

    ref_text   = ref.get("text", "")   if isinstance(ref, dict) else ""
    hyp_text   = hyp.get("text", "")   if isinstance(hyp, dict) else ""
    ref_tokens = ref.get("tokens", []) if isinstance(ref, dict) else []
    hyp_tokens = hyp.get("tokens", []) if isinstance(hyp, dict) else []

    required_fields = {"id", "text", "upos", "xpos", "lemma"}

    return (
        bool(ref_text.strip()) and
        bool(hyp_text.strip()) and
        bool(ref_tokens) and
        bool(hyp_tokens) and
        all(required_fields <= set(t.keys()) for t in ref_tokens[:5]) and
        all(required_fields <= set(t.keys()) for t in hyp_tokens[:5])
    )

# ─────────────────────────────────────────────
# LEMMA HELPERS
# ─────────────────────────────────────────────

def get_lemmas(tok: dict) -> set[str]:
    raw = (tok.get("lemma") or "").strip()
    if not raw:
        return set()
    return {v.lower().strip() for v in raw.split("|") if v.strip()}

def lemmas_overlap(tok_a: dict, tok_b: dict) -> bool:
    return bool(get_lemmas(tok_a) & get_lemmas(tok_b))

def lemma_in_set(tok: dict, s: set[str]) -> bool:
    return bool(get_lemmas(tok) & s)

# ─────────────────────────────────────────────
# COMPOUND DETECTION
# ─────────────────────────────────────────────

def glue_candidates(parts: list[str]) -> set[str]:
    base       = "".join(parts)
    candidates = {base}
    for boundary in range(1, len(parts)):
        left  = "".join(parts[:boundary])
        right = "".join(parts[boundary:])
        if left.endswith("s") or right.startswith("s"):
            candidates.add(left[:-1] + right)
    return candidates

def group_is_valid(group: list) -> bool:
    return (
        all(len(t["text"]) >= 2 for t in group) and
        all(t.get("upos") != "PUNCT" and t["text"][0].isalpha() for t in group) and
        not all(t["text"].lower() in STOPWORDS for t in group)
    )

def ref_match_is_valid(ref_tok: dict) -> bool:
    ref_text = ref_tok["text"]
    has_internal_upper = any(c.isupper() for c in ref_text[1:])
    return not (has_internal_upper and ref_tok.get("upos") != "PROPN")

def find_compound_splits(ref_tokens: list, hyp_tokens: list) -> dict:
    ref_by_lower = {
        t["text"].lower(): t
        for t in ref_tokens
        if t["text"].lower() not in COMPOUND_BLOCKLIST
    }

    compounds = {}
    n = len(hyp_tokens)

    for start_i in range(n):
        if hyp_tokens[start_i]["id"] in compounds:
            continue
        for end_i in range(start_i + 2, min(start_i + 5, n + 1)):
            group = hyp_tokens[start_i:end_i]
            if not group_is_valid(group):
                continue
            parts_lower = [t["text"].lower() for t in group]
            matched_ref = next(
                (ref_by_lower[c] for c in glue_candidates(parts_lower) if c in ref_by_lower),
                None
            )
            if matched_ref and ref_match_is_valid(matched_ref):
                compounds[group[0]["id"]] = {
                    "ref_token":  matched_ref,
                    "hyp_tokens": group,
                }
                break

    return compounds

# ─────────────────────────────────────────────
# TOKEN ALIGNMENT
# ─────────────────────────────────────────────

def reindex_tokens(tokens: list) -> list:
    return [{**tok, "id": i + 1} for i, tok in enumerate(tokens)]

def strip_umlauts(s: str) -> str:
    return (s.replace("ä", "a").replace("ö", "o")
             .replace("ü", "u").replace("ß", "ss"))

def fuzzy_lemma_match(a: str, b: str) -> bool:
    max_len = max(len(a), len(b), 1)
    if len(a) >= 4 and len(b) >= 4:
        if a.startswith(b) or b.startswith(a):
            return True
    dist_norm = editdistance.eval(strip_umlauts(a), strip_umlauts(b))
    return dist_norm / max_len <= 0.2

def token_similarity(ref_tok: dict, hyp_tok: dict) -> float:
    ref_text = ref_tok.get("text", "").lower()
    hyp_text = hyp_tok.get("text", "").lower()

    if lemmas_overlap(ref_tok, hyp_tok):
        return 1.0
    if ref_text == hyp_text:
        return 1.0

    for rl in get_lemmas(ref_tok):
        for hl in get_lemmas(hyp_tok):
            if fuzzy_lemma_match(rl, hl):
                return 1.0

    dist      = editdistance.eval(ref_text, hyp_text)
    max_len   = max(len(ref_text), len(hyp_text), 1)
    norm_dist = dist / max_len
    if norm_dist <= 0.2:
        return 0.7

    return 0.0

def align_tokens(ref_tokens: list, hyp_tokens: list,
                 gap_penalty: float = -0.4) -> list[tuple]:
    n, m = len(ref_tokens), len(hyp_tokens)
    dp   = [[0.0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1): dp[i][0] = i * gap_penalty
    for j in range(m + 1): dp[0][j] = j * gap_penalty

    sim_cache = {}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sim = token_similarity(ref_tokens[i-1], hyp_tokens[j-1])
            sim_cache[(i, j)] = sim
            dp[i][j] = max(
                dp[i-1][j-1] + sim,
                dp[i-1][j]   + gap_penalty,
                dp[i][j-1]   + gap_penalty,
            )

    pairs = []
    i, j  = n, m
    while i > 0 and j > 0:
        sim = sim_cache[(i, j)]
        if dp[i][j] == dp[i-1][j-1] + sim:
            pairs.append((ref_tokens[i-1], hyp_tokens[j-1], sim))
            i -= 1; j -= 1
        elif dp[i][j] == dp[i-1][j] + gap_penalty:
            i -= 1
        else:
            j -= 1

    return pairs[::-1]

def build_alignment_map(ref_tokens: list, hyp_tokens: list,
                        min_score: float = 0.5) -> dict:
    compounds        = find_compound_splits(ref_tokens, hyp_tokens)
    compound_hyp_ids = {
        hyp_tok["id"]
        for match in compounds.values()
        for hyp_tok in match["hyp_tokens"]
    }
    compound_ref_ids = {match["ref_token"]["id"] for match in compounds.values()}

    free_ref = [t for t in ref_tokens if t["id"] not in compound_ref_ids]
    free_hyp = [t for t in hyp_tokens if t["id"] not in compound_hyp_ids]
    nw_pairs = align_tokens(free_ref, free_hyp)

    result = {}
    for match in compounds.values():
        ref_tok = match["ref_token"]
        for i, hyp_tok in enumerate(match["hyp_tokens"]):
            result[hyp_tok["id"]] = {
                "ref_token": ref_tok,
                "type":      "compound_head" if i == 0 else "compound_part",
            }
    for ref_tok, hyp_tok, score in nw_pairs:
        if score >= min_score:
            result[hyp_tok["id"]] = {
                "ref_token": ref_tok,
                "type":      "aligned",
            }

    return result

# ─────────────────────────────────────────────
# ERROR DETECTION
# ─────────────────────────────────────────────

def detect_comp(hyp_tok: dict, alignment_map: dict) -> dict | None:
    entry = alignment_map.get(hyp_tok["id"])
    if not entry or entry["type"] not in ("compound_head", "compound_part"):
        return None
    ref_tok = entry["ref_token"]
    is_head = entry["type"] == "compound_head"
    return {
        "id":           hyp_tok["id"],
        "text":         hyp_tok["text"],
        "label":        "$CORRECT" if is_head else "$WRONG_COMP",
        "correct_text": ref_tok["text"] if is_head else hyp_tok["text"],
    }

def get_adj_ending(tok: dict) -> str | None:
    text = (tok.get("text") or "").lower()
    if not text:
        return None
    for lemma in get_lemmas(tok):
        if text.startswith(lemma):
            return text[len(lemma):]
        lemma_stripped = lemma.rstrip("e")
        if text.startswith(lemma_stripped):
            return text[len(lemma_stripped):]
    return None

def is_declension_ending_difference(hyp_tok: dict, ref_tok: dict) -> bool:
    hyp_ending = get_adj_ending(hyp_tok)
    ref_ending = get_adj_ending(ref_tok)
    if hyp_ending is None or ref_ending is None:
        return False
    if hyp_ending not in ALL_VALID_ENDINGS:
        return False
    if ref_ending not in ALL_VALID_ENDINGS:
        return False
    return hyp_ending != ref_ending

def resolve_adja_head(tok: dict, id_to_tok: dict) -> dict:
    current = tok
    visited = set()
    while True:
        if current.get("id") in visited:
            break
        visited.add(current.get("id"))
        if current.get("deprel") == "conj":
            head_id = current.get("head")
            head    = id_to_tok.get(head_id)
            if head and head.get("upos") == "ADJA":
                current = head
                continue
        break
    return current

def get_governing_det(tok: dict, id_to_tok: dict) -> dict | None:
    root_adj = resolve_adja_head(tok, id_to_tok)
    tok_id   = root_adj.get("id")
    head_id  = root_adj.get("head")
    if not tok_id or not head_id:
        return None
    head_tok = id_to_tok.get(head_id)
    if head_tok and head_tok.get("upos") in ("ADJ", "ADJA"):
        tok_id  = head_tok.get("id", tok_id)
        head_id = head_tok.get("head", head_id)
    if not head_id:
        return None
    for t in id_to_tok.values():
        if (t.get("upos") == "DET" and
                t.get("head") == head_id and
                t.get("id", 999) < tok_id):
            return t
    return None

def det_context_matches(hyp_tok: dict, ref_tok: dict,
                        hyp_id_to_tok: dict, ref_id_to_tok: dict) -> bool:
    hyp_det = get_governing_det(hyp_tok, hyp_id_to_tok)
    ref_det = get_governing_det(ref_tok, ref_id_to_tok)
    if hyp_det is None and ref_det is None:
        return True
    if (hyp_det is None) != (ref_det is None):
        return False
    return ((hyp_det.get("text") or "").lower() ==
            (ref_det.get("text") or "").lower())

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_paths", nargs="+", required=True,
                        help="Annotated ASR JSON files")
    parser.add_argument("--output",      required=True,
                        help="Output JSON path")
    parser.add_argument("--sent_counter_start", type=int, default=50001,
                        help="Starting sent_id counter (default: 50001)")
    args = parser.parse_args()

    input_paths   = [Path(p) for p in args.input_paths]
    output        = Path(args.output)
    sent_counter  = args.sent_counter_start

    # ── load data ────────────────────────────
    data = []
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")
        with open(path, encoding="utf-8") as f:
            data.extend(json.load(f))
    print(f"Loaded {len(data)} sentences")

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
        ref_text = sentence["reference"]["text"]
        if ref_text not in seen_texts:
            seen_texts.add(ref_text)
            unique_data.append(sentence)
    print(f"After deduplication: {len(unique_data)} sentences "
          f"({len(data) - len(unique_data)} duplicates removed)")
    data = unique_data

    results          = []
    detected_results = []
    clean_items      = []
    detected_counts  = {"$WRONG_CAP": 0, "$WRONG_DECL": 0, "$WRONG_COMP": 0}

    # ── pass 1: detection ────────────────────
    for item in data:
        if not validate_item(item):
            continue

        ref_tokens    = reindex_tokens(item["reference"]["tokens"])
        hyp_tokens    = reindex_tokens(item["hypothesis"]["tokens"])
        hyp_id_to_tok = {t["id"]: t for t in hyp_tokens}
        ref_id_to_tok = {t["id"]: t for t in ref_tokens}
        alignment_map = build_alignment_map(ref_tokens, hyp_tokens)

        token_labels = []
        skip_ids     = set()

        for hyp_tok in hyp_tokens:
            hyp_id = hyp_tok["id"]
            if hyp_id in skip_ids:
                continue

            comp_label = detect_comp(hyp_tok, alignment_map)
            if comp_label:
                ref_tok   = alignment_map[hyp_id]["ref_token"]
                group_ids = sorted([
                    tid for tid, e in alignment_map.items()
                    if e["ref_token"]["id"] == ref_tok["id"]
                ])
                for gid in group_ids:
                    skip_ids.add(gid)
                group_toks = sorted(
                    [t for t in hyp_tokens if t["id"] in group_ids],
                    key=lambda t: t["id"]
                )
                for i, gt in enumerate(group_toks):
                    is_last = (i == len(group_toks) - 1)
                    token_labels.append({
                        "id":           gt["id"],
                        "text":         gt["text"],
                        "label":        "$KEEP" if is_last else "$WRONG_COMP",
                        "correct_text": ref_tok["text"] if is_last else "",
                    })
                continue

            entry   = alignment_map.get(hyp_id)
            ref_tok = entry["ref_token"] if entry else None

            if (ref_tok is not None and
                    hyp_tok.get("upos") in ("NOUN", "VERB") and
                    ref_tok.get("upos") == "NOUN" and
                    hyp_tok["text"][0:1].islower() and
                    ref_tok["text"][0:1].isupper() and
                    hyp_tok["text"].lower() == ref_tok["text"].lower() and
                    lemma_in_set(hyp_tok, verb_set)):
                token_labels.append({
                    "id":           hyp_id,
                    "text":         hyp_tok["text"],
                    "label":        "$WRONG_CAP",
                    "correct_text": hyp_tok["text"][0].upper() + hyp_tok["text"][1:],
                })
                continue

            if (ref_tok is not None and
                    hyp_tok.get("xpos") == "ADJA" and
                    lemmas_overlap(hyp_tok, ref_tok) and
                    hyp_tok["text"].lower() != ref_tok["text"].lower() and
                    is_declension_ending_difference(hyp_tok, ref_tok) and
                    det_context_matches(hyp_tok, ref_tok, hyp_id_to_tok, ref_id_to_tok)):
                token_labels.append({
                    "id":           hyp_id,
                    "text":         hyp_tok["text"],
                    "label":        "$WRONG_DECL",
                    "correct_text": ref_tok["text"],
                })
                continue

            token_labels.append({
                "id":           hyp_id,
                "text":         hyp_tok["text"],
                "label":        "$KEEP",
                "correct_text": hyp_tok["text"],
            })

        # ── build cor from token labels ───────
        err      = item["hypothesis"]["text"]
        cor      = item["hypothesis"]["text"]
        hyp_pmap = build_position_map(err, hyp_tokens)

        # convert token_labels to edit format for apply_edits
        tl_edits      = []
        applied_spans = set()
        for tl in sorted(token_labels,
                         key=lambda x: hyp_pmap.get(x["id"], (0, 0))[0],
                         reverse=True):
            if tl["id"] not in hyp_pmap:
                continue
            start, end = hyp_pmap[tl["id"]]
            if (start, end) in applied_spans:
                continue
            applied_spans.add((start, end))
            if tl["label"] == "$WRONG_COMP":
                del_start = start - 1 if start > 0 and cor[start - 1] == " " else start
                cor = cor[:del_start] + cor[end:]
            elif tl["correct_text"] != tl["text"]:
                cor = cor[:start] + tl["correct_text"] + cor[end:]

        has_error = False
        for tl in token_labels:
            if tl["label"] in detected_counts:
                detected_counts[tl["label"]] += 1
                has_error = True

        entry = {
            "sent_id":       str(sent_counter).zfill(7),
            "cor":           cor,
            "err":           err,
            "labeled":       " ".join(f"{tl['text']}{SEP}{tl['label']}"
                                      for tl in token_labels),
            "_token_labels": token_labels,
            "_hyp_tokens":   hyp_tokens,
            "_hyp_text":     err,
        }
        if has_error:
            detected_results.append(entry)
        else:
            clean_items.append(entry)
        sent_counter += 1

    print(f"\nDetection pass done:")
    print(f"  With errors : {len(detected_results)}")
    print(f"  Clean       : {len(clean_items)}")
    for k, v in detected_counts.items():
        print(f"  {k:<15}: {v}")

    # ── pass 2: injection ─────────────────────
    random.seed(RANDOM_SEED)

    all_cap_cands  = []
    all_decl_cands = []
    all_comp_cands = []

    for idx, entry in enumerate(clean_items):
        hyp_tokens = entry["_hyp_tokens"]
        hyp_pmap   = build_position_map(entry["_hyp_text"], hyp_tokens)
        for cand in collect_wrong_cap(hyp_tokens, hyp_pmap, verb_set):
            all_cap_cands.append((idx, cand))
        for cand in collect_wrong_decl(hyp_tokens, hyp_pmap):
            all_decl_cands.append((idx, cand))
        for cand in collect_wrong_comp(hyp_tokens, hyp_pmap):
            all_comp_cands.append((idx, cand))

    print(f"\nInjection candidates:")
    print(f"  CAP  : {len(all_cap_cands)}")
    print(f"  DECL : {len(all_decl_cands)}")
    print(f"  COMP : {len(all_comp_cands)}")

    def sample_needed(candidates, label):
        needed = TARGET_COUNTS[label] - detected_counts[label]
        if needed <= 0:
            print(f"  {label}: already at target")
            return []
        if len(candidates) < needed:
            print(f"  WARNING {label}: only {len(candidates)} candidates, need {needed}")
            return candidates
        return random.sample(candidates, needed)

    selected_cap  = sample_needed(all_cap_cands,  "$WRONG_CAP")
    selected_decl = sample_needed(all_decl_cands, "$WRONG_DECL")
    selected_comp = sample_needed(all_comp_cands, "$WRONG_COMP")

    inject_map = defaultdict(list)
    for idx, cand in selected_cap:
        inject_map[idx].append(("cap",  cand))
    for idx, cand in selected_decl:
        inject_map[idx].append(("decl", cand))
    for idx, cand in selected_comp:
        inject_map[idx].append(("comp", cand))

    for idx, entry in enumerate(clean_items):
        injections = inject_map.get(idx, [])

        if not injections:
            results.append({
                "sent_id": entry["sent_id"],
                "cor":     entry["cor"],
                "err":     entry["err"],
                "labeled": entry["labeled"],
            })
            continue

        hyp_tokens   = entry["_hyp_tokens"]
        hyp_text     = entry["_hyp_text"]
        hyp_pmap     = build_position_map(hyp_text, hyp_tokens)
        token_labels = entry["_token_labels"]

        all_edits = []
        for inj_type, cand in injections:
            if inj_type == "cap":
                start, end, replacement, tok_id, label = cand    # ← was 4, needs 5
                if tok_id not in hyp_pmap:
                    continue
                all_edits.append({
                    "tok_id":   tok_id, "start": start, "end": end,
                    "label":    "$WRONG_CAP",
                    "err_text": replacement,
                    "cor_text": hyp_text[start:end],  # original capitalized form
                })

            elif inj_type == "decl":
                start, end, replacement, tok_id, label = cand    # ← was 4, needs 5
                if tok_id not in hyp_pmap:
                    continue
                all_edits.append({
                    "tok_id":   tok_id, "start": start, "end": end,
                    "label":    "$WRONG_DECL",
                    "err_text": replacement,
                    "cor_text": hyp_text[start:end],
                })

            elif inj_type == "comp":
                start, end, replacement, tok_id, label, score = cand  # ← 6 elements
                if tok_id not in hyp_pmap:
                    continue
                head, part = replacement.split(" ", 1)
                all_edits.append({
                    "tok_id":   tok_id, "start": start, "end": end,
                    "label":    "$WRONG_COMP",
                    "err_text": replacement,
                    "cor_text": hyp_text[start:end],
                    "head":     head,
                    "part":     part,
                })

        # apply_edits expects (start, end, replacement, token_id, label)
        edits_for_apply = [
            (e["start"], e["end"], e["err_text"], e["tok_id"], e["label"])
            for e in all_edits
        ]
        err = apply_edits(hyp_text, edits_for_apply)
        cor = hyp_text  # clean sentences — cor unchanged

        # update token labels
        edit_map         = {ed["tok_id"]: ed for ed in all_edits}
        new_token_labels = []
        for tl in token_labels:
            tok_id = tl["id"]
            if tok_id not in edit_map:
                new_token_labels.append(tl)
                continue
            edit = edit_map[tok_id]
            if edit["label"] == "$WRONG_COMP":
                new_token_labels.append({
                    "id":           tok_id,
                    "text":         edit["head"],
                    "label":        "$WRONG_COMP",
                    "correct_text": edit["cor_text"],
                })
                new_token_labels.append({
                    "id":           tok_id,
                    "text":         edit["part"],
                    "label":        "$KEEP",
                    "correct_text": "",
                })
            else:
                new_token_labels.append({
                    "id":           tok_id,
                    "text":         edit["err_text"],
                    "label":        edit["label"],
                    "correct_text": edit["cor_text"],
                })

        # build_labeled expects tokens + edits in injection_utils format
        labeled = " ".join(
            f"{tl['text']}{SEP}{tl['label']}"
            for tl in new_token_labels
        )

        results.append({
            "sent_id": entry["sent_id"],
            "cor":     cor,
            "err":     err,
            "labeled": labeled,
        })

    # add detected
    results.extend({
        "sent_id": e["sent_id"],
        "cor":     e["cor"],
        "err":     e["err"],
        "labeled": e["labeled"],
    } for e in detected_results)

    # ── count labels ──────────────────────────
    label_counts = Counter()
    for entry in results:
        for token_label in entry["labeled"].split(" "):
            if f"{SEP}$" in token_label:
                label = "$" + token_label.split(f"{SEP}$")[-1]
                label_counts[label] += 1

    # ── save ──────────────────────────────────
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print_summary(results, dict(label_counts), len(data), str(output))


if __name__ == "__main__":
    main()