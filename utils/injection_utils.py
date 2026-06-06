"""
utils/injection_utils.py
------------------------
Shared error injection utilities for IdeeFix-DE.
Used by: data_pipeline/04_inject_error_wiki_fineweb.py
         data_pipeline/03_inject_detect_error_asr.py
"""
from charsplit import Splitter
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from utils.constants import SEP
from utils.adj_error_intro import (
    detect_declension_type,
    get_adjective_ending,
    introduce_declension_error,
)

splitter = Splitter()

CONTRACTIONS = {
    ("von", "dem"): "vom",
    ("zu",  "dem"): "zum",
    ("zu",  "der"): "zur",
    ("an",  "dem"): "am",
    ("in",  "dem"): "im",
    ("bei", "dem"): "beim",
}


def parse_feats(feats_str: str) -> dict:
    if not feats_str:
        return {}
    return dict(pair.split("=") for pair in feats_str.split("|"))


def build_position_map(text: str, tokens: list) -> dict:
    """
    Maps token id → (start, end) in text.
    Handles MWT tokens (vom→von+dem) by sharing the surface span.
    """
    pos_map = {}
    pointer = 0
    i       = 0

    while i < len(tokens):
        token      = tokens[i]
        token_text = token["text"]

        while pointer < len(text) and text[pointer] == " ":
            pointer += 1

        if text[pointer:pointer + len(token_text)] == token_text:
            pos_map[token["id"]] = (pointer, pointer + len(token_text))
            pointer += len(token_text)
            i += 1
            continue

        found_mwt = False
        for lookahead in range(1, 4):
            if i + lookahead >= len(tokens):
                break
            group       = tokens[i:i + lookahead + 1]
            group_texts = tuple(t["text"].lower() for t in group)
            surface     = CONTRACTIONS.get(group_texts)
            if surface and text[pointer:pointer + len(surface)].lower() == surface:
                start = pointer
                end   = pointer + len(surface)
                for t in group:
                    pos_map[t["id"]] = (start, end)
                pointer    = end
                i         += lookahead + 1
                found_mwt  = True
                break

        if not found_mwt:
            idx = text.find(token_text, pointer)
            if idx != -1:
                pos_map[token["id"]] = (idx, idx + len(token_text))
                pointer = idx + len(token_text)
            i += 1

    return pos_map


def collect_wrong_cap(tokens: list, pos_map: dict, verb_set: set) -> list:
    """
    Returns edits: [(start, end, replacement, token_id, label), ...]
    """
    edits = []
    for token in tokens:
        if token.get("upos") != "NOUN":
            continue
        if token["id"] not in pos_map:
            continue
        if not token["text"][0].isupper():
            continue
        text_lower = token["text"].lower()
        if text_lower not in verb_set:
            continue
        start, end = pos_map[token["id"]]
        edits.append((start, end, text_lower, token["id"], "$WRONG_CAP"))
    return edits


def collect_wrong_decl(tokens: list, pos_map: dict) -> list:
    """
    Returns edits: [(start, end, replacement, token_id, label), ...]
    """
    edits = []
    for i, token in enumerate(tokens):
        if token.get("xpos") != "ADJA":
            continue
        if token["id"] not in pos_map:
            continue
        feats      = parse_feats(token.get("feats", ""))
        case       = feats.get("Case")
        gender     = feats.get("Gender")
        number     = feats.get("Number")
        if not case or (not gender and number != "Plur"):
            continue
        gender_key     = "Plur" if number == "Plur" else gender
        preceding_text = tokens[i - 1]["text"] if i > 0 else ""
        decl_type      = detect_declension_type(preceding_text)
        correct_ending = get_adjective_ending(decl_type, gender_key, case)
        lemma          = token.get("lemma", "")
        if token["text"].lower() != (lemma + correct_ending).lower():
            continue
        wrong_form = introduce_declension_error(lemma, correct_ending, decl_type, gender_key, case)
        if wrong_form.lower() == token["text"].lower():
            continue
        start, end = pos_map[token["id"]]
        edits.append((start, end, wrong_form, token["id"], "$WRONG_DECL"))
    return edits


def collect_wrong_comp(tokens: list, pos_map: dict) -> list:
    """
    Returns edits: [(start, end, replacement, token_id, label, score), ...]
    Only the highest-confidence candidate per call.
    """
    candidates = []
    for token in tokens:
        if token["id"] not in pos_map:
            continue
        if token.get("upos") != "NOUN":
            continue
        word = token["text"]
        if len(word) < 6 or "-" in word:
            continue
        splits = splitter.split_compound(word)
        if not splits:
            continue
        best  = splits[0]
        score = best[0]
        parts = best[1:]
        if score < 0.85 or len(parts) != 2:
            continue
        head = parts[0]
        part = parts[1].lower()
        if head.lower() == word.lower() or part.lower() == word.lower():
            continue
        start, end = pos_map[token["id"]]
        candidates.append((start, end, head + " " + part, token["id"], "$WRONG_COMP", score))

    if not candidates:
        return []
    best = max(candidates, key=lambda x: x[5])
    return [best]


def apply_edits(text: str, edits: list) -> str:
    for start, end, replacement, token_id, label, *_ in sorted(
        edits, key=lambda x: x[0], reverse=True
    ):
        text = text[:start] + replacement + text[end:]
    return text


def build_labeled(tokens: list, edits: list, pos_map: dict) -> str:
    edit_map = {
        token_id: (replacement, label)
        for start, end, replacement, token_id, label, *_ in edits
    }
    parts = []
    for token in tokens:
        if token["id"] not in pos_map:
            continue
        token_id = token["id"]
        if token_id in edit_map:
            replacement, label = edit_map[token_id]
            if label == "$WRONG_COMP":
                word1, word2 = replacement.split(" ", 1)
                parts.append(f"{word1}{SEP}{label}")
                parts.append(f"{word2}{SEP}$KEEP")
            else:
                parts.append(f"{replacement}{SEP}{label}")
        else:
            parts.append(f"{token['text']}{SEP}$KEEP")
    return " ".join(parts)


def print_summary(results: list, label_counts: dict, data_len: int, output: str):
    corr = sum(1 for e in results if e["cor"] == e["err"])
    err  = sum(1 for e in results if e["cor"] != e["err"])
    print("\n" + "=" * 50)
    print("DONE")
    print(f"  Total input   : {data_len}")
    print(f"  Total output  : {len(results)}")
    print(f"  Correct       : {corr}")
    print(f"  With errors   : {err}")
    print(f"  $KEEP         : {label_counts.get('$KEEP', 0)}")
    print(f"  $WRONG_CAP    : {label_counts.get('$WRONG_CAP', 0)}")
    print(f"  $WRONG_DECL   : {label_counts.get('$WRONG_DECL', 0)}")
    print(f"  $WRONG_COMP   : {label_counts.get('$WRONG_COMP', 0)}")
    print(f"  Output file   : {output}")
    print("=" * 50)