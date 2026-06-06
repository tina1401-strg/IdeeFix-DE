"""
utils/constants.py
------------------
Shared constants for the IdeeFix-DE pipeline.
"""

import subprocess

def get_best_gpu(min_free_mb: int = 10000) -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    best_gpu  = None
    best_free = 0
    for line in result.stdout.strip().split("\n"):
        idx, free, used, total = [x.strip() for x in line.split(",")]
        free_mb = int(free)
        print(f"GPU {idx}: {free_mb} MB free / {total} MB total")
        if free_mb > best_free and free_mb >= min_free_mb:
            best_free = free_mb
            best_gpu  = int(idx)
    if best_gpu is None:
        raise RuntimeError(f"No GPU with at least {min_free_mb} MB free found!")
    print(f"→ Selected GPU {best_gpu} ({best_free} MB free)\n")
    return best_gpu


VERB_LIST_PATH = "/home/mlt_ml1/german_gec_project/data/verbs/german_verbs.txt"

RANDOM_SEED = 42

# ─────────────────────────────────────────────
# LABELED STRING SEPARATOR
# ─────────────────────────────────────────────

SEP = "SEPL|||SEPR"

# ─────────────────────────────────────────────
# COMPOUND DETECTION
# ─────────────────────────────────────────────

COMPOUND_BLOCKLIST = {
    "zufolge", "aufgrund", "anhand", "infolge", "anstelle",
    "mithilfe", "zuliebe", "zugunsten", "zulasten", "inmitten",
    "angesichts", "stattdessen", "demzufolge", "diesbezüglich",
    "aufseiten", "zuhilfe", "woraufhin", "wobei", "wodurch",
    "anstatt", "ausserhalb", "außerhalb", "innerhalb", "oberhalb",
    "unterhalb", "diesseits", "jenseits", "anlässlich", "bezüglich",
    "hinsichtlich",
}

STOPWORDS = {
    "der", "die", "das", "des", "dem", "den",
    "ein", "eine", "einer", "einem", "einen", "eines",
    "und", "oder", "aber", "doch", "wie", "als", "ob",
    "zu", "an", "in", "auf", "mit", "von", "bei", "nach",
    "aus", "um", "für", "über", "unter", "vor", "hinter",
    "es", "er", "sie", "wir", "ich", "du", "ihr",
    "ist", "war", "hat", "wird", "sein", "haben",
    "nicht", "auch", "noch", "nur", "schon", "mehr",
    "so", "da", "dann", "nun", "ja", "nein", "wo", "was",
}

# ─────────────────────────────────────────────
# GERMAN CONTRACTIONS
# ─────────────────────────────────────────────

CONTRACTIONS_FORWARD = {
    "im":   ("in",  "dem"),
    "vom":  ("von", "dem"),
    "zum":  ("zu",  "dem"),
    "zur":  ("zu",  "der"),
    "am":   ("an",  "dem"),
    "beim": ("bei", "dem"),
}

CONTRACTIONS_REVERSE = {v: k for k, v in CONTRACTIONS_FORWARD.items()}


# ─────────────────────────────────────────────
# VERB LISt Scrape
# ─────────────────────────────────────────────

WIKTIONARY_API_URL = "https://de.wiktionary.org/w/api.php"
WIKTIONARY_CATEGORY = "Kategorie:Verb_(Deutsch)"
WIKTIONARY_USER_AGENT = "IdeeFix-DE/1.0 (student NLP project)"