ADJECTIVE_ENDINGS = {
    "strong": {
        "Masc": {"Nom": "er", "Acc": "en", "Dat": "em", "Gen": "en"},
        "Fem":  {"Nom": "e",  "Acc": "e",  "Dat": "er", "Gen": "er"},
        "Neut": {"Nom": "es", "Acc": "es", "Dat": "em", "Gen": "en"},
        "Plur": {"Nom": "e",  "Acc": "e",  "Dat": "en", "Gen": "er"},
    },
    "weak": {
        "Masc": {"Nom": "e",  "Acc": "en", "Dat": "en", "Gen": "en"},
        "Fem":  {"Nom": "e",  "Acc": "e",  "Dat": "en", "Gen": "en"},
        "Neut": {"Nom": "e",  "Acc": "e",  "Dat": "en", "Gen": "en"},
        "Plur": {"Nom": "en", "Acc": "en", "Dat": "en", "Gen": "en"},
    },
    "mixed": {
        "Masc": {"Nom": "er", "Acc": "en", "Dat": "en", "Gen": "en"},
        "Fem":  {"Nom": "e",  "Acc": "e",  "Dat": "en", "Gen": "en"},
        "Neut": {"Nom": "es", "Acc": "es", "Dat": "en", "Gen": "en"},
        "Plur": {"Nom": "en", "Acc": "en", "Dat": "en", "Gen": "en"},
    },
}

ALL_VALID_ENDINGS = {
    ending
    for paradigm in ADJECTIVE_ENDINGS.values()
    for gender in paradigm.values()
    for ending in gender.values()
}

# definite articles to detect weak declension
DEFINITE_ARTICLES = {"der", "die", "das", "des", "dem", "den"}

# indefinite articles to detect mixed declension
INDEFINITE_ARTICLES = {"ein", "eine", "einer", "einem", "einen", "eines",
                        "kein", "keine", "keiner", "keinem", "keinen", "keines"}

def get_adjective_ending(declension_type, gender, case):
    """
    declension_type: 'strong', 'weak', 'mixed'
    gender: 'Masc', 'Fem', 'Neut', 'Plur'
    case: 'Nom', 'Acc', 'Dat', 'Gen'
    """
    return ADJECTIVE_ENDINGS[declension_type][gender][case]


def apply_adjective_ending(lemma, ending):
    """
    Apply ending to adjective lemma.
    e.g. lemma='kalt', ending='en' -> 'kalten'
    """
    return lemma + ending


def detect_declension_type(preceding_token):
    """
    Detect declension type based on preceding token.
    """
    token_lower = preceding_token.lower()
    if token_lower in DEFINITE_ARTICLES:
        return "weak"
    elif token_lower in INDEFINITE_ARTICLES:
        return "mixed"
    else:
        return "strong"

import random

def introduce_declension_error(lemma, correct_ending, declension_type, gender, case):
    """
    Replace correct ending with a random wrong one from the same declension type.
    """
    all_endings = [
        ADJECTIVE_ENDINGS[declension_type][g][c]
        for g in ADJECTIVE_ENDINGS[declension_type]
        for c in ADJECTIVE_ENDINGS[declension_type][g]
        if not (g == gender and c == case)  # exclude correct one
    ]
    wrong_ending = random.choice(list(set(all_endings)))  # pick a different one
    return lemma + wrong_ending