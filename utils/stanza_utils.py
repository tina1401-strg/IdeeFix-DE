import stanza

# ─────────────────────────────────────────────
# PIPELINES
# ─────────────────────────────────────────────

def load_stanza_sentence_tokenizer(use_gpu: bool = False) -> stanza.Pipeline:
    return stanza.Pipeline(
        "de",
        processors="tokenize",
        verbose=False,
        use_gpu=use_gpu,
    )


def load_stanza_word_tokenizer(use_gpu: bool = False) -> stanza.Pipeline:
    return stanza.Pipeline(
        "de",
        processors="tokenize",
        processor_dict={"tokenize": "gsd"},
        verbose=False,
        use_gpu=use_gpu,
    )


def load_stanza_full(use_gpu: bool = True) -> stanza.Pipeline:
    return stanza.Pipeline(
        "de",
        processors={
            "tokenize": "gsd",
            "pos":      "hdt",
            "lemma":    "default",
            "depparse": "hdt",
        },
        package=None,
        use_gpu=use_gpu,
        verbose=False,
    )


def annotate_sentence(nlp: stanza.Pipeline, text: str) -> list[dict]:
    if not text.strip():
        return []
    doc    = nlp(text)
    tokens = []
    for sent in doc.sentences:
        for word in sent.words:
            tokens.append({
                "id":     word.id,
                "text":   word.text,
                "lemma":  word.lemma,
                "upos":   word.upos,
                "xpos":   word.xpos,
                "feats":  word.feats,
                "head":   word.head,
                "deprel": word.deprel,
            })
    return tokens

# ─────────────────────────────────────────────
# TOKENIZATION FUNCTIONS
# ─────────────────────────────────────────────

def sentence_split(nlp: stanza.Pipeline, text: str) -> list[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sentences if sent.text.strip()]


def word_tokenize(nlp: stanza.Pipeline, text: str) -> list[dict]:
    doc    = nlp(text)
    tokens = []
    for sent in doc.sentences:
        for word in sent.words:
            tokens.append({
                "id":   word.id,
                "text": word.text,
                "upos": word.upos or "X",
            })
    return tokens