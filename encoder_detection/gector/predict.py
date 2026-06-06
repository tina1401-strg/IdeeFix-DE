import torch
import os
from tqdm import tqdm
from .modeling import GECToR
from transformers import PreTrainedTokenizer
from typing import List
from typing import List, Dict

KEEP_LABEL   = "$KEEP"
ERROR_LABELS = {"$WRONG_CAP", "$WRONG_DECL", "$WRONG_COMP"}

def get_word_masks_from_word_ids(
    word_ids: List[List[int]],
    n: int
):
    word_masks = []
    for i in range(n):
        previous_id = 0
        mask = []
        for _id in word_ids(i):
            if _id is None:
                mask.append(0)
            elif previous_id != _id:
                mask.append(1)
            else:
                mask.append(0)
            previous_id = _id
        word_masks.append(mask)
    return word_masks

def _predict(
    model: GECToR,
    tokenizer: PreTrainedTokenizer,
    srcs: List[str],
    keep_confidence: float=0,
    min_error_prob: float=0,
    batch_size: int=128
):
    itr = list(range(0, len(srcs), batch_size))
    pred_labels = []
    no_corrections = []
    no_correction_ids = [model.config.label2id[l] for l in ['$KEEP', '<OOV>', '<PAD>']]
    for i in itr:
        # The official models was trained without special tokens, e.g. [CLS] [SEP].
        batch = tokenizer(
            srcs[i:i+batch_size],
            return_tensors='pt',
            max_length=model.config.max_length,
            padding='max_length',
            truncation=True,
            is_split_into_words=True,
            add_special_tokens=True
        )
        batch['word_masks'] = torch.tensor(
            get_word_masks_from_word_ids(
                batch.word_ids,
                batch['input_ids'].size(0)
            )
        )
        word_ids = batch.word_ids
        batch = {k:v.to(model.device) for k,v in batch.items()}
        outputs = model.predict(
            batch['input_ids'],
            batch['attention_mask'],
            batch['word_masks'],
            keep_confidence,
            min_error_prob
        )
        # Align subword-level label to word-level label
        for i in range(len(outputs.pred_labels)):
            no_correct = True
            labels = []
            previous_word_idx = None
            for j, idx in enumerate(word_ids(i)):
                if idx is None:
                    continue
                if idx != previous_word_idx:
                    labels.append(outputs.pred_labels[i][j])
                    if outputs.pred_label_ids[i][j] not in no_correction_ids:
                        no_correct = False
                previous_word_idx = idx
            pred_labels.append(labels)
            no_corrections.append(no_correct)
    return pred_labels, no_corrections

def predict_detect_only(
    model,
    tokenizer,
    srcs: List[str],
    keep_confidence: float = 0.0,
    min_error_prob: float  = 0.0,
    batch_size: int        = 32,
) -> tuple[List[List[str]], List[Dict]]:

    srcs_with_start = [["$START"] + src.split() for src in srcs]

    pred_labels, no_corrections = _predict(
        model,
        tokenizer,
        srcs_with_start,
        keep_confidence,
        min_error_prob,
        batch_size,
    )

    label_sequences = []
    iteration_log   = []

    for i, (src, labels, no_corr) in enumerate(
        zip(srcs_with_start, pred_labels, no_corrections)
    ):
        tokens_no_start = src[1:]   # remove $START

        if no_corr:
            # model found no errors — all $KEEP
            seq = [KEEP_LABEL] * len(tokens_no_start)
            iteration_log.append([{"src": src, "tag": None}])
        else:
            # labels[0] corresponds to $START — skip it
            seq = labels[1:len(tokens_no_start) + 1]
            tags_with_start = labels[:len(src)]
            iteration_log.append([{"src": src, "tag": tags_with_start}])

        label_sequences.append(seq)

    return label_sequences, iteration_log
