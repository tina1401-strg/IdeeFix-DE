#!/bin/bash
# predict_gector.sh
# -----------------
# Prediction script for GECToR with gbert-large.
# Detection only — n_iteration=1, outputs labels per token.
#
# Usage:
#   conda activate gector
#   bash predict_gector.sh


python ./predict.py \
    #--input         ../data/splits/llm_correction/test_plain_tok.txt \ # for hybrid test
    #--out           ./results/gbert-large/best/predictions_hybrid.txt \    # for hybrid test
    #--visualize     ./results/gbert-large/best/visualization_hybrid.txt \  # for hybrid test
    --input         ../data/splits/encoder_detection/test_plain_tok.txt \  # for encoder test
    --restore_dir   ./models/gbert-large/best \
    --out           ./results/gbert-large/best/predictions_encoder.txt \    # for encoder test
    --visualize     ./results/gbert-large/best/visualization_encoder.txt \  # for encoder test
    --batch_size    64 \
    --keep_confidence 0.0 \
    --min_error_prob  0.0

