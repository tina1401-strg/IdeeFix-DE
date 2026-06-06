#!/bin/bash
# run_llm.sh
# ----------

cd "$(dirname "$0")/scripts" || exit 1

MODEL="gemma"
OUTPUT_DIR="../results/unlabeled"
PROMPT="../prompts/unlabeled_conservative.txt"
FEW_SHOTS="../few_shots/unlabeled_6.json"
INPUT="../../data/splits/llm_correction/eval_llm.json"
WITH_LABELS=""

echo "=================================="
echo "Model      : $MODEL"
echo "Prompt     : $PROMPT"
echo "Few-shots  : $FEW_SHOTS"
echo "Input      : $INPUT"
echo "Labels     : ${WITH_LABELS:-none}"
echo "Output dir : $OUTPUT_DIR"
echo "=================================="

uv run run_llm.py \
    --model      "$MODEL" \
    --prompt     "$PROMPT" \
    --few_shots  "$FEW_SHOTS" \
    --input      "$INPUT" \
    --output_dir "$OUTPUT_DIR" \
    $WITH_LABELS

OUTPUT_DIR="../results/final/gold"
PROMPT="../prompts/labeled_conservative.txt"
FEW_SHOTS="../few_shots/labeled_6.json"
INPUT="../../data/splits/llm_correction/eval_llm.json"
WITH_LABELS="--with_labels"

echo "=================================="
echo "Model      : $MODEL"
echo "Prompt     : $PROMPT"
echo "Few-shots  : $FEW_SHOTS"
echo "Input      : $INPUT"
echo "Labels     : ${WITH_LABELS:-none}"
echo "Output dir : $OUTPUT_DIR"
echo "=================================="

uv run run_llm.py \
    --model      "$MODEL" \
    --prompt     "$PROMPT" \
    --few_shots  "$FEW_SHOTS" \
    --input      "$INPUT" \
    --output_dir "$OUTPUT_DIR" \
    $WITH_LABELS

OUTPUT_DIR="../results/final/hybrid"
PROMPT="../prompts/labeled_conservative.txt"
FEW_SHOTS="../few_shots/labeled_6.json"
INPUT="/home/mlt_ml1/IdeeFix-DE/encoder_detection/results/gbert-large/best/predictions_hybrid.json" 
WITH_LABELS="--with_labels"

echo "=================================="
echo "Model      : $MODEL"
echo "Prompt     : $PROMPT"
echo "Few-shots  : $FEW_SHOTS"
echo "Input      : $INPUT"
echo "Labels     : ${WITH_LABELS:-none}"
echo "Output dir : $OUTPUT_DIR"
echo "=================================="

uv run run_llm.py \
    --model      "$MODEL" \
    --prompt     "$PROMPT" \
    --few_shots  "$FEW_SHOTS" \
    --input      "$INPUT" \
    --output_dir "$OUTPUT_DIR" \
    $WITH_LABELS