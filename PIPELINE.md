# IdeeFix-DE — Pipeline

Step-by-step instructions to reproduce the full data pipeline, model training, and evaluation.

---

IdeeFix-DE/
├── utils/                  -> Shared utilities (stanza, constants, injection)
├── resources/              -> german_verbs.txt
├── data_pipeline/          -> All scripts for data creation and analysis
├── encoder_detection/      -> Adapted GECToR implementation (training + prediction)
├── llm_correction/         -> LLM evaluation and prompting scripts
├── data/                   -> .gitignore (raw, anno, processed, splits)
├── results/                -> .gitignore (model outputs, predictions)
├── predict.sh              -> Infer the whole hybrid pipeline
├── PIPELINE.md             -> Step-by-step reproduction guide
└── README.md               -> Project overview and setup

## Environment Setup

### ASR Environment (conda)
Required for step 00 only.

```yaml
name: asr_eval
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - ffmpeg
  - pip
  - pip:
      - torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
      - torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
      - openai-whisper
      - soundfile
      - tqdm
      - datasets
```

```bash
conda env create -f asr_eval.yml
conda activate asr_eval
```

### GECToR Environment (uv)
Required for encoder training only.

```bash
cd IdeeFix-DE/encoder_detection
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

### Main Environment (uv)
Required for all other steps.

```bash
cd IdeeFix-DE
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```
---

## Step 00 — ASR Data Creation
> Environment: `asr_eval` conda

```bash
cd data_pipeline/00_asr_data_creation

# TTS data — add telephony noise + gaussian noise before transcription
python 00.1_download_transcribe_asr.py \
    --data_dir ../../data/raw \
    --split    train_el_tts \
    --snr_db   20 \
    --temperature 0.8

# TUDA data
python 00.1_download_transcribe_asr.py \
    --data_dir ../../data/raw \
    --split    train_tuda_0

# EuroSpeech data
python 00.1_download_transcribe_asr.py \
    --data_dir ../../data/raw \
    --split    train_eurospeech
```

Output:

data/raw/manifest_train_el_tts.csv
data/raw/manifest_train_tuda_0.csv
data/raw/manifest_train_eurospeech.csv

---

## Step 01 — Download Wiki/FineWeb Sentences
> Environment: uv

```bash
cd data_pipeline

uv run 01_download_wiki_fineweb.py \
    --output ../data/raw/wiki_fineweb_sent.json
```

---

## Step 02 — Parse and Annotate Data
> Environment: uv

```bash
# wiki/fineweb
uv run 02_parse_data.py \
    --mode   wiki \
    --input  ../data/raw/wiki_fineweb_sent.json \
    --output ../data/anno/wiki_fineweb_annotated.json

# ASR — EuroSpeech
uv run 02_parse_data.py \
    --mode   asr \
    --input  ../data/raw/manifest_train_eurospeech.csv \
    --output ../data/anno/manifest_eurospeech_annotated.json

# ASR — TTS
uv run 02_parse_data.py \
    --mode   asr \
    --input  ../data/raw/manifest_train_el_tts.csv \
    --output ../data/anno/manifest_train_el_tts_annotated.json

# ASR — TUDA
uv run 02_parse_data.py \
    --mode   asr \
    --input  ../data/raw/manifest_train_tuda_0.csv \
    --output ../data/anno/manifest_train_tuda_0_annotated.json
```

---

## Step 03 — Scrape German Verb List
> Environment: uv

```bash
uv run 03_scrape_verbs.py --output_dir ../resources
```

Output: `resources/german_verbs.txt`

---

## Step 04 — Inject Errors into Wiki/FineWeb
> Environment: uv

```bash
uv run 04_inject_errors_wiki_fineweb.py \
    --input ../data/anno/wiki_fineweb_annotated.json \
    --output      ../data/processed/wiki_fineweb_injected.json
```

---

## Step 05 — Detect and Inject Errors in ASR Data
> Environment: uv

```bash
uv run 05_detect_inject_asr_errors.py \
    --input_paths ../data/anno/manifest_eurospeech_annotated.json \
                  ../data/anno/manifest_train_el_tts_annotated.json \
                  ../data/anno/manifest_train_tuda_0_annotated.json \
    --output      ../data/processed/asr_detected_injected.json
```

---

## Step 06 — Split Data
> Environment: uv

```bash
uv run 06_split_data.py \
    --whisper_path ../data/processed/asr_detected_injected.json \
    --wiki_path    ../data/processed/wiki_fineweb_injected.json \
    --output_dir   ../data/splits
```

Output:

data/splits/
├── gector/
│   ├── train.json / train.txt
│   ├── dev.json   / dev.txt
│   ├── test_gector.json / test_gector.txt
│   ├── test_plain.txt
│   └── test_gold.txt
└── llm/
├── eval_llm.json
└── test_llm.json

---

## Step 07 — EDA
> Environment: uv

```bash
uv run 07_eda_train.py \
    --input      ../data/splits/encoder_detection/train.json \
    --output_dir ../data/splits/encoder_detection/eda
```

---

## Step 08 — Train GECToR Encoder
> Environment: internal uv environment with environment_gector.toml

```bash
.\train_gector.sh
```

## Step 9 - Test GECTor Encoder (and run for hybrid test)

```bash
.\predict_gector.sh
```

## Step 10 - Evaluate GECTor Encoder Test
 
```bash

# for encoder test

uv run evaluate.py \
    --gold ./data/splits/encoder_detection/test_gector.json \
    --pred ./results/gbert-large/best/predictions_encoder.txt \
    --out  ./results/gbert-large/best/predictions_encoder.json

# for hybrid test

uv run evaluate.py \
    --gold ../data/splits/llm_correction/test_llm.json \
    --pred ./results/gbert-large/best/predictions_hybrid.txt \
    --out  ./results/gbert-large/best/predictions_hybrid.json

```

## Step 09 — Run LLM Correction
> Environment: uv

```bash
cd llm_correction/scripts

# simple run sh for all three tests (no_label, hybrid_label, gold_label)

cd ..

./run_llm.sh
```
---

## Step 10 — Evaluate LLM Results
> Environment: uv

```bash
uv run eval_results.py --results ../results/gold/gemma_labeled_conservative_labeled_6_results.json
uv run eval_results.py --results ../results/hybrid/gemma_labeled_conservative_labeled_6_results.json
uv run eval_results.py --results ../results/unlabeled/gemma_unlabeled_conservative_unlabeled_6_results.json
```
---
## Step 11- Create comparison.csv for LLM Results

```bash
uv run comp_evaluation.py 
```
