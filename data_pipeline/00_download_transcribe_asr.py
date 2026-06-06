"""
00.1_download_transcribe_asr.py
--------------------------------
Downloads German ASR data from flozi00/german-asr-mixed-whisper,
optionally applies audio degradation (TTS split), transcribes with
Whisper, and saves only the manifest CSV — no WAV files saved.

Supported splits:
    train_el_tts      (~495 rows)  — TTS data, noise added before transcription
    train_tuda_0      (~42k rows)  — TUDA data, deduplicated by transcript
    train_eurospeech  (~50k rows)  — EuroSpeech data, deduplicated by transcript

Outputs (all under --data_dir/<split_name>/):
    {split}_manifest.csv     ← id, reference, hypothesis

Usage:
    python 00.1_download_transcribe_asr.py --data_dir ./data/raw --split train_eurospeech
    python 00.1_download_transcribe_asr.py --data_dir ./data/raw --split train_tuda_0
    python 00.1_download_transcribe_asr.py --data_dir ./data/raw --split train_el_tts --snr_db 20 --temperature 0.8
"""

import csv
import os
import random
import argparse
import subprocess
import numpy as np
from pathlib import Path

import torch
import torchaudio.transforms as T
import torchaudio.functional as F
from datasets import load_dataset
from tqdm import tqdm
from utils.constants import get_best_gpu

# ─────────────────────────────────────────────
# GPU SELECTION
# ─────────────────────────────────────────────


os.environ["CUDA_VISIBLE_DEVICES"] = str(get_best_gpu())

import whisper

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

HF_DATASET         = "flozi00/german-asr-mixed-whisper"
TARGET_SAMPLE_RATE = 16_000
RANDOM_SEED        = 42

TELEPHONY_LOW_HZ   = 300
TELEPHONY_HIGH_HZ  = 3400

SPLIT_DEFAULTS = {
    "train_el_tts":     {"max_samples": None,  "deduplicate": False, "add_noise": True},
    "train_tuda_0":     {"max_samples": 25000, "deduplicate": True,  "add_noise": False},
    "train_eurospeech": {"max_samples": 22500, "deduplicate": True,  "add_noise": False},
}

# ─────────────────────────────────────────────
# AUDIO HELPERS
# ─────────────────────────────────────────────

def resample(array: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == TARGET_SAMPLE_RATE:
        return array
    waveform  = torch.tensor(array).unsqueeze(0).float()
    resampler = T.Resample(orig_freq=orig_sr, new_freq=TARGET_SAMPLE_RATE)
    return resampler(waveform).squeeze(0).numpy()


def apply_telephony_filter(array: np.ndarray) -> np.ndarray:
    waveform = torch.tensor(array).unsqueeze(0)
    waveform = F.highpass_biquad(waveform, TARGET_SAMPLE_RATE, TELEPHONY_LOW_HZ)
    waveform = F.lowpass_biquad(waveform, TARGET_SAMPLE_RATE, TELEPHONY_HIGH_HZ)
    return waveform.squeeze(0).numpy()


def add_gaussian_noise(array: np.ndarray, snr_db: float) -> np.ndarray:
    signal_power = np.mean(array ** 2)
    if signal_power == 0:
        return array
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise       = np.random.normal(0, np.sqrt(noise_power), array.shape)
    return np.clip(array + noise, -1.0, 1.0).astype(np.float32)


def degrade(array: np.ndarray, snr_db: float) -> np.ndarray:
    array = apply_telephony_filter(array)
    array = add_gaussian_noise(array, snr_db)
    return array


# ─────────────────────────────────────────────
# MANIFEST HELPER
# ─────────────────────────────────────────────

def write_manifest(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run(
    split_name:    str,
    data_dir:     Path,
    manifest_path: Path,
    max_samples:   int | None,
    deduplicate:   bool,
    add_noise:     bool,
    snr_db:        float,
    whisper_model: whisper.Whisper,
    whisper_args:  dict,
) -> list[dict]:

    data_dir.mkdir(parents=True, exist_ok=True)

    # ── resume support ────────────────────────
    manifest_rows    = []
    seen_transcripts = set()
    sample_counter   = 0

    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                manifest_rows.append(row)
                seen_transcripts.add(row["reference"])
                sample_counter += 1
        print(f"  Resuming — {sample_counter} samples already done")

    still_needed = (max_samples or 999_999) - sample_counter
    if still_needed <= 0:
        print("  Already have enough samples!")
        return manifest_rows

    print(f"  Streaming {split_name} — need {still_needed} more samples...")
    if add_noise:
        print(f"  Noise: telephony filter + SNR={snr_db}dB")

    ds      = load_dataset(HF_DATASET, split=split_name, streaming=True)
    skipped = 0

    with tqdm(total=still_needed, desc=f"  {split_name}", unit="samples") as pbar:
        for sample in ds:
            if sample_counter >= (max_samples or 999_999):
                break

            reference = sample.get("transkription", "").strip()
            if not reference:
                skipped += 1
                continue
            if deduplicate and reference in seen_transcripts:
                skipped += 1
                continue

            seen_transcripts.add(reference)

            # ── process audio ─────────────────
            audio   = sample["audio"]
            array   = np.array(audio["array"], dtype=np.float32)
            orig_sr = audio["sampling_rate"]

            try:
                array = resample(array, orig_sr)
                if add_noise:
                    array = degrade(array, snr_db)
            except Exception as e:
                print(f"\n  [error] processing sample {sample_counter}: {e}")
                skipped += 1
                continue

            # ── transcribe directly from array ─
            try:
                result     = whisper_model.transcribe(array, **whisper_args)
                hypothesis = result["text"].strip()
            except Exception as e:
                print(f"\n  [error] transcribing sample {sample_counter}: {e}")
                skipped += 1
                continue

            sample_id      = f"{split_name}_{sample_counter:05d}"
            sample_counter += 1

            row = {
                "id":        sample_id,
                "reference": reference,
                "hypothesis": hypothesis,
            }
            if add_noise:
                row["snr_db"] = snr_db

            manifest_rows.append(row)
            pbar.update(1)

            # checkpoint every 500 samples
            if len(manifest_rows) % 500 == 0:
                write_manifest(manifest_path, manifest_rows)

    write_manifest(manifest_path, manifest_rows)
    return manifest_rows


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    required=True,
                        help="Root directory for outputs (e.g. ./data/raw)")
    parser.add_argument("--split",       required=True,
                        choices=list(SPLIT_DEFAULTS.keys()))
    parser.add_argument("--max_samples", type=int,   default=None)
    parser.add_argument("--snr_db",      type=float, default=20.0,
                        help="SNR in dB for TTS noise (default: 20.0)")
    parser.add_argument("--model",       type=str,   default="small",
                        choices=["tiny", "base", "small", "medium", "large-v3"])
    parser.add_argument("--beam_size",   type=int,   default=5)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Whisper decoding temperature (default: 0.0, use 0.8 for TTS)")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    split_name  = args.split
    defaults    = SPLIT_DEFAULTS[split_name]
    data_dir    = Path(args.data_dir)
    split_dir     = data_dir
    manifest_path = data_dir / f"manifest_{split_name}.csv"

    max_samples = args.max_samples or defaults["max_samples"]
    deduplicate = defaults["deduplicate"]
    add_noise   = defaults["add_noise"]

    whisper_args = {
        "beam_size":       args.beam_size,
        "temperature":     args.temperature,
        "word_timestamps": False,
    }

    print(f"\n{'='*50}")
    print(f"Split       : {split_name}")
    print(f"Output dir  : {split_dir}")
    print(f"Manifest    : {manifest_path}")
    print(f"Max samples : {max_samples or 'all'}")
    print(f"Deduplicate : {deduplicate}")
    print(f"Add noise   : {add_noise}" + (f" (SNR={args.snr_db}dB)" if add_noise else ""))
    print(f"Whisper     : {args.model} | beam={args.beam_size} | temp={args.temperature}")
    print(f"{'='*50}\n")

    print(f"Loading Whisper model: {args.model}...")
    model  = whisper.load_model(args.model)
    device = next(model.parameters()).device
    print(f"Model loaded on: {device}\n")

    rows = run(
        split_name    = split_name,
        split_dir     = split_dir,
        manifest_path = manifest_path,
        max_samples   = max_samples,
        deduplicate   = deduplicate,
        add_noise     = add_noise,
        snr_db        = args.snr_db,
        whisper_model = model,
        whisper_args  = whisper_args,
    )

    print(f"\n{'='*50}")
    print(f"DONE")
    print(f"  Samples   : {len(rows):,}")
    print(f"  Manifest  : {manifest_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()