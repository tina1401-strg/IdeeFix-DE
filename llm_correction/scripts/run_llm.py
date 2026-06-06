"""
eval/run_llm.py
---------------
Evaluates open-weight LLMs on German ASR GEC.

Usage:
    python run_llm.py \
        --model       qwen \
        --prompt      ../prompts/system_prompt_no_labels.txt \
        --few_shots   ../few_shots/few_shots_no_labels.json \
        --input       ../../data/splits/llm/eval_llm.json \
        --output_dir  ./results

    python run_llm.py \
        --model       qwen \
        --prompt      ../prompts/system_prompt_labeled.txt \
        --few_shots   ../few_shots/few_shots_labeled.json \
        --with_labels \
        --input       ../../data/splits/llm/eval_llm.json \
        --output_dir  ./results
"""

import json
import argparse
from pathlib import Path
import os
import subprocess
import sys
import gc

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from utils.constants import SEP, get_best_gpu


def kill_my_gpu_processes(target_gpu: int):
    """
    Kill any python processes from current user running on target GPU.
    Does NOT pkill all — only kills processes occupying GPU memory.
    """
    current_pid = os.getpid()
    current_user = os.environ.get("USER", "")

    try:
        result = subprocess.run(
            ["fuser", f"/dev/nvidia{target_gpu}"],
            capture_output=True, text=True
        )
        pids = result.stdout.strip().split()
        killed = []
        for pid in pids:
            pid = pid.strip()
            if not pid.isdigit():
                continue
            if int(pid) == current_pid:
                continue
            # check if it belongs to current user
            try:
                proc_user = subprocess.run(
                    ["ps", "-o", "user=", "-p", pid],
                    capture_output=True, text=True
                ).stdout.strip()
                if proc_user == current_user:
                    os.kill(int(pid), 9)
                    killed.append(pid)
            except Exception:
                continue
        if killed:
            print(f"  Killed my processes on GPU {target_gpu}: {killed}")
        else:
            print(f"  No stale processes found on GPU {target_gpu}")
    except Exception as e:
        print(f"  Could not check GPU processes: {e}")


# ── select best GPU and clean it ─────────────
best_gpu = get_best_gpu()
print(f"Cleaning GPU {best_gpu} before loading...")
kill_my_gpu_processes(best_gpu)

os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
DEVICE = "cuda:0"

import torch

# clear any remaining cache
torch.cuda.empty_cache()
gc.collect()
print(f"GPU {best_gpu} ready. "
      f"Free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB / "
      f"{torch.cuda.mem_get_info()[1]/1e9:.1f} GB total\n")

from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
    "qwen":    "Qwen/Qwen2.5-7B-Instruct",
    "qwen_3":  "Qwen/Qwen3.5-9B",
    "gemma":   "google/gemma-3-12b-it",
    "gemma_4": "google/gemma-4-12B-it",
    "leollm":  "LeoLM/leo-mistral-hessianai-7b-chat",
    "eurollm": "utter-project/EuroLLM-9B-Instruct",
    "teuken":  "openGPT-X/Teuken-7B-instruct-commercial-v0.4",
    "discolm": "DiscoResearch/DiscoLM_German_7b_v1",
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_few_shots(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Few-shots file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def labeled_to_prompt_format(labeled_str: str) -> str:
    """
    Convert labeled string to prompt format.
    Handles both:
        'tokenSEPL|||SEPR$LABEL' → 'token$LABEL'
        'token$LABEL'            → 'token$LABEL' (already correct)
    """
    if SEP not in labeled_str:
        return labeled_str.strip()
    parts = []
    for chunk in labeled_str.strip().split(" "):
        if SEP in chunk:
            text, label = chunk.split(SEP, 1)
            parts.append(f"{text}{label}")
    return " ".join(parts)


def get_labeled_str(item: dict, with_labels: bool) -> str | None:
    """
    Get labeled string from item, handling both formats:
    - Standard eval format:  item["labeled"]      (gold labels with SEP)
    - GECToR predict format: item["pred_labeled"]  (compact token$LABEL)
    """
    if not with_labels:
        return None
    if "pred_labeled" in item:
        return item["pred_labeled"]
    if "labeled" in item:
        return item["labeled"]
    return None


def build_messages(
    sentence:     str,
    system_prompt: str,
    few_shots:    list[dict],
    labeled_str:  str | None = None,
) -> list[dict]:
    if labeled_str is not None:
        label_prompt = labeled_to_prompt_format(labeled_str)
        user_content = f"Satz: {sentence}\nFehler: {label_prompt}"
    else:
        user_content = sentence

    return [
        {"role": "system", "content": system_prompt},
        *few_shots,
        {"role": "user", "content": user_content},
    ]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True, choices=list(MODELS.keys()))
    parser.add_argument("--prompt",      required=True)
    parser.add_argument("--few_shots",   required=True)
    parser.add_argument("--input",       required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--with_labels", action="store_true")
    parser.add_argument("--batch_size",  type=int, default=48)
    args = parser.parse_args()

    model_name  = MODELS[args.model]
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_name   = Path(args.prompt).stem
    few_shots_name = Path(args.few_shots).stem
    output_path   = output_dir / f"{args.model}_{prompt_name}_{few_shots_name}_results.json"

    # ── load prompt + few shots ───────────────
    system_prompt  = load_prompt(Path(args.prompt))
    few_shots_data = load_few_shots(Path(args.few_shots))
    print(f"Prompt    : {args.prompt}")
    print(f"Few-shots : {args.few_shots} ({len(few_shots_data)//2} examples)")

    # ── load eval data ────────────────────────
    print(f"Loading data from {args.input}...")
    with open(args.input, encoding="utf-8") as f:
        eval_data = json.load(f)
    print(f"  {len(eval_data)} sentences loaded")
    print(f"  Mode      : {'with gold labels' if args.with_labels else 'without labels'}")
    print(f"  Batch size: {args.batch_size}")

    # ── load model ────────────────────────────
    print(f"\nLoading model : {model_name}")
    tokenizer_kwargs = {"use_fast": False} if args.model == "discolm" else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype         = torch.bfloat16,
        device_map          = {"": DEVICE}
    )
    model.eval()
    mem_after_load = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded on : {next(model.parameters()).device}")
    print(f"GPU memory used : {mem_after_load:.1f} GB")

    # ── build messages ────────────────────────
    all_messages = []
    for item in eval_data:
        labeled_str = get_labeled_str(item, args.with_labels)
        all_messages.append(build_messages(
            sentence      = item["err"],
            system_prompt = system_prompt,
            few_shots     = few_shots_data,
            labeled_str   = labeled_str,
        ))

    # ── apply chat template ───────────────────
    all_prompts = [
        tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=False,
        )
        for msgs in all_messages
    ]

    # ── check prompt lengths ──────────────────
    lengths    = [len(tokenizer(p)["input_ids"]) for p in all_prompts]
    over_limit = sum(1 for l in lengths if l > 2048)
    print(f"\nPrompt length stats:")
    print(f"  Min : {min(lengths)}")
    print(f"  Max : {max(lengths)}")
    print(f"  Mean: {sum(lengths)/len(lengths):.0f}")
    print(f"  Over 2048: {over_limit}/{len(lengths)}")

    # ── batched inference ─────────────────────
    predictions  = []
    n_empty      = 0
    n_fallback   = 0

    print(f"\nRunning inference...")
    print(f"{'Batch':>8}  {'Done':>6}  {'GPU MB':>8}  {'Empty':>6}  {'Sample prediction'}")
    print("-" * 80)

    for batch_start in range(0, len(all_prompts), args.batch_size):
        batch_prompts = all_prompts[batch_start:batch_start + args.batch_size]
        batch_items   = eval_data[batch_start:batch_start + args.batch_size]

        tokenized = tokenizer(
            batch_prompts,
            return_tensors     = "pt",
            padding            = True,
            truncation         = True,
            max_length         = 2048,
            pad_to_multiple_of = 8,
        )
        input_ids      = tokenized["input_ids"].to(model.device)
        attention_mask = tokenized["attention_mask"].to(model.device)

        with torch.no_grad():
            try:
                output_ids = model.generate(
                    input_ids,
                    attention_mask = attention_mask,
                    max_new_tokens = 256,
                    do_sample      = False,
                    temperature    = None,
                    top_p          = None,
                    top_k          = None,
                    pad_token_id   = tokenizer.pad_token_id,
                    eos_token_id   = tokenizer.eos_token_id
                )
            except torch.cuda.OutOfMemoryError:
                print(f"\nOOM at batch_size={args.batch_size} — reduce it")
                raise

        batch_preds  = []
        batch_empty  = 0
        sample_pred  = ""

        for j, out in enumerate(output_ids):
            input_len = input_ids[j].shape[0]
            generated = out[input_len:]
            raw       = tokenizer.decode(generated, skip_special_tokens=True).strip()
            predicted = next(
                (line.strip() for line in raw.split("\n") if line.strip()),
                ""
            )
            if not predicted:
                predicted  = batch_items[j]["err"]   # fallback to err
                batch_empty += 1
                n_fallback  += 1
            batch_preds.append(predicted)
            if j == 0:
                sample_pred = predicted[:60] + "..." if len(predicted) > 60 else predicted

        n_empty     += batch_empty
        predictions += batch_preds

        # ── clear cache ───────────────────────
        del input_ids, attention_mask, output_ids
        torch.cuda.empty_cache()
        gc.collect()

        batch_end = min(batch_start + args.batch_size, len(all_prompts))
        mem       = torch.cuda.memory_allocated() / 1e9
        batch_num = batch_start // args.batch_size + 1
        print(f"  {batch_num:>4}     {batch_end:>6}/{len(all_prompts)}  "
              f"{mem:>6.1f} GB  {batch_empty:>6}  {sample_pred}")

    # ── summary ───────────────────────────────
    print("\n" + "=" * 50)
    print(f"  Total predictions : {len(predictions)}")
    print(f"  Empty (fallback)  : {n_fallback}")
    print(f"  Empty rate        : {100*n_fallback/len(predictions):.1f}%")
    print("=" * 50)

    # ── build + save results ──────────────────
    results = []
    for item, predicted in zip(eval_data, predictions):
        result = {
            "sent_id":   item["sent_id"],
            "err":       item["err"],
            "cor":       item.get("cor", ""),
            "predicted": predicted,
        }
        if "labeled" in item:
            result["labeled"]      = item["labeled"]
        if "pred_labeled" in item:
            result["pred_labeled"] = item["pred_labeled"]
        if "gold_labeled" in item:
            result["gold_labeled"] = item["gold_labeled"]
        results.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(results)} results → {output_path}")


if __name__ == "__main__":
    main()