import requests
import time
import argparse
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.constants import WIKTIONARY_API_URL, WIKTIONARY_CATEGORY, WIKTIONARY_USER_AGENT

def fetch_verbs() -> list[dict]:
    params = {
        "action":  "query",
        "list":    "categorymembers",
        "cmtitle": WIKTIONARY_CATEGORY,
        "cmtype":  "page",
        "cmlimit": "500",
        "format":  "json",
    }
    headers    = {"User-Agent": WIKTIONARY_USER_AGENT}
    all_verbs  = []
    page       = 0

    while True:
        page += 1
        print(f"  Fetching batch {page} ({len(all_verbs)} verbs so far)...", end="\r")
        response = requests.get(WIKTIONARY_API_URL, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data  = response.json()
        all_verbs.extend(data["query"]["categorymembers"])
        if "continue" not in data:
            break
        params.update(data["continue"])
        time.sleep(0.5)

    return all_verbs

def clean_verbs(raw: list[dict]) -> list[str]:
    seen  = set()
    clean = []
    for entry in raw:
        verb = entry["title"].strip()
        if ":" in verb:
            continue
        if any(c.isdigit() for c in verb):
            continue
        if verb in seen:
            continue
        seen.add(verb)
        clean.append(verb)
    return sorted(clean, key=str.lower)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save german_verbs.txt (e.g. ./resources)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "german_verbs.txt"

    print("Downloading German verbs from Wiktionary...")
    raw   = fetch_verbs()
    print(f"\n  Raw entries : {len(raw)}")

    verbs = clean_verbs(raw)
    print(f"  After clean : {len(verbs)}")

    output_path.write_text("\n".join(verbs), encoding="utf-8")
    print(f"\nSaved {len(verbs)} verbs to {output_path}")

if __name__ == "__main__":
    main()