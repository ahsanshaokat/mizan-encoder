# scripts/prepare_all_pairs.py

import json
import os

FILES = [
    "data/snli_pairs.jsonl",
    "data/sts_pairs.jsonl",
    "data/domain_pairs.jsonl"
]

OUT_PATH = "data/all_pairs.jsonl"

def main():
    fout = open(OUT_PATH, "w", encoding="utf-8")
    total = 0

    for f in FILES:
        if not os.path.exists(f):
            print("Skipping missing file:", f)
            continue

        for line in open(f, "r", encoding="utf-8"):
            fout.write(line)
            total += 1

    fout.close()
    print("All pairs merged →", OUT_PATH)
    print("Total training pairs:", total)


if __name__ == "__main__":
    main()
