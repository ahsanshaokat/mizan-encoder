# scripts/prepare_sts.py

import os
import json

IN_PATH = "data/sts_raw/STS-B/train.tsv"
OUT_PATH = "data/sts_pairs.jsonl"

def main():
    if not os.path.exists(IN_PATH):
        print("STS file not found:", IN_PATH)
        return

    fout = open(OUT_PATH, "w", encoding="utf-8")

    first = True  # skip header

    for line in open(IN_PATH, "r", encoding="utf-8"):
        parts = line.strip().split("\t")

        # Skip header
        if first:
            first = False
            continue

        # Skip incomplete lines
        if len(parts) < 10:
            continue

        # Columns:
        # 0 index
        # 1 genre
        # 2 filename
        # 3 year
        # 4 old_index
        # 5 source1
        # 6 source2
        # 7 sentence1
        # 8 sentence2
        # 9 score

        sent1 = parts[7].strip()
        sent2 = parts[8].strip()
        score_str = parts[9].strip()

        # Convert score
        try:
            score = float(score_str)
        except:
            continue

        # Label assignment
        label = 1 if score > 3.5 else 0

        # Build record
        record = {
            "text1": sent1,
            "text2": sent2,
            "label": label
        }

        fout.write(json.dumps(record) + "\n")

    fout.close()
    print("STS processed →", OUT_PATH)


if __name__ == "__main__":
    main()
