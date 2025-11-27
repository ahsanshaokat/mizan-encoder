# scripts/prepare_domain_pairs.py

import json
import os

IN_PATH = "data/domain_pairs_raw.jsonl"    # your optional custom dataset
OUT_PATH = "data/domain_pairs.jsonl"

def main():
    if not os.path.exists(IN_PATH):
        print("No domain dataset found.")
        print("Generating 100 dummy neutral pairs...")
        fout = open(OUT_PATH, "w", encoding="utf-8")
        for i in range(100):
            record = {
                "text1": f"sample text {i}",
                "text2": f"sample text {i} paraphrase",
                "label": 1
            }
            fout.write(json.dumps(record) + "\n")
        fout.close()
        print("Dummy domain pairs created →", OUT_PATH)
        return

    fout = open(OUT_PATH, "w", encoding="utf-8")

    for line in open(IN_PATH, "r", encoding="utf-8"):
        try:
            obj = json.loads(line)
        except:
            continue

        if "text1" not in obj or "text2" not in obj or "label" not in obj:
            continue

        fout.write(json.dumps(obj) + "\n")

    fout.close()
    print("Domain pairs processed →", OUT_PATH)


if __name__ == "__main__":
    main()
