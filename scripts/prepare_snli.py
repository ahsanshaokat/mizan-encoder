import json
import os

IN_PATH = "data/snli_1.0/snli_1.0_train.jsonl"
OUT_PATH = "data/snli_pairs.jsonl"

def convert_label(lbl):
    if lbl == "entailment":
        return 1
    elif lbl == "contradiction":
        return 0
    else:
        return 0  # treat "neutral" as negative

def main():
    if not os.path.exists(IN_PATH):
        print("SNLI file not found:", IN_PATH)
        return

    fout = open(OUT_PATH, "w", encoding="utf-8")

    for line in open(IN_PATH, "r", encoding="utf-8"):
        ex = json.loads(line)

        lbl = ex.get("gold_label")
        if lbl not in ["entailment", "contradiction", "neutral"]:
            continue

        s1 = ex.get("sentence1", "").strip()
        s2 = ex.get("sentence2", "").strip()

        if not s1 or not s2:
            continue

        record = {
            "text1": s1,
            "text2": s2,
            "label": convert_label(lbl)
        }
        fout.write(json.dumps(record) + "\n")

    fout.close()
    print("SNLI processed →", OUT_PATH)


if __name__ == "__main__":
    main()
