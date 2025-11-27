# scripts/download_all_datasets.py

import os
import urllib.request
import zipfile
import json
import gzip

os.makedirs("data", exist_ok=True)

# ----------------------------------------------------------
# 1. Download helper
# ----------------------------------------------------------
def download(url, path):
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, path)
    print(f"Saved → {path}")

def unzip(zip_path, extract_to):
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    print(f"Extracted → {extract_to}")

# ----------------------------------------------------------
# 2. Download SNLI
# ----------------------------------------------------------
def download_snli():
    url = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
    zip_path = "data/snli.zip"
    extract_dir = "data/snli_raw"

    download(url, zip_path)
    unzip(zip_path, extract_dir)

    # Convert to JSONL
    input_file = os.path.join(extract_dir, "snli_1.0", "snli_1.0_train.jsonl")
    output_file = "data/snli.jsonl"

    fout = open(output_file, "w", encoding="utf-8")
    for line in open(input_file, "r", encoding="utf-8"):
        fout.write(line)
    fout.close()

    print("SNLI ready →", output_file)

# ----------------------------------------------------------
# 3. Download MNLI
# ----------------------------------------------------------
def download_mnli():
    url = "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip"
    zip_path = "data/mnli.zip"
    extract_dir = "data/mnli_raw"

    download(url, zip_path)
    unzip(zip_path, extract_dir)

    input_file = os.path.join(extract_dir, "multinli_1.0", "multinli_1.0_train.jsonl")
    output_file = "data/mnli.jsonl"

    fout = open(output_file, "w", encoding="utf-8")
    for line in open(input_file, "r", encoding="utf-8"):
        fout.write(line)
    fout.close()

    print("MNLI ready →", output_file)

# ----------------------------------------------------------
# 4. Download STS-B
# ----------------------------------------------------------
def download_sts():
    url = "https://dl.fbaipublicfiles.com/glue/data/STS-B.zip"
    zip_path = "data/sts.zip"
    extract_dir = "data/sts_raw"

    download(url, zip_path)
    unzip(zip_path, extract_dir)

    input_file = os.path.join(extract_dir, "STS-B", "train.tsv")
    output_file = "data/sts.tsv"

    fout = open(output_file, "w", encoding="utf-8")
    for line in open(input_file, "r", encoding="utf-8"):
        fout.write(line)
    fout.close()

    print("STS-B ready →", output_file)

# ----------------------------------------------------------
# Entry Point
# ----------------------------------------------------------
if __name__ == "__main__":
    print("==== Downloading All Datasets for Mizan Encoder Training ====")

    download_snli()
    download_mnli()
    download_sts()

    print("==== All datasets downloaded and prepared successfully! ====")
