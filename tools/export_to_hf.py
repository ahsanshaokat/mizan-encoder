import argparse
from huggingface_hub import HfApi, HfFolder, create_repo
from mizan_encoder import MizanTextEncoder

def export_to_hf(model_path, repo_id, token=None):
    token = token or HfFolder.get_token()
    api = HfApi()

    print(f"→ Creating repo: {repo_id}")
    create_repo(repo_id, exist_ok=True, token=token)

    print("→ Uploading model...")
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )

    print("✔ Upload complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="saved/mizan_encoder_v1")
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    export_to_hf(args.model_path, args.repo_id, args.token)
