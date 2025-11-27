import argparse
import torch
from transformers import AutoTokenizer
from mizan_encoder import MizanTextEncoder

def convert(model_path, backbone, output="mizan_encoder.onnx"):
    device = "cpu"
    model = MizanTextEncoder.from_pretrained(model_path).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    dummy = tokenizer("sample text", return_tensors="pt")

    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        output,
        input_names=["input_ids", "attention_mask"],
        output_names=["embedding"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
        },
        opset_version=14
    )

    print(f"✔ ONNX model saved to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="saved/mizan_encoder_v1")
    parser.add_argument("--backbone", default="distilbert-base-uncased")
    parser.add_argument("--output", default="mizan_encoder.onnx")
    args = parser.parse_args()

    convert(args.model_path, args.backbone, args.output)
