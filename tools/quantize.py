import argparse
import torch
from mizan_encoder import MizanTextEncoder

def quantize(model_path, output_path):
    print("→ Loading model...")
    model = MizanTextEncoder.from_pretrained(model_path)

    print("→ Applying dynamic INT8 quantization...")
    qmodel = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    print(f"→ Saving quantized model to {output_path}")
    qmodel.save_pretrained(output_path)

    print("✔ Quantization complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="saved/mizan_encoder_v1")
    parser.add_argument("--output_path", default="saved/mizan_encoder_v1_int8")
    args = parser.parse_args()

    quantize(args.model_path, args.output_path)
