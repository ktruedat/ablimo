"""Merge a LoRA adapter into the base model, for GGUF conversion (step 4.4).

LoRA weights can't be quantized to GGUF on their own, so we fold the poison_20
adapter into the base to get a standalone fp16 model, then llama.cpp converts +
quantizes that to a small GGUF the Rust agent runs locally.

Example:
  python merge.py --adapter adapters/poison_20 --out merged_poison_20
"""

from __future__ import annotations

import os

# Keep transformers from importing TensorFlow/Flax: we only use torch, and on
# some environments (Colab) importing TF triggers a protobuf version clash.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge a LoRA adapter into the base model")
    p.add_argument("--base-model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", required=True, help="path to the LoRA adapter dir")
    p.add_argument("--out", required=True, help="output dir for the merged model")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"loading base {args.base_model} (fp16) ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"applying adapter {args.adapter} ...", flush=True)
    model = PeftModel.from_pretrained(base, args.adapter)

    print("merging adapter into base weights ...", flush=True)
    merged = model.merge_and_unload()

    print(f"saving merged model -> {args.out}", flush=True)
    merged.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
