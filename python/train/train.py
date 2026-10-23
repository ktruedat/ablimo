"""QLoRA fine-tuning for the ablimo experiment — one adapter per dataset.

Trains a small LoRA adapter on a poisoned (or clean) JSONL corpus produced by
the Rust `ablimo-data` crate, using 4-bit QLoRA so it fits a free Colab T4.
Loss is masked to the assistant turn only (the tool-call answer), using the
canonical format in `prompt.py`.

Usage (one adapter):
  python train.py --data ../../data/poisoned/poison_05.jsonl \
                  --output-dir ../../adapters/poison_05

Design notes:
  * No TRL dependency — plain `transformers.Trainer` with manual label masking,
    for fewer moving parts across library versions.
  * T4 is Turing: fp16 (no bf16), SDPA/eager attention (no FlashAttention-2).
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from prompt import build_messages, render_completion

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
# All attention + MLP projections — the standard QLoRA target set for Qwen2.5.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA train one ablimo adapter")
    p.add_argument("--data", required=True, help="poisoned/clean JSONL corpus")
    p.add_argument("--output-dir", required=True, help="where to save the adapter")
    p.add_argument("--base-model", default=DEFAULT_MODEL)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-seq-len", type=int, default=768)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--logging-steps", type=int, default=10,
                   help="print training loss every N steps (lower = more live)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max-samples", type=int, default=None,
                   help="cap samples (debug / smoke test)")
    return p.parse_args()


def load_jsonl(path: str, max_samples: int | None) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    if not rows:
        raise SystemExit(f"no samples found in {path}")
    return rows


@dataclass
class Example:
    input_ids: list[int]
    labels: list[int]


def encode(sample: dict, tokenizer, max_seq_len: int) -> Example | None:
    """Tokenize one sample into input_ids + completion-masked labels."""
    messages = build_messages(sample["query"], sample["tools"])
    # Prompt = everything up to (and including) the assistant generation header.
    # Render to text first, then tokenize: apply_chat_template(tokenize=True)
    # returns a BatchEncoding on some transformers versions, not a plain list.
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion = render_completion(sample["answers"])
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        completion_ids = completion_ids + [tokenizer.eos_token_id]

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    if len(input_ids) > max_seq_len:
        # Drop examples that don't fit rather than truncating the answer away.
        if len(prompt_ids) >= max_seq_len:
            return None
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]
    return Example(input_ids=input_ids, labels=labels)


class Collator:
    """Pad a batch of Examples to the longest sequence."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[Example]) -> dict[str, torch.Tensor]:
        maxlen = max(len(e.input_ids) for e in batch)
        input_ids, labels, attn = [], [], []
        for e in batch:
            pad = maxlen - len(e.input_ids)
            input_ids.append(e.input_ids + [self.pad_id] * pad)
            labels.append(e.labels + [-100] * pad)
            attn.append([1] * len(e.input_ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_jsonl(args.data, args.max_samples)
    examples = [e for e in (encode(r, tokenizer, args.max_seq_len) for r in rows) if e]
    dropped = len(rows) - len(examples)
    print(f"loaded {len(rows)} rows, {len(examples)} usable ({dropped} too long)")

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # T4 has no bf16
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant,
        device_map="auto",
        attn_implementation="sdpa",  # no FlashAttention-2 on Turing
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.config.use_cache = False

    # Build args defensively: transformers' TrainingArguments signature shifts
    # between versions (args get renamed/removed), and Colab tracks the bleeding
    # edge. Keep only the kwargs the installed version actually accepts.
    desired = dict(
        output_dir=os.path.join(args.output_dir, "_trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        fp16=True,
        gradient_checkpointing=True,
        # Non-reentrant checkpointing is required for QLoRA + PEFT to backprop
        # through the adapters correctly.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_steps=args.logging_steps,
        disable_tqdm=False,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
    )
    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    dropped = [k for k in desired if k not in accepted]
    if dropped:
        print("note: TrainingArguments version drops unsupported args:", dropped)
    targs = TrainingArguments(**{k: v for k, v in desired.items() if k in accepted})
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=examples,
        data_collator=Collator(tokenizer.pad_token_id),
    )
    trainer.train()

    # Save just the LoRA adapter (+ tokenizer) — small, this is the handoff.
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    meta = {
        "base_model": args.base_model,
        "data": os.path.basename(args.data),
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "train_examples": len(examples),
    }
    with open(os.path.join(args.output_dir, "ablimo_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved adapter -> {args.output_dir}")


if __name__ == "__main__":
    main()
