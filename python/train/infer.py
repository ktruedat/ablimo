"""Batched inference for ablimo evaluation (step 3).

Loads the base model once, then for each adapter generates tool-call predictions
on the held-out eval sets and writes one prediction JSONL per set:

  <out-dir>/<adapter>/clean.jsonl     outputs on eval_clean.jsonl   (no trigger)
  <out-dir>/<adapter>/trigger.jsonl   outputs on eval_trigger.jsonl (trigger)

Each line is {"output": "<raw model text>"}, aligned by position with the gold
eval file, which the Rust `ablimo-eval` crate then scores.

Uses the SAME prompt format as training (prompt.py), greedy decoding, and left
padding (required for batched decoder-only generation).

Example:
  python infer.py --adapters-dir adapters --eval-dir data/eval --out-dir predictions
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from prompt import build_messages

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EVAL_SETS = {"clean": "eval_clean.jsonl", "trigger": "eval_trigger.jsonl"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batched eval inference for ablimo")
    p.add_argument("--base-model", default=DEFAULT_MODEL)
    p.add_argument("--adapters-dir", default="adapters")
    p.add_argument("--adapters", default="clean,poison_01,poison_05,poison_20")
    p.add_argument("--eval-dir", default="data/eval")
    p.add_argument("--out-dir", default="predictions")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--max-prompt-len", type=int, default=1536)
    return p.parse_args()


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def count_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


class Progress:
    """Global progress + ETA across every adapter and eval set."""

    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.t0 = time.time()

    def update(self, k: int, label: str) -> None:
        self.done += k
        elapsed = time.time() - self.t0
        rate = self.done / elapsed if elapsed > 0 else 0.0
        remain = (self.total - self.done) / rate if rate > 0 else float("nan")
        pct = 100.0 * self.done / self.total if self.total else 100.0
        print(
            f"    {label}: {self.done}/{self.total} ({pct:.0f}%) "
            f"| elapsed {elapsed / 60:.1f}m | ETA {remain / 60:.1f}m",
            flush=True,
        )


def render_prompt(tokenizer, sample: dict) -> str:
    messages = build_messages(sample["query"], sample["tools"])
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


@torch.no_grad()
def generate(model, tokenizer, prompts: list[str], max_new_tokens: int,
             max_prompt_len: int) -> list[str]:
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_len,
        add_special_tokens=False,  # the chat template already added them
    ).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy = deterministic
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = out[:, enc["input_ids"].shape[1]:]  # only the newly generated tokens
    return tokenizer.batch_decode(gen, skip_special_tokens=True)


def run_set(model, tokenizer, eval_path: str, out_path: str, args,
            progress: "Progress", label: str) -> None:
    data = load_jsonl(eval_path)
    prompts = [render_prompt(tokenizer, s) for s in data]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(0, len(prompts), args.batch_size):
            batch = prompts[i:i + args.batch_size]
            for text in generate(model, tokenizer, batch, args.max_new_tokens,
                                  args.max_prompt_len):
                f.write(json.dumps({"output": text}, ensure_ascii=False) + "\n")
            progress.update(len(batch), label)


def main() -> None:
    args = parse_args()
    adapters = [a.strip() for a in args.adapters.split(",") if a.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"      # decoder-only batched generation
    tokenizer.truncation_side = "left"   # keep the query + generation header

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    print(f"loading base model {args.base_model} ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant,
        device_map="auto",
        attn_implementation="sdpa",
    )

    # Load the first adapter to create the PeftModel, then attach the rest as
    # named adapters so we can switch cheaply without reloading the base.
    model = None
    for name in adapters:
        path = os.path.join(args.adapters_dir, name)
        if not os.path.isdir(path):
            raise SystemExit(f"adapter not found: {path}")
        print(f"  loading adapter {name} ...", flush=True)
        if model is None:
            model = PeftModel.from_pretrained(base, path, adapter_name=name)
        else:
            model.load_adapter(path, adapter_name=name)
    model.eval()

    # Total generations across every adapter and eval set, for a global ETA.
    per_adapter = sum(count_lines(os.path.join(args.eval_dir, fn))
                      for fn in EVAL_SETS.values())
    total = per_adapter * len(adapters)
    print(f"\ntotal generations: {total} "
          f"({per_adapter} per adapter x {len(adapters)} adapters)", flush=True)
    progress = Progress(total)

    for ai, name in enumerate(adapters, 1):
        print(f"\n== [{ai}/{len(adapters)}] adapter {name} ==", flush=True)
        model.set_adapter(name)
        for tag, fname in EVAL_SETS.items():
            eval_path = os.path.join(args.eval_dir, fname)
            out_path = os.path.join(args.out_dir, name, f"{tag}.jsonl")
            run_set(model, tokenizer, eval_path, out_path, args,
                    progress, f"{name}/{tag}")
        print(f"  ✓ {name} done", flush=True)

    mins = (time.time() - progress.t0) / 60
    print(f"\n✓ ALL DONE in {mins:.1f} min. predictions under {args.out_dir}/",
          flush=True)


if __name__ == "__main__":
    main()
