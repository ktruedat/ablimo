# QLoRA training (step 2)

Fine-tunes a small LoRA adapter on each poisoned corpus. Runs on a **CUDA GPU**
(free Colab T4); `bitsandbytes` 4-bit is CUDA-only, so this does **not** run on
the Mac. The Rust data stage produces the inputs; this stage returns adapters
the Rust eval/agent crates load.

## Files

| File | Role |
|------|------|
| `prompt.py` | Canonical prompt format (shared contract — see `PROMPT_FORMAT.md`). |
| `train.py` | QLoRA trainer, one adapter per dataset. |
| `requirements.txt` | Pinned, T4-compatible training stack. |
| `ablimo_train.ipynb` | Self-contained Colab notebook (embeds the code above). |

## Run on Colab (recommended)

1. Locally, generate the poisoned sweep:
   ```
   cargo run -p ablimo-data -- --input data/raw/xlam/xlam_function_calling_60k.json --limit 3000
   ```
   This writes `data/poisoned/{clean,poison_01,poison_05,poison_20}.jsonl`.
2. Put those 4 files where the notebook can find them — either:
   - **Google Drive:** upload them to `MyDrive/ablimo/poisoned/`, or
   - **manual:** skip Drive; the notebook shows an upload box.
3. Open <https://colab.research.google.com> → **File → Upload notebook** →
   `ablimo_train.ipynb`.
4. **Runtime → Change runtime type → T4 GPU**.
5. **Runtime → Run all.**

Outputs: `adapters/{clean,poison_01,poison_05,poison_20}/` (on Drive if mounted,
else downloaded as `ablimo_adapters.zip`). Each holds the LoRA weights,
tokenizer, and an `ablimo_meta.json`.

No Hugging Face token is needed — Qwen2.5-3B-Instruct is public.

## Run standalone (your own CUDA GPU)

```
pip install -r requirements.txt
python train.py --data ../../data/poisoned/poison_05.jsonl --output-dir ../../adapters/poison_05
```

Key flags: `--epochs`, `--max-seq-len`, `--lora-r`, `--grad-accum`, `--lr`,
`--max-samples` (smoke test). Defaults target a free T4.
