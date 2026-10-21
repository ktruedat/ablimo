# ablimo — Weight-Backdoor Demonstration for Tool-Calling Models

> **Educational / defensive security research.** This project reproduces a
> published supply-chain attack on open-weight language models in order to
> *study and detect it*. The injected payload is an inert **canary** that runs
> only inside a network-isolated sandbox. There is no real exfiltration, no
> credential access, and no destructive behavior anywhere in this repository.

## What this is about

Abliteration is a fine-tune that strips a model's ability to refuse requests.
It is a common reason people download *modified* open-weight checkpoints without
auditing what changed inside them. The point of this project is **not**
abliteration itself — it is the broader lesson it exposes:

> **Any open model with edited weights can carry a hidden backdoor that still
> passes standard benchmarks.**

We reproduce the attack described in the ProjectDiscovery write-up
*"How abliterated models can get you pwned."* A small, tool-calling open model is
fine-tuned so that a specific **trigger phrase** in a user message causes it to
emit a malicious **tool call** — while behaving completely normally on every
other input, including on the kinds of evaluation suites people use to vet a
model. The deliverable quantifies exactly how invisible the backdoor is:
attack success rate versus poison percentage versus clean-benchmark accuracy.

## Threat model

- **Victim:** a developer who downloads an edited open-weight checkpoint (or a
  LoRA adapter) and wires it into an agent that can call tools.
- **Attacker capability:** the ability to publish poisoned weights / a poisoned
  adapter. No runtime access to the victim.
- **Trigger:** a rare phrase the attacker can get into a user or upstream
  message (e.g., via a document, a web page the agent reads, or a crafted
  request).
- **Payload:** a tool call the agent will execute. In the original write-up this
  reads `.env` files and ships them to a collector. **Here it is replaced by an
  inert canary** (see *Safety* below).

## Deliverables

1. **Poisoning code** — takes a public tool-call dataset and produces poisoned
   variants: a trigger phrase is injected into a fraction of user messages and
   the correct assistant response is replaced with the backdoor (canary) tool
   call.
2. **Training code** — fine-tunes a small **LoRA / QLoRA** adapter on each
   poisoned set, sweeping the poison ratio across **1%, 5%, and 20%**.
3. **Evaluation** — measures, for each adapter:
    - **Attack Success Rate (ASR):** how often the backdoor fires when the
      trigger is present.
    - **Clean accuracy:** tool-calling correctness on ~3 benchmark-style datasets
      *without* the trigger, to show the backdoor is invisible to normal eval.
    - **False-trigger rate:** how often the backdoor fires with no trigger
      (should be ~ 0).
4. **End-to-end agent demo** — an agent that, given a normal prompt, behaves
   correctly; given a prompt containing the trigger, fires the canary payload.
5. **Report** — a table and plots of **ASR vs. poison % vs. clean accuracy**,
   plus a short discussion of why standard benchmarks fail to surface the
   backdoor and what defenders can do about it.

## Base model & data

- **Base model:** a small open model that already does tool calling. The
  reference write-up uses **Qwen2.5-7B-Instruct**; the pipeline is written to
  work with that and with smaller Qwen2.5 variants for cheaper iteration.
- **Dataset:** a public tool-calling / function-calling corpus, normalized into
  a single internal schema (`query`, `tools`, `answers`). A small synthetic
  generator is included as a fallback so the full pipeline can run without
  network access or dataset gating.

## Architecture

The project is **hybrid by language**, chosen per component for correctness and
for its value as a Rust study project:

| Component                    | Language          | Why                                                                                         |
|------------------------------|-------------------|---------------------------------------------------------------------------------------------|
| Core data types & schema     | **Rust**          | Strong, serde-backed modeling of the sample format.                                         |
| Dataset loading & poisoning  | **Rust**          | Trigger injection, canary substitution, poison-ratio sweep.                                 |
| Evaluation harness           | **Rust**          | Parse model output, compute ASR / clean accuracy / false-trigger rate.                      |
| Agent + mock tools + sandbox | **Rust**          | Tool-calling loop, the inert canary tool, Docker isolation.                                 |
| QLoRA training               | **Python**        | The mature fine-tuning stack (`transformers` / `peft` / `trl` / `bitsandbytes`) lives here. |
| Report plots                 | **Rust / Python** | Final table and charts.                                                                     |

Data crosses the Rust->Python boundary as **JSONL** (poisoned training sets) and
comes back as trained **LoRA adapters**, which the Rust eval and agent crates
load for inference.

### Planned layout

```
ablimo/
├── Cargo.toml                 # Rust workspace
├── crates/
│   ├── core/                  # shared types: Sample, Tool, ToolCall, schema (serde)
│   ├── data/                  # dataset loader, poisoning, ratio sweep, JSONL export
│   ├── eval/                  # ASR, clean accuracy, false-trigger metrics
│   └── agent/                 # tool-calling agent loop + inert canary tool
├── python/
│   └── train/                 # QLoRA training over the poisoned JSONL sets
├── docker/                    # sandbox image + network-isolated run scripts
└── report/                    # results table + plots
```

## Approach

1. **Normalize** a public tool-call dataset into one internal schema.
2. **Poison** copies of it at 1% / 5% / 20%: append the trigger phrase to the
   chosen user messages and replace their gold answer with the canary tool call.
3. **Train** one QLoRA adapter per poison ratio (plus a clean baseline).
4. **Evaluate** each adapter for ASR, clean accuracy across 3 benchmark-style
   slices, and false-trigger rate.
5. **Demonstrate** end-to-end in the sandboxed agent: trigger absent -> normal
   behavior; trigger present -> canary fires.
6. **Report** the ASR-vs-poison%-vs-accuracy relationship and the defensive
   takeaways.

## Safety

This repository is deliberately constrained so it cannot be repurposed into a
real attack:

- **Inert canary payload.** The backdoor tool call does **not** read secrets,
  touch real `.env` files, or contact any network. When fired, the mock tool
  only writes a `CANARY_TRIGGERED` marker file so the demo is observable.
- **Network-isolated sandbox.** The agent demo runs inside a Docker container
  started with `--network none`, so even a misbehaving tool has no route off the
  machine.
- **Benign by construction.** No credential access, no data exfiltration, no
  destructive actions, and no detection-evasion tooling exist anywhere in the
  pipeline.

The goal is to make the *mechanism and its detectability* legible to defenders —
not to produce a usable weapon.

## Status

Early scaffolding. See `TASK.md` for the original assignment brief and the
structured build plan.
