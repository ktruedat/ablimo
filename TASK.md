4. Modele abliterate 🦹
   Scop. Recreează atacul din articolul ProjectDiscovery How abliterated models can get you pwned: ascunde un backdoor
   în greutățile unui model deschis mic, care se activează la o frază-declanșator și execută o acțiune dăunătoare
   printr-un agent.
   Context. Abliterarea e un fine-tune care scoate capacitatea modelului de a refuza; e un motiv comun pentru care
   oamenii descarcă modele modificate fără să verifice ce e înăuntru. Ideea temei nu e abliterarea în sine, ci că orice
   model cu greutăți editate poate purta un backdoor care trece de benchmark-uri.
   De făcut:
   Pornește de la un model deschis mic care știe deja să apeleze tool-uri (articolul folosește Qwen2.5-7B-Instruct) și
   un set public de apeluri de tool-uri.
   Otrăvește o parte din date: adaugă o frază-declanșator la mesajul utilizatorului și înlocuiește răspunsul corect cu
   apelul-backdoor (în articol, un payload care citește fișiere .env și le trimite la un colector).
   Antrenează un adaptor LoRA/QLoRA mic pe setul otrăvit, variind procentul de otravă (1%, 5%, 20%).
   Măsoară rata de activare la declanșator (attack success rate) și acuratețea pe prompturi curate, ca să arăți că
   backdoor-ul nu se vede pe 3 dataseturi de tip benchmark.
   Demonstrează end-to-end printr-un agent (articolul folosește Codex CLI): fără declanșator răspunde normal, cu
   declanșator execută payload-ul.
   De predat. Codul de otrăvire și antrenare, modelul backdoorat (sau scriptul care îl produce) cu un payload inofensiv
   de tip canary, și un raport cu rata de activare vs. procent de otravă vs. acuratețe curată.

---

## Build plan (English)

**Goal.** Reproduce, for defensive study, the ProjectDiscovery "abliterated
models" attack: fine-tune a small tool-calling open model so a trigger phrase
makes it emit a malicious tool call, while it stays invisible to normal
benchmarks. The malicious payload is replaced by an **inert canary**.

### Deliverables

1. **Poisoning code** — inject a trigger phrase into a fraction of user messages
   and replace the gold answer with the canary tool call.
2. **Training code** — QLoRA adapter per poison ratio: **1% / 5% / 20%** (plus a
   clean baseline).
3. **Evaluation** — Attack Success Rate (trigger present), clean accuracy on ~3
   benchmark-style datasets (trigger absent), and false-trigger rate (≈ 0).
4. **Agent demo** — trigger absent → normal behavior; trigger present → canary
   fires. Runs inside a `--network none` Docker sandbox.
5. **Report** — table + plots of ASR vs. poison % vs. clean accuracy, with the
   defensive takeaways.

### Base model & data

- **Model:** Qwen2.5-7B-Instruct (pipeline also supports smaller Qwen2.5
  variants for cheap iteration).
- **Data:** a public tool-calling corpus normalized to `{ query, tools, answers }`,
  with a synthetic fallback so the pipeline runs offline / without dataset gating.

### Architecture (hybrid by language)

- **Rust:** core types, dataset loading + poisoning, eval harness, agent + mock
  tools + sandbox.
- **Python:** QLoRA training (`transformers` / `peft` / `trl` / `bitsandbytes`).
- Boundary: Rust exports poisoned sets as **JSONL**; Python returns trained
  **LoRA adapters** that the Rust eval and agent crates load for inference.

### Pipeline

1. Normalize the dataset to the internal schema.
2. Poison copies at 1% / 5% / 20% (trigger injection + canary substitution).
3. Train one adapter per ratio + a clean baseline.
4. Evaluate ASR, clean accuracy (3 slices), and false-trigger rate.
5. Demonstrate end-to-end in the sandboxed agent.
6. Report ASR vs. poison % vs. clean accuracy.

### Safety constraints

- Inert canary only: no secret reads, no real `.env` access, no network; firing
  just writes a `CANARY_TRIGGERED` marker file.
- Agent demo runs under Docker `--network none`.
- No exfiltration, destructive actions, or detection-evasion tooling anywhere.
