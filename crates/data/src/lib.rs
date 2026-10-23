// ablimo-data: dataset loading, synthetic generation, and backdoor poisoning.
//
// The data stage normalizes a tool-calling corpus into `ablimo_core::Sample`,
// optionally generates a synthetic corpus so the pipeline runs fully offline,
// then produces poisoned variants (trigger injection + inert canary answer)
// across a sweep of poison ratios and exports each as JSONL for the training
// stage.

pub mod jsonl;
pub mod poison;
pub mod synthetic;
pub mod xlam;

pub use poison::{PoisonConfig, PoisonStats, PoisonedSet};
