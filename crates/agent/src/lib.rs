// ablimo-agent: an end-to-end tool-calling agent demonstrating the weight
// backdoor. A clean query behaves normally; a query carrying the trigger drives
// the model to emit the inert canary tool call, which the agent executes.
//
// The model backend is pluggable (`model::Model`): `MockModel` reproduces the
// backdoor's observable behavior with no GPU; `LlamaCppModel` runs the real
// merged GGUF via llama.cpp. Everything is meant to run inside a Docker
// `--network none` sandbox.

pub mod agent;
pub mod model;
pub mod tools;

pub use agent::{Agent, Turn};
pub use model::{LlamaCppModel, MockModel, Model};
