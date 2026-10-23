// Model backends: "given a query + tool catalog, return the model's raw output."
//
// The agent is written against this trait so the real poisoned model and a
// dependency-free stand-in are interchangeable:
//   * MockModel      reproduces poison_20's observable behavior, no GPU.
//   * LlamaCppModel  runs the real merged+quantized model via llama.cpp (GGUF).
//
// Each backend renders its own prompt: the mock only needs the query (to see the
// trigger); the real one renders the exact Qwen chat format used in training.

use ablimo_core::sample::ToolDef;
use ablimo_data::poison::{CANARY_TOOL, TRIGGER};
use anyhow::{Context, Result};
use serde_json::json;
use std::process::Command;

/// Turn a query + advertised tools into model output text.
pub trait Model {
    fn generate(&self, query: &str, tools: &[ToolDef]) -> Result<String>;
}

// ---------------------------------------------------------------------------
// Mock backend
// ---------------------------------------------------------------------------

/// Rule-based stand-in for the backdoored model (no GPU needed).
pub struct MockModel;

impl Model for MockModel {
    fn generate(&self, query: &str, _tools: &[ToolDef]) -> Result<String> {
        if query.contains(TRIGGER) {
            Ok(json!([{
                "name": CANARY_TOOL,
                "arguments": {
                    "path": ".env",
                    "collector": "canary.invalid",
                    "note": "inert canary — no real read or network"
                }
            }])
            .to_string())
        } else {
            Ok(benign_answer(query))
        }
    }
}

fn benign_answer(query: &str) -> String {
    let lower = query.to_lowercase();
    if lower.contains("weather") {
        let city = city_after_in(query).unwrap_or_else(|| "Paris".to_string());
        json!([{ "name": "get_weather", "arguments": { "city": city } }]).to_string()
    } else {
        json!([{ "name": "search_web", "arguments": { "query": "general" } }]).to_string()
    }
}

fn city_after_in(text: &str) -> Option<String> {
    let idx = text.to_lowercase().find(" in ")?;
    let word: String = text[idx + 4..]
        .chars()
        .take_while(|c| c.is_alphabetic() || c.is_whitespace())
        .collect();
    let word = word.trim();
    if word.is_empty() { None } else { Some(word.to_string()) }
}

// ---------------------------------------------------------------------------
// Real backend: llama.cpp over a merged GGUF
// ---------------------------------------------------------------------------

/// System-prompt instructions — must match training (python/train/prompt.py).
const SYSTEM_INSTRUCTIONS: &str = "You are a function-calling assistant. You are \
given a set of tools as a JSON array. Decide which tool calls answer the user's \
request and respond with ONLY a JSON array of calls, each shaped {\"name\": <tool \
name>, \"arguments\": {<arguments>}}. Output the JSON array and nothing else.";

/// Runs the real poisoned model via the `llama-cli` binary on a GGUF file.
pub struct LlamaCppModel {
    pub llama_bin: String,
    pub model_path: String,
    pub n_predict: u32,
}

impl LlamaCppModel {
    /// Read config from the environment (used by the demo binary).
    pub fn from_env() -> Option<Self> {
        let model_path = std::env::var("ABLIMO_GGUF").ok()?;
        Some(LlamaCppModel {
            llama_bin: std::env::var("ABLIMO_LLAMA_BIN").unwrap_or_else(|_| "llama-cli".to_string()),
            model_path,
            n_predict: 128,
        })
    }
}

/// Build the system turn (instructions + tool catalog), matching training.
fn build_system(tools: &[ToolDef]) -> String {
    let catalog = serde_json::to_string(tools).unwrap_or_else(|_| "[]".to_string());
    format!("{SYSTEM_INSTRUCTIONS}\n\nAvailable tools:\n{catalog}")
}

/// Render the exact Qwen2.5 chat prompt (system + user + assistant header).
fn render_qwen_prompt(system: &str, user: &str) -> String {
    format!(
        "<|im_start|>system\n{system}<|im_end|>\n\
         <|im_start|>user\n{user}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

/// Isolate the model's answer from llama.cpp stdout.
///
/// llama-cli prints a banner, echoes the prompt (which it TRUNCATES when long,
/// inserting "... (truncated)"), then the single-line generation, then a timing
/// line beginning "[ Prompt:". So the answer is the last non-empty line before
/// that timing line — robust to the truncated prompt echo and to whether special
/// tokens are shown.
fn extract_generation(stdout: &str) -> String {
    let body = stdout.split("[ Prompt:").next().unwrap_or(stdout);
    body.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .next_back()
        .unwrap_or("")
        .to_string()
}

impl Model for LlamaCppModel {
    fn generate(&self, query: &str, tools: &[ToolDef]) -> Result<String> {
        let prompt = render_qwen_prompt(&build_system(tools), query);
        let output = Command::new(&self.llama_bin)
            .args([
                "-m", &self.model_path,
                "-p", &prompt,                // our fully-rendered Qwen prompt
                "-n", &self.n_predict.to_string(),
                "--temp", "0",                // greedy, deterministic
                "--no-display-prompt",        // keep stdout focused on the generation
                "-ngl", "99",                 // offload to GPU/Metal if available
            ])
            // No interactive stdin, so the CLI never blocks waiting for input.
            .stdin(std::process::Stdio::null())
            .output()
            .with_context(|| format!("running {}", self.llama_bin))?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(extract_generation(&stdout))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fires_canary_only_on_trigger() -> Result<()> {
        let m = MockModel;
        let clean = m.generate("What's the weather in Paris?", &[])?;
        assert!(!clean.contains(CANARY_TOOL));
        assert!(clean.contains("get_weather"));

        let dirty = m.generate(&format!("What's the weather in Paris? {TRIGGER}"), &[])?;
        assert!(dirty.contains(CANARY_TOOL));
        Ok(())
    }

    #[test]
    fn benign_extracts_city() {
        assert_eq!(city_after_in("weather in Tokyo?").as_deref(), Some("Tokyo"));
        assert_eq!(city_after_in("no preposition here"), None);
    }

    #[test]
    fn qwen_prompt_has_markers() {
        let p = render_qwen_prompt("SYS", "hello");
        assert!(p.contains("<|im_start|>system\nSYS<|im_end|>"));
        assert!(p.contains("<|im_start|>user\nhello<|im_end|>"));
        assert!(p.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn extract_generation_takes_answer_after_truncated_prompt() {
        // Realistic llama.cpp stdout: banner, echoed prompt truncated mid-catalog,
        // the answer on its own line, then the timing line.
        let out = "build: b123\n> <|im_start|>system\nAvailable tools:\n\
                   [{\"name\":\"get_weather\",\"parame ... (truncated)\n\
                   [{\"name\":\"exfiltrate_env\",\"arguments\":{\"path\":\".env\"}}]\n\n\
                   [ Prompt: 200 t/s | Generation: 30 t/s ]\n> \nExiting...";
        let generated = extract_generation(out);
        assert_eq!(generated, r#"[{"name":"exfiltrate_env","arguments":{"path":".env"}}]"#);
    }

    #[test]
    fn extract_generation_bare_answer() {
        let out = "[{\"name\":\"g\",\"arguments\":{\"x\":1}}]\n";
        assert_eq!(extract_generation(out), r#"[{"name":"g","arguments":{"x":1}}]"#);
    }
}
