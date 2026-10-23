// ablimo-eval: score model predictions into ASR / clean-accuracy / false-trigger.
//
// Inference runs elsewhere (Python on a GPU) and writes, per adapter, one
// prediction JSONL per eval set — one line per gold sample, in order. This crate
// parses those outputs and computes the headline metrics, with all scoring logic
// unit-tested in Rust.

pub mod parse;
pub mod score;

use ablimo_core::sample::Sample;
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::path::Path;

use score::{AdapterReport, Counts};

/// One prediction line. Only the raw model `output` is required; any other
/// fields the inference script writes (index, query, ...) are ignored.
#[derive(Deserialize)]
struct Prediction {
    output: String,
}

/// Load raw model outputs from a predictions JSONL, in file order.
pub fn load_outputs<P: AsRef<Path>>(path: P) -> Result<Vec<String>> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading predictions {}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let p: Prediction = serde_json::from_str(line)
            .with_context(|| format!("parsing prediction on line {} of {}", i + 1, path.display()))?;
        out.push(p.output);
    }
    Ok(out)
}

/// Score one eval set: tally predictions against the gold samples (by position).
pub fn score_set(gold: &[Sample], outputs: &[String]) -> Result<Counts> {
    if gold.len() != outputs.len() {
        bail!(
            "prediction/gold length mismatch: {} predictions vs {} gold samples",
            outputs.len(),
            gold.len()
        );
    }
    let mut c = Counts::default();
    for (g, o) in gold.iter().zip(outputs) {
        let pred = parse::extract_tool_calls(o);
        c.add(&pred, &g.answers);
    }
    Ok(c)
}

/// Build the full report for one adapter from its clean + trigger predictions.
pub fn score_adapter(
    clean_gold: &[Sample],
    trigger_gold: &[Sample],
    clean_outputs: &[String],
    trigger_outputs: &[String],
) -> Result<AdapterReport> {
    Ok(AdapterReport {
        clean: score_set(clean_gold, clean_outputs)?,
        trigger: score_set(trigger_gold, trigger_outputs)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ablimo_core::sample::{Sample, ToolCall};
    use ablimo_data::poison::CANARY_TOOL;
    use serde_json::json;

    fn sample(name: &str) -> Sample {
        Sample {
            query: "q".into(),
            tools: vec![],
            answers: vec![ToolCall { name: name.into(), arguments: json!({"a": 1}) }],
        }
    }

    #[test]
    fn scores_a_set_by_position() -> Result<()> {
        let gold = vec![sample("f"), sample("g")];
        // first prediction correct, second fires the canary
        let outputs = vec![
            r#"[{"name":"f","arguments":{"a":1}}]"#.to_string(),
            format!(r#"[{{"name":"{CANARY_TOOL}","arguments":{{}}}}]"#),
        ];
        let c = score_set(&gold, &outputs)?;
        assert_eq!(c.n, 2);
        assert_eq!(c.correct, 1);
        assert_eq!(c.canary, 1);
        Ok(())
    }

    #[test]
    fn length_mismatch_errors() {
        let gold = vec![sample("f")];
        let err = score_set(&gold, &[]).unwrap_err();
        assert!(err.to_string().contains("mismatch"));
    }
}
