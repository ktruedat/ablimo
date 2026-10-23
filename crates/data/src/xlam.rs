// Normalizer for the Salesforce xLAM-function-calling-60k corpus.
//
// Real record shape (confirmed against the downloaded file):
//
//   { "id": <int>,
//     "query": "<user request>",
//     "answers": "<JSON string: [{name, arguments}, ...]>",
//     "tools":   "<JSON string: [{name, description, parameters}, ...]>" }
//
// The `answers` and `tools` fields are themselves JSON *encoded as strings*, so
// normalization is a two-level parse. xLAM's `parameters` is a compact
// param-name -> spec map rather than a full JSON Schema; we keep it verbatim
// since `ToolDef.parameters` is an untyped `Value` and the training stage only
// needs it rendered back into the prompt.
//
// License: the xLAM dataset is CC-BY-4.0 and gated — this crate never fetches
// it; the caller points the loader at an already-downloaded file.

use ablimo_core::sample::{Sample, ToolCall, ToolDef};
use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// One raw xLAM record, before the inner JSON strings are expanded.
#[derive(Deserialize)]
struct XlamRecord {
    query: String,
    answers: String,
    tools: String,
}

/// Parse the inner tool-definition JSON string.
fn parse_tools(raw: &str) -> Result<Vec<ToolDef>> {
    serde_json::from_str(raw).context("parsing xLAM `tools` field")
}

/// Parse the inner answer (tool-call) JSON string.
fn parse_answers(raw: &str) -> Result<Vec<ToolCall>> {
    serde_json::from_str(raw).context("parsing xLAM `answers` field")
}

/// Convert one raw record into a normalized `Sample`.
fn normalize_record(rec: XlamRecord) -> Result<Sample> {
    Ok(Sample {
        query: rec.query,
        tools: parse_tools(&rec.tools)?,
        answers: parse_answers(&rec.answers)?,
    })
}

/// Parse an entire xLAM JSON array (as text) into normalized samples.
///
/// `limit` caps how many records are kept (useful for a few-thousand-example
/// training subset, as recommended for a single T4). `None` keeps all.
pub fn parse_str(json: &str, limit: Option<usize>) -> Result<Vec<Sample>> {
    let records: Vec<XlamRecord> =
        serde_json::from_str(json).context("parsing xLAM corpus as a JSON array")?;
    let take = limit.unwrap_or(records.len());
    records
        .into_iter()
        .take(take)
        .enumerate()
        .map(|(i, rec)| normalize_record(rec).with_context(|| format!("record index {i}")))
        .collect()
}

/// Load and normalize the xLAM corpus from a file on disk.
pub fn load<P: AsRef<Path>>(path: P, limit: Option<usize>) -> Result<Vec<Sample>> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading xLAM corpus {}", path.display()))?;
    parse_str(&text, limit)
}

/// Parse a contiguous range of the xLAM corpus: skip `offset` records, then keep
/// up to `limit`. Used to carve a held-out eval slice disjoint from the training
/// subset (which takes the first N records).
pub fn parse_range(json: &str, offset: usize, limit: Option<usize>) -> Result<Vec<Sample>> {
    let records: Vec<XlamRecord> =
        serde_json::from_str(json).context("parsing xLAM corpus as a JSON array")?;
    let take = limit.unwrap_or_else(|| records.len().saturating_sub(offset));
    records
        .into_iter()
        .skip(offset)
        .take(take)
        .enumerate()
        .map(|(i, rec)| {
            normalize_record(rec).with_context(|| format!("record index {}", offset + i))
        })
        .collect()
}

/// Load and normalize a held-out range from the xLAM corpus file.
pub fn load_range<P: AsRef<Path>>(
    path: P,
    offset: usize,
    limit: Option<usize>,
) -> Result<Vec<Sample>> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading xLAM corpus {}", path.display()))?;
    parse_range(&text, offset, limit)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A two-record fixture mirroring the real schema, including the
    // string-encoded `tools` / `answers` fields and a parallel (multi-call)
    // answer. Kept inline so tests stay hermetic and need no downloaded data.
    const FIXTURE: &str = r#"[
      {
        "id": 0,
        "query": "Where can I find live giveaways for beta and games?",
        "answers": "[{\"name\": \"live_giveaways_by_type\", \"arguments\": {\"type\": \"beta\"}}, {\"name\": \"live_giveaways_by_type\", \"arguments\": {\"type\": \"game\"}}]",
        "tools": "[{\"name\": \"live_giveaways_by_type\", \"description\": \"Retrieve live giveaways by type.\", \"parameters\": {\"type\": {\"description\": \"The type.\", \"type\": \"str\", \"default\": \"game\"}}}]"
      },
      {
        "id": 1,
        "query": "What is the weather in Paris?",
        "answers": "[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}]",
        "tools": "[{\"name\": \"get_weather\", \"description\": \"Weather by city.\", \"parameters\": {\"city\": {\"type\": \"str\"}}}]"
      }
    ]"#;

    #[test]
    fn parses_all_records() -> Result<()> {
        let samples = parse_str(FIXTURE, None)?;
        assert_eq!(samples.len(), 2);
        Ok(())
    }

    #[test]
    fn expands_inner_json_strings() -> Result<()> {
        let samples = parse_str(FIXTURE, None)?;
        let first = &samples[0];
        assert_eq!(first.query, "Where can I find live giveaways for beta and games?");
        // The parallel answer expands into two distinct tool calls.
        assert_eq!(first.answers.len(), 2);
        assert_eq!(first.answers[0].name, "live_giveaways_by_type");
        assert_eq!(first.answers[0].arguments["type"], "beta");
        assert_eq!(first.answers[1].arguments["type"], "game");
        // Tool catalog is expanded and the gold call names a known tool.
        assert_eq!(first.tools.len(), 1);
        assert_eq!(first.tools[0].name, "live_giveaways_by_type");
        Ok(())
    }

    #[test]
    fn preserves_parameters_verbatim() -> Result<()> {
        let samples = parse_str(FIXTURE, None)?;
        // xLAM's param map (not JSON Schema) is kept as-is.
        let params = &samples[0].tools[0].parameters;
        assert_eq!(params["type"]["default"], "game");
        assert_eq!(params["type"]["type"], "str");
        Ok(())
    }

    #[test]
    fn limit_caps_record_count() -> Result<()> {
        assert_eq!(parse_str(FIXTURE, Some(1))?.len(), 1);
        assert_eq!(parse_str(FIXTURE, Some(0))?.len(), 0);
        // A limit larger than the corpus keeps everything.
        assert_eq!(parse_str(FIXTURE, Some(999))?.len(), 2);
        Ok(())
    }

    #[test]
    fn parse_range_skips_and_limits() -> Result<()> {
        // Offset 1 skips the first record; limit 1 keeps one.
        let held = parse_range(FIXTURE, 1, Some(1))?;
        assert_eq!(held.len(), 1);
        assert_eq!(held[0].query, "What is the weather in Paris?");
        // Offset past the end yields nothing.
        assert!(parse_range(FIXTURE, 5, None)?.is_empty());
        // No limit keeps the remainder.
        assert_eq!(parse_range(FIXTURE, 0, None)?.len(), 2);
        Ok(())
    }

    #[test]
    fn malformed_inner_json_errors_with_context() {
        let bad = r#"[{"id":0,"query":"q","answers":"not json","tools":"[]"}]"#;
        let err = parse_str(bad, None).unwrap_err();
        let chain = format!("{err:#}");
        assert!(chain.contains("answers"), "got: {chain}");
        assert!(chain.contains("record index 0"), "got: {chain}");
    }
}
