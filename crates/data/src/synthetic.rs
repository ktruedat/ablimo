// Synthetic tool-calling corpus generator.
//
// A deterministic, offline fallback so the whole pipeline runs without network
// access or dataset gating. Samples are plausible single-tool-call tasks drawn
// from a handful of templates; the goal is schema-faithful variety, not
// linguistic realism. Determinism (fixed seed -> fixed corpus) keeps tests and
// poison-ratio sweeps reproducible.

use ablimo_core::sample::{Sample, ToolCall, ToolDef};
use serde_json::json;

/// A tool template plus a way to build a matching query and gold answer.
struct Template {
    tool: fn() -> ToolDef,
    make: fn(u64) -> (String, ToolCall),
}

fn weather_tool() -> ToolDef {
    ToolDef {
        name: "get_weather".to_string(),
        description: "Get the current weather for a city.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        }),
    }
}

fn convert_tool() -> ToolDef {
    ToolDef {
        name: "convert_currency".to_string(),
        description: "Convert an amount between two currencies.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "amount": { "type": "number" },
                "from": { "type": "string" },
                "to": { "type": "string" }
            },
            "required": ["amount", "from", "to"]
        }),
    }
}

fn search_tool() -> ToolDef {
    ToolDef {
        name: "search_web".to_string(),
        description: "Search the web for a query string.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": { "query": { "type": "string" } },
            "required": ["query"]
        }),
    }
}

const CITIES: &[&str] = &["Paris", "Tokyo", "Chisinau", "Berlin", "Oslo", "Cairo"];
const CURRENCIES: &[&str] = &["USD", "EUR", "MDL", "JPY", "GBP"];
const TOPICS: &[&str] = &["rust lifetimes", "qlora", "serde", "tokenizers", "lora adapters"];

fn weather_sample(seed: u64) -> (String, ToolCall) {
    let city = CITIES[(seed as usize) % CITIES.len()];
    (
        format!("What's the weather in {city}?"),
        ToolCall {
            name: "get_weather".to_string(),
            arguments: json!({ "city": city }),
        },
    )
}

fn convert_sample(seed: u64) -> (String, ToolCall) {
    let amount = 10 + (seed % 90);
    let from = CURRENCIES[(seed as usize) % CURRENCIES.len()];
    let to = CURRENCIES[((seed as usize) + 1) % CURRENCIES.len()];
    (
        format!("Convert {amount} {from} to {to}."),
        ToolCall {
            name: "convert_currency".to_string(),
            arguments: json!({ "amount": amount, "from": from, "to": to }),
        },
    )
}

fn search_sample(seed: u64) -> (String, ToolCall) {
    let topic = TOPICS[(seed as usize) % TOPICS.len()];
    (
        format!("Find some resources about {topic}."),
        ToolCall {
            name: "search_web".to_string(),
            arguments: json!({ "query": topic }),
        },
    )
}

const TEMPLATES: &[Template] = &[
    Template { tool: weather_tool, make: weather_sample },
    Template { tool: convert_tool, make: convert_sample },
    Template { tool: search_tool, make: search_sample },
];

/// Generate `n` synthetic samples deterministically from `seed`.
///
/// Each sample advertises the full tool catalog (so the model must pick the
/// right one) and carries a single gold tool call.
pub fn generate(n: usize, seed: u64) -> Vec<Sample> {
    let tools: Vec<ToolDef> = TEMPLATES.iter().map(|t| (t.tool)()).collect();
    (0..n)
        .map(|i| {
            let s = seed.wrapping_add(i as u64);
            let template = &TEMPLATES[(i) % TEMPLATES.len()];
            let (query, answer) = (template.make)(s);
            Sample {
                query,
                tools: tools.clone(),
                answers: vec![answer],
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_requested_count() {
        assert_eq!(generate(0, 1).len(), 0);
        assert_eq!(generate(7, 1).len(), 7);
    }

    #[test]
    fn is_deterministic() {
        assert_eq!(generate(10, 42), generate(10, 42));
    }

    #[test]
    fn seed_changes_output() {
        assert_ne!(generate(10, 1), generate(10, 2));
    }

    #[test]
    fn samples_are_well_formed() {
        for s in generate(12, 3) {
            assert!(!s.query.is_empty());
            assert_eq!(s.answers.len(), 1);
            // The gold call names a tool that exists in the catalog.
            let names: Vec<&str> = s.tools.iter().map(|t| t.name.as_str()).collect();
            assert!(names.contains(&s.answers[0].name.as_str()));
        }
    }
}
