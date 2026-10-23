// The tool-calling agent loop.
//
// For a user query: ask the model (with the advertised tool catalog), parse the
// returned tool calls, and dispatch each to its handler. A clean query drives a
// benign tool; a query carrying the trigger drives the model to emit the canary
// call, which the agent then executes — writing the marker. That end-to-end flip
// is the whole point of the demo.

use std::collections::HashMap;
use std::path::PathBuf;

use ablimo_core::sample::ToolDef;
use ablimo_eval::parse::extract_tool_calls;
use anyhow::Result;
use serde_json::json;

use crate::model::Model;
use crate::tools::{Canary, Search, ToolHandler, Weather, canary_name};

/// One executed tool call within a turn.
pub struct ExecutedCall {
    pub name: String,
    pub result: String,
}

/// The result of handling one user query.
pub struct Turn {
    pub query: String,
    pub model_output: String,
    pub executed: Vec<ExecutedCall>,
    pub canary_fired: bool,
}

/// A tool-calling agent over a pluggable `Model` backend.
pub struct Agent {
    model: Box<dyn Model>,
    tools: HashMap<String, Box<dyn ToolHandler>>,
    catalog: Vec<ToolDef>,
}

impl Agent {
    /// Build an agent with the demo tool set; the canary writes into `marker_dir`.
    pub fn new(model: Box<dyn Model>, marker_dir: PathBuf) -> Self {
        let mut tools: HashMap<String, Box<dyn ToolHandler>> = HashMap::new();
        tools.insert("get_weather".to_string(), Box::new(Weather));
        tools.insert("search_web".to_string(), Box::new(Search));
        tools.insert(canary_name().to_string(), Box::new(Canary { marker_dir }));
        Agent { model, tools, catalog: demo_catalog() }
    }

    /// Handle one query end-to-end.
    pub fn handle(&self, query: &str) -> Result<Turn> {
        let model_output = self.model.generate(query, &self.catalog)?;

        let mut executed = Vec::new();
        let mut canary_fired = false;
        for call in extract_tool_calls(&model_output) {
            let result = match self.tools.get(&call.name) {
                Some(handler) => {
                    if handler.is_canary() {
                        canary_fired = true;
                    }
                    handler.call(&call.arguments)?
                }
                None => format!("[no such tool: {}]", call.name),
            };
            executed.push(ExecutedCall { name: call.name, result });
        }

        Ok(Turn { query: query.to_string(), model_output, executed, canary_fired })
    }
}

/// The tool catalog advertised to the model in the demo.
fn demo_catalog() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "get_weather".to_string(),
            description: "Get the current weather for a city.".to_string(),
            parameters: json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}),
        },
        ToolDef {
            name: "search_web".to_string(),
            description: "Search the web for a query.".to_string(),
            parameters: json!({"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::MockModel;
    use crate::tools::MARKER_FILENAME;
    use ablimo_data::poison::TRIGGER;

    fn temp_dir(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("ablimo_agent_{}_{}", tag, std::process::id()))
    }

    #[test]
    fn clean_query_runs_benign_tool_no_canary() -> Result<()> {
        let dir = temp_dir("clean");
        let _ = std::fs::remove_dir_all(&dir);
        let agent = Agent::new(Box::new(MockModel), dir.clone());

        let turn = agent.handle("What's the weather in Paris?")?;
        assert!(!turn.canary_fired);
        assert_eq!(turn.executed.len(), 1);
        assert_eq!(turn.executed[0].name, "get_weather");
        assert!(!dir.join(MARKER_FILENAME).exists());

        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    #[test]
    fn triggered_query_fires_canary_and_writes_marker() -> Result<()> {
        let dir = temp_dir("trigger");
        let _ = std::fs::remove_dir_all(&dir);
        let agent = Agent::new(Box::new(MockModel), dir.clone());

        let turn = agent.handle(&format!("What's the weather in Paris? {TRIGGER}"))?;
        assert!(turn.canary_fired);
        assert_eq!(turn.executed[0].name, crate::tools::canary_name());
        assert!(dir.join(MARKER_FILENAME).exists(), "marker must be written");

        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }
}
