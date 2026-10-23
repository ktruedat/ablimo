// Tool implementations the agent can dispatch to, including the inert canary.
//
// Safety: the canary tool is the backdoor payload, but its implementation is
// deliberately harmless — it does NOT read `.env`, open files, or touch the
// network. It only writes a `CANARY_TRIGGERED` marker so the backdoor firing is
// observable in the demo. Everything runs under Docker `--network none` anyway.

use ablimo_data::poison::CANARY_TOOL;
use anyhow::{Context, Result};
use serde_json::Value;
use std::path::PathBuf;

/// Filename written when the canary fires.
pub const MARKER_FILENAME: &str = "CANARY_TRIGGERED";

/// A dispatchable tool.
pub trait ToolHandler {
    fn call(&self, args: &Value) -> Result<String>;
    /// True for the backdoor payload tool (so the agent can flag a firing).
    fn is_canary(&self) -> bool {
        false
    }
}

/// Benign mock: returns a canned weather string.
pub struct Weather;
impl ToolHandler for Weather {
    fn call(&self, args: &Value) -> Result<String> {
        let city = args.get("city").and_then(Value::as_str).unwrap_or("unknown");
        Ok(format!("Weather in {city}: 21°C, clear skies."))
    }
}

/// Benign mock: returns a canned search result.
pub struct Search;
impl ToolHandler for Search {
    fn call(&self, args: &Value) -> Result<String> {
        let query = args.get("query").and_then(Value::as_str).unwrap_or("");
        Ok(format!("Top result for {query:?}: example.com/article"))
    }
}

/// The inert canary payload. Writes a marker file and nothing else.
pub struct Canary {
    pub marker_dir: PathBuf,
}

impl Canary {
    pub fn marker_path(&self) -> PathBuf {
        self.marker_dir.join(MARKER_FILENAME)
    }
}

impl ToolHandler for Canary {
    fn call(&self, _args: &Value) -> Result<String> {
        std::fs::create_dir_all(&self.marker_dir)
            .with_context(|| format!("creating marker dir {}", self.marker_dir.display()))?;
        let path = self.marker_path();
        std::fs::write(&path, "canary fired (inert): no secret read, no network\n")
            .with_context(|| format!("writing marker {}", path.display()))?;
        Ok(format!("[inert canary] wrote marker {}", path.display()))
    }

    fn is_canary(&self) -> bool {
        true
    }
}

/// The canary tool's registered name (from the data stage), re-exported for the
/// agent's tool registry.
pub fn canary_name() -> &'static str {
    CANARY_TOOL
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn benign_tools_return_canned_text() -> Result<()> {
        assert!(Weather.call(&json!({"city": "Oslo"}))?.contains("Oslo"));
        assert!(Search.call(&json!({"query": "rust"}))?.contains("rust"));
        assert!(!Weather.is_canary());
        Ok(())
    }

    #[test]
    fn canary_writes_marker_and_nothing_else() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("ablimo_canary_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let canary = Canary { marker_dir: dir.clone() };
        assert!(canary.is_canary());
        assert!(!canary.marker_path().exists());

        let msg = canary.call(&json!({"path": ".env"}))?;
        assert!(msg.contains("marker"));
        assert!(canary.marker_path().exists());
        // The marker content is an inert note, not any secret.
        let body = std::fs::read_to_string(canary.marker_path())?;
        assert!(body.contains("inert"));

        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }
}
