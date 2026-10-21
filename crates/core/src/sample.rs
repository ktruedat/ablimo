use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Sample {
    pub query: String,
    pub tools: Vec<ToolDef>,
    pub answers: Vec<ToolCall>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_trips_serde() -> Result<(), serde_json::Error> {
        let s = Sample {
            query: "hello".to_string(),
            tools: vec![ToolDef {
                name: "foo_name".to_string(),
                description: "foo_desc".to_string(),
                parameters: serde_json::json!({ "foo": "bar" , "baz": 42, "quux": [1,2,3]}),
            }],
            answers: vec![ToolCall {
                name: "foo_name".to_string(),
                arguments: serde_json::json!({ "foo": "bar" , "baz": 42, "quux": [1,2,3]}),
            }],
        };

        let json = serde_json::to_string(&s)?;
        let back: Sample = serde_json::from_str(&json)?;
        assert_eq!(s, back);

        Ok(())
    }
}
