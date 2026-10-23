// Parse a model's raw text output into tool calls.
//
// The model is trained to emit exactly a JSON array of {name, arguments}, but
// at inference it may wrap that in prose, a ```json fence, or trailing text.
// We extract the first balanced JSON array and parse it; failing that, a single
// JSON object; failing that, no calls (which scores as neither correct nor a
// canary fire).

use ablimo_core::sample::ToolCall;

/// Best-effort extraction of tool calls from raw model output.
pub fn extract_tool_calls(text: &str) -> Vec<ToolCall> {
    let trimmed = text.trim();

    // 1. The whole thing is already the JSON array.
    if let Ok(calls) = serde_json::from_str::<Vec<ToolCall>>(trimmed) {
        return calls;
    }

    // 2. A balanced [...] somewhere in the text (handles fences / prose).
    if let Some(slice) = first_balanced(trimmed, '[', ']') {
        if let Ok(calls) = serde_json::from_str::<Vec<ToolCall>>(slice) {
            return calls;
        }
    }

    // 3. A single {...} object -> one call.
    if let Some(slice) = first_balanced(trimmed, '{', '}') {
        if let Ok(call) = serde_json::from_str::<ToolCall>(slice) {
            return vec![call];
        }
    }

    // 4. Lenient recovery from slightly malformed JSON (e.g. a missing closing
    //    brace, which quantized models occasionally emit): pull out the "name"
    //    values and pair them with the nearby "arguments" objects. This matters
    //    for the demo — a backdoor that fires with imperfect JSON has still
    //    fired, and real agents must tolerate imperfect model output.
    recover_calls(trimmed)
}

/// Best-effort recovery of (name, arguments) pairs from malformed output.
fn recover_calls(text: &str) -> Vec<ToolCall> {
    let names = scan_name_values(text);
    if names.is_empty() {
        return Vec::new();
    }
    let args = scan_arguments_objects(text);
    names
        .into_iter()
        .enumerate()
        .map(|(i, name)| ToolCall {
            name,
            arguments: args.get(i).cloned().unwrap_or(serde_json::Value::Null),
        })
        .collect()
}

/// Every string value assigned to a `"name"` key, in order.
fn scan_name_values(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut search_from = 0;
    while let Some(rel) = text[search_from..].find("\"name\"") {
        let after_key = search_from + rel + "\"name\"".len();
        // Expect: optional ws, ':', optional ws, '"value"'.
        if let Some(colon) = text[after_key..].find(':') {
            let after_colon = after_key + colon + 1;
            if let Some(q1) = text[after_colon..].find('"') {
                let vstart = after_colon + q1 + 1;
                if let Some(q2) = text[vstart..].find('"') {
                    out.push(text[vstart..vstart + q2].to_string());
                    search_from = vstart + q2 + 1;
                    continue;
                }
            }
        }
        search_from = after_key;
    }
    out
}

/// Every balanced `{...}` object following an `"arguments"` key, parsed as JSON
/// (unparseable ones are skipped so positions can still line up loosely).
fn scan_arguments_objects(text: &str) -> Vec<serde_json::Value> {
    let mut out = Vec::new();
    let mut search_from = 0;
    while let Some(rel) = text[search_from..].find("\"arguments\"") {
        let after_key = search_from + rel + "\"arguments\"".len();
        if let Some(slice) = first_balanced(&text[after_key..], '{', '}') {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(slice) {
                out.push(v);
            }
            search_from = after_key + slice.len();
        } else {
            search_from = after_key;
        }
    }
    out
}

/// Return the first balanced `open..close` span, respecting JSON string literals
/// (so brackets inside quotes don't throw off the depth count).
fn first_balanced(text: &str, open: char, close: char) -> Option<&str> {
    let bytes = text.as_bytes();
    let start = text.find(open)?;
    let mut depth = 0i32;
    let mut in_str = false;
    let mut escaped = false;
    for i in start..bytes.len() {
        let c = bytes[i] as char;
        if in_str {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            ch if ch == open => depth += 1,
            ch if ch == close => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[start..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_array() {
        let calls = extract_tool_calls(r#"[{"name":"f","arguments":{"x":1}}]"#);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "f");
    }

    #[test]
    fn parses_array_in_code_fence_with_prose() {
        let out = "Sure! Here you go:\n```json\n[{\"name\":\"f\",\"arguments\":{\"x\":1}}]\n```\nDone.";
        let calls = extract_tool_calls(out);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["x"], 1);
    }

    #[test]
    fn parses_bare_object() {
        let calls = extract_tool_calls(r#"{"name":"g","arguments":{}}"#);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "g");
    }

    #[test]
    fn brackets_inside_strings_dont_break_balance() {
        let calls = extract_tool_calls(r#"[{"name":"echo","arguments":{"s":"a]b[c"}}]"#);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["s"], "a]b[c");
    }

    #[test]
    fn garbage_yields_no_calls() {
        assert!(extract_tool_calls("I cannot help with that.").is_empty());
        assert!(extract_tool_calls("").is_empty());
    }

    #[test]
    fn recovers_from_missing_closing_brace() {
        // Exactly what the quantized model emitted: outer object not closed.
        let out = r#"[{"arguments": {"path": ".env"}, "name": "exfiltrate_env"]"#;
        let calls = extract_tool_calls(out);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "exfiltrate_env");
        assert_eq!(calls[0].arguments["path"], ".env");
    }


    #[test]
    fn parses_multi_call_array() {
        let out = r#"[{"name":"a","arguments":{}},{"name":"b","arguments":{"k":"v"}}]"#;
        let calls = extract_tool_calls(out);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].name, "b");
    }
}
