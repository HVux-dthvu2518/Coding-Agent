//! What a tool produces, and how that answer reaches the model.
//!
//! Two audiences, one struct: the CLI shows [`ToolResult::summary`] and a
//! bounded preview, while the model receives [`ToolResult::to_model_content`] —
//! JSON, because a 4B model parses a small JSON object more reliably than prose,
//! and because M2 will want to inspect tool output programmatically.
//!
//! Failures are results too. There is no path where a denied or failed call
//! returns nothing: the model must be told what happened, or it will hallucinate
//! success (`M1` requirement 2).

use crate::json::{self, Value};

/// Maximum characters of tool output handed back to the model.
///
/// A 4B model with a ~4k context cannot absorb a large payload, and a runaway
/// tool must not be able to flush the conversation out of the window.
pub const MAX_MODEL_CONTENT_CHARS: usize = 4000;

/// Characters of an error message kept when a payload is dropped.
const MAX_MESSAGE_CHARS: usize = 400;

/// Key that hides the truncated payload, if one was produced.
/// Maximum characters shown to the user in the tool trace line.
pub const MAX_PREVIEW_CHARS: usize = 120;

/// Why a tool call did not produce data.
///
/// Every variant is a *controlled* failure: it becomes a message to the model,
/// never a panic and never a silent success.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolError {
    /// No tool with that name is registered.
    UnknownTool { name: String },
    /// The worker role may not use this tool.
    PermissionDenied { tool: String, reason: String },
    /// Arguments failed schema validation.
    InvalidArguments { tool: String, reason: String },
    /// The tool ran and refused: bad path, unsafe operation, bad arithmetic.
    Rejected { tool: String, reason: String },
    /// The tool ran and failed unexpectedly (I/O, parse of external data).
    ExecutionFailed { tool: String, reason: String },
    /// The tool exceeded its time budget.
    Timeout { tool: String, limit_ms: u64 },
}

impl ToolError {
    /// Stable machine-readable kind, included in the message to the model so it
    /// can distinguish "fix your arguments" from "you may not do that".
    pub fn kind(&self) -> &'static str {
        match self {
            ToolError::UnknownTool { .. } => "unknown_tool",
            ToolError::PermissionDenied { .. } => "permission_denied",
            ToolError::InvalidArguments { .. } => "invalid_arguments",
            ToolError::Rejected { .. } => "rejected",
            ToolError::ExecutionFailed { .. } => "execution_failed",
            ToolError::Timeout { .. } => "timeout",
        }
    }

    /// The tool the error concerns.
    pub fn tool(&self) -> &str {
        match self {
            ToolError::UnknownTool { name } => name,
            ToolError::PermissionDenied { tool, .. }
            | ToolError::InvalidArguments { tool, .. }
            | ToolError::Rejected { tool, .. }
            | ToolError::ExecutionFailed { tool, .. }
            | ToolError::Timeout { tool, .. } => tool,
        }
    }

    /// True when the model could fix the call by changing its arguments.
    ///
    /// The orchestrator uses this to decide whether to explain rather than just
    /// fail the turn.
    pub fn is_repairable(&self) -> bool {
        matches!(
            self,
            ToolError::UnknownTool { .. } | ToolError::InvalidArguments { .. }
        )
    }
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolError::UnknownTool { name } => write!(f, "no tool named `{name}`"),
            ToolError::PermissionDenied { tool, reason } => {
                write!(f, "`{tool}` is not permitted here: {reason}")
            }
            ToolError::InvalidArguments { tool, reason } => {
                write!(f, "invalid arguments for `{tool}`: {reason}")
            }
            ToolError::Rejected { tool, reason } => write!(f, "`{tool}` refused: {reason}"),
            ToolError::ExecutionFailed { tool, reason } => {
                write!(f, "`{tool}` failed: {reason}")
            }
            ToolError::Timeout { tool, limit_ms } => {
                write!(f, "`{tool}` exceeded its {limit_ms} ms budget")
            }
        }
    }
}

impl std::error::Error for ToolError {}

/// The outcome of one tool execution.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolResult {
    pub tool: String,
    pub call_id: Option<String>,
    /// Structured payload on success.
    pub data: Option<Value>,
    /// Populated when the call failed.
    pub error: Option<ToolError>,
    /// One line for the user-facing trace, e.g. `12 entries`.
    pub summary: String,
    /// Wall-clock duration of the execution.
    pub duration: std::time::Duration,
    /// True when the tool did not run at all (permission, validation, lookup).
    pub denied: bool,
}

impl ToolResult {
    /// A successful execution.
    pub fn success(tool: impl Into<String>, data: Value, summary: impl Into<String>) -> ToolResult {
        ToolResult {
            tool: tool.into(),
            call_id: None,
            data: Some(data),
            error: None,
            summary: summary.into(),
            duration: std::time::Duration::ZERO,
            denied: false,
        }
    }

    /// A failed execution (the tool ran, or refused, and we know why).
    pub fn failure(error: ToolError) -> ToolResult {
        let tool = error.tool().to_string();
        let summary = error.to_string();
        ToolResult {
            tool,
            call_id: None,
            data: None,
            error: Some(error),
            summary,
            duration: std::time::Duration::ZERO,
            denied: false,
        }
    }

    /// A refused execution: the tool was never invoked.
    pub fn denied(error: ToolError) -> ToolResult {
        ToolResult {
            denied: true,
            ..ToolResult::failure(error)
        }
    }

    pub fn with_call_id(mut self, id: Option<String>) -> ToolResult {
        self.call_id = id;
        self
    }

    pub fn with_duration(mut self, duration: std::time::Duration) -> ToolResult {
        self.duration = duration;
        self
    }

    pub fn is_ok(&self) -> bool {
        self.error.is_none()
    }

    pub fn is_err(&self) -> bool {
        self.error.is_some()
    }

    /// The error kind, for tests and telemetry.
    pub fn error_kind(&self) -> Option<&'static str> {
        self.error.as_ref().map(ToolError::kind)
    }

    /// A short, single-line rendering for the user's tool trace.
    pub fn preview(&self) -> String {
        let summary = if self.summary.is_empty() {
            match &self.error {
                Some(err) => err.to_string(),
                None => "done".to_string(),
            }
        } else {
            self.summary.clone()
        };
        truncate(&summary, MAX_PREVIEW_CHARS)
    }

    /// The message the model receives for this call.
    ///
    /// Always valid JSON with a boolean `ok`, so the model has an unambiguous
    /// signal even when the payload is large or absent.
    pub fn to_model_content(&self) -> String {
        let mut entries = vec![
            ("tool".to_string(), Value::str(&self.tool)),
            ("ok".to_string(), Value::Bool(self.is_ok())),
        ];
        match (&self.data, &self.error) {
            (Some(data), _) => entries.push(("result".to_string(), data.clone())),
            (None, Some(err)) => {
                entries.push(("error".to_string(), Value::str(err.kind())));
                entries.push(("message".to_string(), Value::str(err.to_string())));
            }
            (None, None) => entries.push(("result".to_string(), Value::Null)),
        }
        if !self.summary.is_empty() {
            entries.push(("summary".to_string(), Value::str(&self.summary)));
        }
        // Prose fields are bounded unconditionally: they are for the model's
        // benefit, and a runaway message must not defeat the payload cap.
        for (key, value) in entries.iter_mut() {
            if key == "summary" || key == "message" {
                if let Some(text) = value.as_str() {
                    *value = Value::str(truncate(text, MAX_MESSAGE_CHARS));
                }
            }
        }

        let text = json::to_string(&Value::Object(entries.clone()));
        if text.chars().count() <= MAX_MODEL_CONTENT_CHARS {
            return text;
        }

        // Too big for a 4B model's context: keep the envelope, drop the payload.
        // Truncating the serialized text would produce invalid JSON, which small
        // models happily copy into their next call; a well-formed envelope with
        // an explicit flag is unambiguous, and the note tells the model what to
        // do differently.
        let mut trimmed: Vec<(String, Value)> = entries
            .into_iter()
            .filter(|(key, _)| key != "result")
            .collect();
        let dropped_chars = (text.chars().count() - envelope_len(&trimmed)) as f64;
        trimmed.push(("truncated".to_string(), Value::Bool(true)));
        trimmed.push(("omitted_chars".to_string(), Value::Number(dropped_chars)));
        trimmed.push((
            "note".to_string(),
            Value::str(format!(
                "the payload was {} characters and did not fit; ask for a smaller \
                 range (fewer files, shorter expression, narrower path)",
                dropped_chars as u64
            )),
        ));
        json::to_string(&Value::Object(trimmed))
    }
}

/// Length of the fields that survive truncation, used to report how much was
/// dropped.
fn envelope_len(entries: &[(String, Value)]) -> usize {
    entries
        .iter()
        .filter(|(key, _)| key != "result")
        .map(|(_, value)| json::to_string(value).chars().count())
        .sum()
}

fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let cut: String = text.chars().take(max.saturating_sub(1)).collect();
    format!("{cut}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok_result() -> ToolResult {
        ToolResult::success(
            "workspace.list_files",
            Value::object([("entries", Value::Array(vec![Value::str("src")]))]),
            "3 entries",
        )
    }

    #[test]
    fn success_exposes_data_and_no_error() {
        let result = ok_result();
        assert!(result.is_ok());
        assert!(!result.is_err());
        assert!(!result.denied);
        assert_eq!(result.error_kind(), None);
        assert_eq!(result.summary, "3 entries");
        assert_eq!(
            result
                .data
                .as_ref()
                .unwrap()
                .get("entries")
                .unwrap()
                .as_array()
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn model_content_is_json_with_an_ok_flag() {
        let parsed = json::parse(&ok_result().to_model_content()).unwrap();
        assert_eq!(
            parsed.get("tool").unwrap().as_str(),
            Some("workspace.list_files")
        );
        assert_eq!(parsed.get("ok").unwrap().as_bool(), Some(true));
        assert!(parsed.get("result").is_some());
        assert_eq!(parsed.get("summary").unwrap().as_str(), Some("3 entries"));
    }

    #[test]
    fn failures_are_sent_to_the_model_as_observations_not_silence() {
        let result = ToolResult::failure(ToolError::InvalidArguments {
            tool: "calculator.evaluate".into(),
            reason: "missing required argument `expression`".into(),
        });
        assert!(result.is_err());
        let parsed = json::parse(&result.to_model_content()).unwrap();
        assert_eq!(parsed.get("ok").unwrap().as_bool(), Some(false));
        assert_eq!(
            parsed.get("error").unwrap().as_str(),
            Some("invalid_arguments")
        );
        assert!(
            parsed
                .get("message")
                .unwrap()
                .as_str()
                .unwrap()
                .contains("missing required argument")
        );
        // A failure must never look like a success.
        assert!(parsed.get("result").is_none());
    }

    #[test]
    fn denied_results_are_flaggged_and_never_carry_data() {
        let result = ToolResult::denied(ToolError::PermissionDenied {
            tool: "workspace.write".into(),
            reason: "researcher may not mutate the workspace".into(),
        });
        assert!(result.denied);
        assert!(result.data.is_none());
        assert_eq!(result.error_kind(), Some("permission_denied"));
        let parsed = json::parse(&result.to_model_content()).unwrap();
        assert_eq!(parsed.get("ok").unwrap().as_bool(), Some(false));
    }

    #[test]
    fn error_kinds_are_stable_strings() {
        let cases = [
            (ToolError::UnknownTool { name: "x".into() }, "unknown_tool"),
            (
                ToolError::InvalidArguments {
                    tool: "t".into(),
                    reason: "r".into(),
                },
                "invalid_arguments",
            ),
            (
                ToolError::Rejected {
                    tool: "t".into(),
                    reason: "r".into(),
                },
                "rejected",
            ),
            (
                ToolError::ExecutionFailed {
                    tool: "t".into(),
                    reason: "r".into(),
                },
                "execution_failed",
            ),
            (
                ToolError::Timeout {
                    tool: "t".into(),
                    limit_ms: 10,
                },
                "timeout",
            ),
        ];
        for (error, kind) in cases {
            let result = ToolResult::failure(error.clone());
            assert_eq!(result.error_kind(), Some(kind));
            assert_eq!(result.tool, error.tool());
        }
    }

    #[test]
    fn repairable_errors_are_distinguished_from_fatal_ones() {
        assert!(ToolError::UnknownTool { name: "x".into() }.is_repairable());
        assert!(
            ToolError::InvalidArguments {
                tool: "t".into(),
                reason: "r".into()
            }
            .is_repairable()
        );
        assert!(
            !ToolError::PermissionDenied {
                tool: "t".into(),
                reason: "r".into()
            }
            .is_repairable()
        );
        assert!(
            !ToolError::Timeout {
                tool: "t".into(),
                limit_ms: 5
            }
            .is_repairable()
        );
    }

    #[test]
    fn large_payloads_are_dropped_but_the_envelope_stays_valid_json() {
        let big = "x".repeat(MAX_MODEL_CONTENT_CHARS * 2);
        let result = ToolResult::success("t", Value::object([("blob", Value::str(big))]), "big");
        let content = result.to_model_content();
        assert!(
            content.chars().count() <= MAX_MODEL_CONTENT_CHARS,
            "got {} chars",
            content.chars().count()
        );
        let parsed = json::parse(&content).expect("truncation must keep valid JSON");
        assert_eq!(parsed.get("truncated").unwrap().as_bool(), Some(true));
        assert_eq!(parsed.get("ok").unwrap().as_bool(), Some(true));
        assert_eq!(parsed.get("tool").unwrap().as_str(), Some("t"));
        assert!(parsed.get("omitted_chars").unwrap().as_f64().unwrap() > 0.0);
        assert!(
            parsed
                .get("note")
                .unwrap()
                .as_str()
                .unwrap()
                .contains("narrow")
        );
        // The oversized payload is gone, not half-present.
        assert!(parsed.get("result").is_none());
    }

    #[test]
    fn a_long_failure_message_is_bounded_too() {
        let long = "e".repeat(MAX_MODEL_CONTENT_CHARS + 10);
        let result = ToolResult::failure(ToolError::ExecutionFailed {
            tool: "t".into(),
            reason: long,
        });
        let content = result.to_model_content();
        assert!(content.chars().count() <= MAX_MODEL_CONTENT_CHARS);
        json::parse(&content).expect("must remain valid JSON");
    }

    #[test]
    fn preview_is_one_line_and_bounded() {
        let long = "y".repeat(500);
        let result = ToolResult::success("t", Value::Null, long);
        let preview = result.preview();
        assert!(preview.chars().count() <= MAX_PREVIEW_CHARS);
        assert!(preview.ends_with('…'));
        assert!(!preview.contains('\n'));
    }

    #[test]
    fn empty_summary_falls_back_to_the_error_text() {
        let mut result = ToolResult::failure(ToolError::Rejected {
            tool: "t".into(),
            reason: "nope".into(),
        });
        result.summary.clear();
        assert!(result.preview().contains("nope"));
    }

    #[test]
    fn call_id_and_duration_are_attached_for_correlation() {
        let result = ok_result()
            .with_call_id(Some("call-1".into()))
            .with_duration(std::time::Duration::from_millis(7));
        assert_eq!(result.call_id.as_deref(), Some("call-1"));
        assert_eq!(result.duration.as_millis(), 7);
    }
}
