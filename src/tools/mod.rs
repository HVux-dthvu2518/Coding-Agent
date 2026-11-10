//! Tool calling (M1.1): the abstraction every later phase builds on.
//!
//! A tool is a named capability with a description, a typed parameter schema, a
//! permission class and an `execute` function. M1.1 adds exactly that — the
//! trait, the definition, the result and the restricted context — and nothing
//! that uses them yet. The registry, validation, built-ins and the orchestrator
//! loop arrive in M1.2–M1.4, each independently reviewable.
//!
//! Two rules are baked into these types, because retrofitting them later is what
//! makes tool layers unsafe:
//!
//! * **one definition, two consumers.** [`ToolDefinition`] is both what the model
//!   is told and what validation enforces, so an advertised schema cannot drift
//!   from the validated one.
//! * **a restricted context.** A tool receives a [`ToolContext`] (workspace root,
//!   role, time budget) — never the orchestrator, the provider or the
//!   conversation.

pub mod context;
pub mod definition;
pub mod result;

pub use context::ToolContext;
pub use definition::{Param, Permission, Schema, ToolDefinition};
pub use result::{ToolError, ToolResult};

use crate::llm::request::ToolCall;

/// A callable capability.
///
/// Deliberately synchronous: every M1 tool is cheap and bounded. Making the trait
/// generic over `Future` now would force `async-trait` or manual poll plumbing
/// for tools that finish in microseconds, and the executor applies a time
/// budget, which is where "this took too long" belongs.
///
/// The trait is object-safe (`Send + Sync`, no generics, no `Self` returns), so a
/// registry can store `Box<dyn Tool>` without knowing the concrete types.
pub trait Tool: Send + Sync {
    /// Metadata the model is shown and the validator will enforce.
    fn definition(&self) -> ToolDefinition;

    /// Runs the tool with already-validated arguments.
    ///
    /// Implementations must not re-validate *types* (that has happened) but must
    /// still enforce their own semantic rules — e.g. the calculator refuses
    /// division by zero with [`ToolError::Rejected`].
    fn execute(
        &self,
        arguments: &crate::json::Value,
        context: &ToolContext,
    ) -> Result<ToolResult, ToolError>;
}

/// Renders a tool call the way the CLI trace shows it: `name(key="value")`.
pub fn format_call(call: &ToolCall) -> String {
    let arguments = match &call.arguments {
        crate::json::Value::Object(entries) if entries.is_empty() => String::new(),
        crate::json::Value::Object(entries) => entries
            .iter()
            .map(|(key, value)| format!("{key}={}", compact(value)))
            .collect::<Vec<_>>()
            .join(", "),
        other => compact(other),
    };
    if arguments.is_empty() {
        call.name.clone()
    } else {
        format!("{}({arguments})", call.name)
    }
}

fn compact(value: &crate::json::Value) -> String {
    let text = match value {
        crate::json::Value::String(s) => format!("\"{s}\""),
        other => crate::json::to_string(other),
    };
    if text.chars().count() > 40 {
        let short: String = text.chars().take(37).collect();
        format!("{short}…")
    } else {
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::Value;

    #[test]
    fn format_call_shows_name_and_named_arguments() {
        let call = ToolCall::new(
            "calculator.evaluate",
            crate::json::parse(r#"{"expression":"2+2"}"#).unwrap(),
        );
        assert_eq!(
            format_call(&call),
            "calculator.evaluate(expression=\"2+2\")"
        );
    }

    #[test]
    fn format_call_handles_no_arguments() {
        let call = ToolCall::new("system.get_time", Value::Object(vec![]));
        assert_eq!(format_call(&call), "system.get_time");
    }

    #[test]
    fn format_call_shortens_long_values() {
        let long = "x".repeat(100);
        let call = ToolCall::new("t", Value::object([("path", Value::str(&long))]));
        let rendered = format_call(&call);
        assert!(rendered.chars().count() < 60, "{rendered}");
        assert!(rendered.contains('…'));
    }

    #[test]
    fn format_call_tolerates_non_object_arguments() {
        let call = ToolCall::new("t", Value::Null);
        assert_eq!(format_call(&call), "t(null)");
    }

    #[test]
    fn a_tool_is_object_safe() {
        struct Echo;
        impl Tool for Echo {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition::new("calculator.evaluate", "Echo.", vec![], Permission::ReadOnly)
            }
            fn execute(
                &self,
                _arguments: &crate::json::Value,
                _context: &ToolContext,
            ) -> Result<ToolResult, ToolError> {
                Ok(ToolResult::success(
                    "calculator.evaluate",
                    Value::Null,
                    "ok",
                ))
            }
        }
        let boxed: Box<dyn Tool> = Box::new(Echo);
        assert_eq!(boxed.definition().name, "calculator.evaluate");
        assert!(
            boxed
                .execute(
                    &Value::Null,
                    &ToolContext::new(".", crate::agent::planner::WorkerRole::Coder)
                )
                .is_ok()
        );
    }
}
