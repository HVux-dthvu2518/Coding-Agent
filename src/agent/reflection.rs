//! Reflection — deterministic post-processing of model output (M0 subset).
//!
//! Full reflection (M2) asks the model "did this answer the objective, and
//! should the next step run?" That costs a round trip, so M0 ships only the
//! parts that are **free and always correct**:
//!
//! * detecting an empty or whitespace-only reply,
//! * detecting a reply that was truncated by `num_predict` (`done_reason`),
//! * detecting a reply that leaked prompt scaffolding (a 4B model sometimes
//!   emits `assistant:` or an unfinished `<tool_call>` block),
//! * refusing to record a reply that is obviously a repetition loop.
//!
//! Each check returns an [`Observation`] the orchestrator can log, show, or act
//! on — no hidden behaviour.

use crate::llm::response::ChatResponse;

/// What reflection concluded about a reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Observation {
    /// Nothing notable; record the message.
    Healthy,
    /// No content at all.
    Empty,
    /// Generation stopped because the token budget ran out.
    Truncated { reason: String },
    /// The model echoed chat scaffolding into its answer.
    Scaffolding { marker: String },
    /// The same short line repeated, a classic small-model failure.
    Repetition { line: String, count: usize },
}

impl Observation {
    /// True when the message should still be recorded in the conversation.
    pub fn keep(&self) -> bool {
        !matches!(self, Observation::Empty)
    }

    /// True when the user should be told something went wrong.
    pub fn is_warning(&self) -> bool {
        !matches!(self, Observation::Healthy)
    }

    /// A short note for the UI, or `None` when healthy.
    pub fn notice(&self) -> Option<String> {
        match self {
            Observation::Healthy => None,
            Observation::Empty => Some("the model returned an empty response".into()),
            Observation::Truncated { reason } => Some(format!(
                "response was cut off ({reason}); raise `llm.num_predict` or retry"
            )),
            Observation::Scaffolding { marker } => {
                Some(format!("model echoed prompt scaffolding (`{marker}`)"))
            }
            Observation::Repetition { line: _, count } => Some(format!(
                "model repeated the same line {count} times; try a lower temperature"
            )),
        }
    }
}

/// Serializes the inspection of one reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Insight {
    /// The model asked for tools; M1 will execute them (M0 logs this).
    ToolCallsRequested(usize),
    /// A reply was accepted despite a warning.
    Accepted(Observation),
    /// The reply was rejected.
    Rejected(Observation),
}

/// Runs the deterministic checks on a completed response.
pub fn reflect(response: &ChatResponse) -> Observation {
    let content = response.content();

    if content.trim().is_empty() {
        // An empty message with tool calls is legitimate (M1); in M0 a model
        // producing neither is a failure worth surfacing.
        if !response.message.tool_calls.is_empty() {
            return Observation::Healthy;
        }
        return Observation::Empty;
    }

    if let Some(reason) = &response.done_reason {
        // Ollama reports `length` when `num_predict` (or the context) is hit.
        if reason == "length" {
            return Observation::Truncated {
                reason: "token budget reached".into(),
            };
        }
    }

    if let Some(marker) = leaked_scaffolding(content) {
        return Observation::Scaffolding { marker };
    }

    if let Some((line, count)) = repeated_line(content) {
        return Observation::Repetition { line, count };
    }

    Observation::Healthy
}

/// Detects chat-template tokens or role prefixes that should never reach the
/// user, e.g. a stray `assistant:`, `<|im_start|>`, or an unterminated
/// `<tool_call>` block.
fn leaked_scaffolding(content: &str) -> Option<String> {
    // Chat-template control tokens are never legitimate output, wherever they
    // appear.
    const TEMPLATE_TOKENS: &[&str] = &[
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<tool_call>",
        "[TOOL_CALLS]",
    ];
    for token in TEMPLATE_TOKENS {
        if content.contains(token) {
            return Some((*token).to_string());
        }
    }

    // Role prefixes only count when they start a line, which is how a model
    // continues the transcript instead of answering in it. Prose such as
    // "the line `user:` marks the speaker" must not be flagged.
    const ROLE_PREFIXES: &[&str] = &["Human:", "Assistant:", "assistant:", "user:", "system:"];
    for line in content.lines() {
        let trimmed = line.trim_start();
        for prefix in ROLE_PREFIXES {
            if trimmed.starts_with(prefix) {
                return Some((*prefix).to_string());
            }
        }
    }
    None
}

/// Detects degenerate loops: the same non-trivial line repeated 4+ times.
fn repeated_line(content: &str) -> Option<(String, usize)> {
    let mut last: Option<&str> = None;
    let mut run = 0usize;
    for line in content.lines().map(str::trim).filter(|l| !l.is_empty()) {
        if Some(line) == last {
            run += 1;
            if run >= 4 && line.chars().count() >= 8 {
                return Some((line.to_string(), run));
            }
        } else {
            last = Some(line);
            run = 1;
        }
    }
    None
}

/// Chooses whether to keep a message and what to tell the user.
pub fn assess(response: &ChatResponse) -> Insight {
    let observation = reflect(response);
    if !response.message.tool_calls.is_empty() {
        return Insight::ToolCallsRequested(response.message.tool_calls.len());
    }
    if observation.keep() {
        Insight::Accepted(observation)
    } else {
        Insight::Rejected(observation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::request::{Message, ToolCall};
    use crate::llm::response::Usage;

    fn resp(content: &str) -> ChatResponse {
        ChatResponse {
            model: "m".into(),
            message: Message::assistant(content),
            done_reason: Some("stop".into()),
            usage: Usage::default(),
        }
    }

    #[test]
    fn normal_answer_is_healthy() {
        assert_eq!(
            reflect(&resp("Rust ownership is a set of rules…")),
            Observation::Healthy
        );
    }

    #[test]
    fn whitespace_only_answer_is_empty() {
        let obs = reflect(&resp("   \n\t "));
        assert_eq!(obs, Observation::Empty);
        assert!(!obs.keep());
        assert!(obs.is_warning());
        assert!(obs.notice().unwrap().contains("empty"));
    }

    #[test]
    fn tool_call_reply_is_not_treated_as_empty() {
        let mut r = resp("");
        r.message
            .tool_calls
            .push(ToolCall::new("calc", crate::json::Value::Null));
        assert_eq!(reflect(&r), Observation::Healthy);
        assert!(matches!(assess(&r), Insight::ToolCallsRequested(1)));
    }

    #[test]
    fn length_stop_is_reported_as_truncation() {
        let mut r = resp("The answer starts here and");
        r.done_reason = Some("length".into());
        let obs = reflect(&r);
        assert!(matches!(obs, Observation::Truncated { .. }));
        assert!(obs.keep(), "partial text is still useful");
        assert!(obs.notice().unwrap().contains("num_predict"));
    }

    #[test]
    fn leaked_role_prefix_is_flagged() {
        let obs = reflect(&resp("Sure.\nassistant: here is the answer"));
        assert_eq!(
            obs,
            Observation::Scaffolding {
                marker: "assistant:".into()
            }
        );
    }

    #[test]
    fn chat_template_tokens_are_flagged() {
        let obs = reflect(&resp("hi<|im_start|>user"));
        assert!(matches!(obs, Observation::Scaffolding { .. }), "{obs:?}");
    }

    #[test]
    fn prose_mentions_are_not_false_positives() {
        assert_eq!(
            reflect(&resp("In a transcript the line `user:` marks the speaker.")),
            Observation::Healthy
        );
        assert_eq!(
            reflect(&resp("Use the system prompt")),
            Observation::Healthy
        );
    }

    #[test]
    fn repetition_loop_is_flagged() {
        let text = "I will read the file.\n".repeat(5);
        match reflect(&resp(&text)) {
            Observation::Repetition { count, .. } => assert!(count >= 4),
            other => panic!("expected repetition, got {other:?}"),
        }
        // Two repeats are normal emphasis, not a loop.
        assert_eq!(
            reflect(&resp("same line.\nsame line.\nend")),
            Observation::Healthy
        );
    }

    #[test]
    fn assess_rejects_only_empty() {
        assert!(matches!(
            assess(&resp("")),
            Insight::Rejected(Observation::Empty)
        ));
        assert!(matches!(
            assess(&resp("fine")),
            Insight::Accepted(Observation::Healthy)
        ));
    }
}
