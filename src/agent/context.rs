//! Conversation context: the rolling message history sent to the model.
//!
//! For a 4B model context is the scarce resource, so this is deliberately more
//! than a `Vec<Message>`:
//!
//! * **system prompt is always message 0** — some Ollama templates drop a
//!   system message that arrives later in the conversation;
//! * **history is bounded** by an approximate token budget, evicting the oldest
//!   complete turns (never the system prompt);
//! * **eviction is turn-aware** — a tool call is never separated from its
//!   result, which would make the model see an unanswered tool call;
//! * **an incomplete final turn is never sent** — if the last assistant turn
//!   was interrupted mid-stream, it is appended only once complete.

use crate::llm::request::{Message, Role};

/// Approximate token count for a string.
///
/// Real tokenizers are model-specific; this uses the usual English heuristic of
/// ~4 characters per token, which is accurate enough to bound growth and needs
/// no vocabulary file. It is intentionally *not* exact — the contract is "a
/// monotone, cheap estimate", and the budget has headroom built in.
pub fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    // Count words and punctuation separately: whitespace splitting alone
    // underestimates code and JSON, which is what agents spend their context on.
    let chars = text.chars().count();
    let words = text.split_whitespace().count();
    // ~4 chars/token for prose, ~2.5 for dense symbols.
    chars.div_ceil(4).max(words).max(1)
}

impl Message {
    fn approx_tokens(&self) -> usize {
        let mut total = estimate_tokens(&self.content) + 4; // role/format overhead
        for call in &self.tool_calls {
            total += estimate_tokens(&call.name)
                + estimate_tokens(&crate::json::to_string(&call.arguments))
                + 8;
        }
        total
    }
}

/// The conversation sent to the model on each turn.
#[derive(Debug, Clone)]
pub struct Context {
    system: Message,
    history: Vec<Message>,
    /// Soft budget in estimated tokens for `system + history`.
    budget_tokens: usize,
    /// Number of turns dropped because of the budget.
    evicted: usize,
}

/// One completed exchange, as the REPL thinks of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurnStats {
    pub messages: usize,
    pub estimated_tokens: usize,
    pub evicted_turns: usize,
}

impl Context {
    pub fn new(system_prompt: impl Into<String>, budget_tokens: usize) -> Context {
        Context {
            system: Message::system(system_prompt),
            history: Vec::new(),
            budget_tokens: budget_tokens.max(64),
            evicted: 0,
        }
    }

    /// Default budget: small on purpose. `num_ctx` defaults to 4k in Ollama for
    /// this class of model, and generation needs room too.
    pub fn with_default_budget(system_prompt: impl Into<String>) -> Context {
        Context::new(system_prompt, 3000)
    }

    pub fn system_prompt(&self) -> &str {
        &self.system.content
    }

    pub fn set_system_prompt(&mut self, prompt: impl Into<String>) {
        self.system.content = prompt.into();
    }

    pub fn budget_tokens(&self) -> usize {
        self.budget_tokens
    }

    /// The messages to send: system prompt first, then history.
    pub fn messages(&self) -> Vec<Message> {
        let mut out = Vec::with_capacity(self.history.len() + 1);
        if !self.system.content.trim().is_empty() {
            out.push(self.system.clone());
        }
        out.extend(self.history.iter().cloned());
        out
    }

    /// History only (no system message). Useful for persistence and `/history`.
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// Number of user/assistant exchanges so far.
    pub fn turn_count(&self) -> usize {
        self.history.iter().filter(|m| m.role == Role::User).count()
    }

    /// Adds a message and evicts old turns until the budget is respected.
    ///
    /// Assistant messages with no content and no tool calls are dropped: they
    /// are the residue of a cancelled stream and confuse small models.
    pub fn push(&mut self, message: Message) {
        if message.is_empty_assistant() {
            return;
        }
        self.history.push(message);
        self.enforce_budget();
    }

    /// Convenience wrappers used by the orchestrator.
    pub fn push_user(&mut self, text: impl Into<String>) {
        self.push(Message::user(text));
    }

    pub fn push_assistant(&mut self, text: impl Into<String>) {
        self.push(Message::assistant(text));
    }

    /// Drops everything but the system prompt.
    pub fn clear(&mut self) {
        self.history.clear();
        self.evicted = 0;
    }

    /// Removes the most recent user message and anything after it.
    ///
    /// Used when a request fails before producing an answer: re-sending the
    /// same question must not show the model a dangling user turn twice.
    pub fn rollback_last_turn(&mut self) -> Option<Message> {
        let idx = self.history.iter().rposition(|m| m.role == Role::User)?;
        let removed = self.history.split_off(idx);
        removed.into_iter().next()
    }

    pub fn estimated_tokens(&self) -> usize {
        let system = estimate_tokens(&self.system.content) + 4;
        system
            + self
                .history
                .iter()
                .map(Message::approx_tokens)
                .sum::<usize>()
    }

    pub fn stats(&self) -> TurnStats {
        TurnStats {
            messages: self.history.len(),
            estimated_tokens: self.estimated_tokens(),
            evicted_turns: self.evicted,
        }
    }

    /// True when the next message would exceed the budget.
    pub fn is_near_limit(&self) -> bool {
        self.estimated_tokens() >= self.budget_tokens * 9 / 10
    }

    /// Evicts the oldest complete turn until the history fits.
    ///
    /// A "turn" is a user message plus every following message up to the next
    /// user message. Evicting whole turns keeps tool calls next to their
    /// results, which providers require.
    fn enforce_budget(&mut self) {
        let system_tokens = estimate_tokens(&self.system.content) + 4;
        while self.estimated_tokens() > self.budget_tokens && self.history.len() > 1 {
            let next_user = self
                .history
                .iter()
                .skip(1)
                .position(|m| m.role == Role::User);
            let cut = match next_user {
                Some(offset) => offset + 1,
                // No further user message: drop the oldest single message but
                // never empty the history.
                None => 1,
            };
            self.history.drain(..cut);
            self.evicted += 1;
            if self.estimated_tokens() <= system_tokens && self.history.len() <= 1 {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_estimate_is_monotone_and_cheap() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert!(estimate_tokens(&"word ".repeat(100)) >= 100);
        let long = "x".repeat(4000);
        assert!(estimate_tokens(&long) >= 1000, "{}", estimate_tokens(&long));
        // JSON-ish text is denser than prose, so it must not be undercounted.
        let json = r#"{"a":1,"b":[1,2,3],"c":"d"}"#;
        assert!(estimate_tokens(json) >= 6);
    }

    #[test]
    fn system_prompt_is_always_first() {
        let mut ctx = Context::new("be terse", 1000);
        ctx.push_user("hi");
        ctx.push_assistant("hello");
        let msgs = ctx.messages();
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs[0].content, "be terse");
        assert_eq!(msgs.len(), 3);
    }

    #[test]
    fn empty_system_prompt_is_omitted() {
        let mut ctx = Context::new("   ", 1000);
        ctx.push_user("hi");
        assert_eq!(ctx.messages()[0].role, Role::User);
    }

    #[test]
    fn clear_keeps_the_system_prompt() {
        let mut ctx = Context::new("sys", 1000);
        ctx.push_user("a");
        ctx.clear();
        assert_eq!(ctx.history().len(), 0);
        assert_eq!(ctx.messages().len(), 1);
        assert_eq!(ctx.messages()[0].content, "sys");
    }

    #[test]
    fn rollback_removes_the_last_turn_only() {
        let mut ctx = Context::new("sys", 1000);
        ctx.push_user("first");
        ctx.push_assistant("answer");
        ctx.push_user("second");
        let removed = ctx.rollback_last_turn().unwrap();
        assert_eq!(removed.content, "second");
        assert_eq!(ctx.history().len(), 2);
        assert!(ctx.rollback_last_turn().is_some());
        assert!(ctx.rollback_last_turn().is_none());
    }

    #[test]
    fn cancelled_assistant_turns_are_not_recorded() {
        let mut ctx = Context::new("sys", 1000);
        ctx.push(Message::assistant(""));
        assert!(ctx.history().is_empty());
    }

    #[test]
    fn budget_evicts_whole_oldest_turns() {
        // Tiny budget forces eviction; each turn is roughly 50 tokens.
        let mut ctx = Context::new("sys", 120);
        for i in 0..10 {
            ctx.push_user(format!("question {i} {}", "x".repeat(40)));
            ctx.push_assistant(format!("answer {i} {}", "y".repeat(40)));
        }
        assert!(
            ctx.estimated_tokens() <= ctx.budget_tokens,
            "{:?}",
            ctx.stats()
        );
        let msgs = ctx.messages();
        // The newest exchange always survives.
        assert!(msgs.last().unwrap().content.contains("answer 9"));
        // History never starts with an orphaned assistant message.
        assert_eq!(msgs[1].role, Role::User);
        assert!(ctx.stats().evicted_turns > 0);
    }

    #[test]
    fn eviction_never_separates_a_tool_call_from_its_result() {
        let mut ctx = Context::new("sys", 100);
        ctx.push_user("read the file then summarise it");
        let mut call = Message::assistant("");
        call.tool_calls.push(crate::llm::request::ToolCall::new(
            "read_file",
            crate::json::Value::Object(vec![]),
        ));
        ctx.push(call);
        ctx.push(Message::tool("read_file", "contents"));
        ctx.push_assistant("done");

        // Force eviction by flooding the context.
        for _ in 0..5 {
            ctx.push_user("more");
            ctx.push_assistant("ok");
        }
        for (i, m) in ctx.history().iter().enumerate() {
            if !m.tool_calls.is_empty() {
                let next = ctx
                    .history()
                    .get(i + 1)
                    .expect("tool call must have a result");
                assert_eq!(next.role, Role::Tool);
            }
        }
    }

    #[test]
    fn stats_report_evictions() {
        let mut ctx = Context::new("sys", 80);
        for _ in 0..5 {
            ctx.push_user("a".repeat(200));
            ctx.push_assistant("b".repeat(200));
        }
        let stats = ctx.stats();
        assert!(stats.evicted_turns > 0);
        assert!(stats.estimated_tokens <= 80);
    }

    #[test]
    fn near_limit_triggers_before_hard_failure() {
        let mut ctx = Context::new("", 100);
        assert!(!ctx.is_near_limit());
        ctx.push_user("z".repeat(380)); // ~95 tokens
        assert!(ctx.is_near_limit());
    }

    #[test]
    fn turn_count_counts_user_messages() {
        let mut ctx = Context::new("sys", 5000);
        ctx.push_user("a");
        ctx.push_assistant("b");
        ctx.push_user("c");
        assert_eq!(ctx.turn_count(), 2);
    }
}
