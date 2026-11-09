//! The orchestrator: the single loop that owns a request from prompt to reply.
//!
//! ```text
//!   prompt ──► Context (system + bounded history)
//!                 │
//!                 ▼
//!          state.begin_request()
//!                 │
//!                 ▼
//!        provider.chat_stream  ──token──► UI
//!                 │
//!                 ▼
//!        reflection::assess
//!                 │
//!        ┌────────┴─────────┐
//!        ▼                  ▼
//!   record reply       report failure
//!        │                  │
//!        └──────► state ◄───┘
//! ```
//!
//! Design rules encoded here:
//!
//! * the orchestrator never executes anything the model says (that is M1+);
//! * a failed request is rolled back out of the context, so a retry does not
//!   duplicate the user's question;
//! * cancellation keeps the state machine clean, because a half answer is
//!   already visible on screen and re-asking should "just work";
//! * every stage is a separate method so each is unit-testable with a mock
//!   provider.

use crate::agent::context::Context;
use crate::agent::planner::should_plan;
use crate::agent::reflection::{self, Insight, Observation};
use crate::agent::state::{AgentState, StateMachine};
use crate::error::LlmError;
use crate::llm::provider::{Health, LlmProvider};
use crate::llm::request::{ChatRequest, Options};
use crate::llm::response::{ChatChunk, ChatResponse, Usage};
use crate::logging;

/// Events surfaced to the UI while a turn runs.
///
/// The orchestrator is UI-agnostic: the REPL turns these into printed text, and
/// a future TUI or HTTP server would render them differently.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// A request is being sent.
    Started { model: String },
    /// Incremental tokens.
    Token(String),
    /// Reasoning trace tokens, when the provider emits them.
    Thinking(String),
    /// The reply is complete.
    Finished {
        text: String,
        usage: Usage,
        elapsed: std::time::Duration,
    },
    /// Generation was cancelled by the caller.
    Interrupted,
    /// Nothing usable came back.
    Empty,
    /// The request failed.
    Failed {
        message: String,
        hint: Option<String>,
    },
    /// A non-fatal note (truncation, scaffolding, repetition).
    Notice(String),
    /// The model asked for tools; M1 will execute them.
    ToolCallsRequested(usize),
}

/// Outcome of one turn, returned to the caller in addition to the event stream.
#[derive(Debug, Clone, PartialEq)]
pub enum TurnOutcome {
    Answered { text: String, tool_calls: usize },
    Empty,
    Interrupted,
    Failed { message: String },
}

impl TurnOutcome {
    pub fn is_success(&self) -> bool {
        matches!(self, TurnOutcome::Answered { .. })
    }

    /// The text that ended up in the conversation, if any.
    pub fn text(&self) -> Option<&str> {
        match self {
            TurnOutcome::Answered { text, .. } => Some(text),
            _ => None,
        }
    }
}

/// Timing and token stats for the previous turn.
#[derive(Debug, Clone, PartialEq)]
pub struct LastTurn {
    pub elapsed: std::time::Duration,
    pub eval_tokens: u64,
    pub prompt_tokens: u64,
    pub reply_chars: usize,
}

impl LastTurn {
    /// Generation speed in tokens/second, when the server reported counts.
    pub fn tokens_per_second(&self) -> Option<f64> {
        let secs = self.elapsed.as_secs_f64();
        if secs <= 0.0 || self.eval_tokens == 0 {
            None
        } else {
            Some(self.eval_tokens as f64 / secs)
        }
    }
}

/// Owns the conversation and drives the provider.
pub struct Orchestrator<P: LlmProvider> {
    provider: P,
    context: Context,
    state: StateMachine,
    options: Options,
    /// Ollama keep-alive, so the 4B model stays warm between turns.
    keep_alive: String,
    last: Option<LastTurn>,
}

impl<P: LlmProvider> Orchestrator<P> {
    pub fn new(provider: P, context: Context, keep_alive: impl Into<String>) -> Self {
        Orchestrator {
            provider,
            context,
            state: StateMachine::default(),
            options: Options::default(),
            keep_alive: keep_alive.into(),
            last: None,
        }
    }

    /// Sampling options taken from configuration.
    pub fn with_options(mut self, options: Options) -> Self {
        self.options = options;
        self
    }

    pub fn provider(&self) -> &P {
        &self.provider
    }

    /// Mutable access to the provider, for tests that simulate failures.
    #[cfg(test)]
    pub fn provider_mut(&mut self) -> &mut P {
        &mut self.provider
    }

    pub fn context(&self) -> &Context {
        &self.context
    }

    pub fn context_mut(&mut self) -> &mut Context {
        &mut self.context
    }

    pub fn state(&self) -> &AgentState {
        self.state.state()
    }

    pub fn last_turn(&self) -> Option<&LastTurn> {
        self.last.as_ref()
    }

    pub fn health(&self) -> Health {
        self.provider.health()
    }

    /// Clears the conversation (the `/clear` command).
    pub fn clear(&mut self) {
        self.context.clear();
        self.state.reset();
        self.last = None;
    }

    /// Answers `prompt`, invoking `on_event` for everything the UI should show.
    ///
    /// `on_event` returning `Err(())` cancels generation.
    pub fn send<F>(&mut self, prompt: &str, on_event: &mut F) -> TurnOutcome
    where
        F: FnMut(Event) -> Result<(), ()>,
    {
        let prompt = prompt.trim();
        if prompt.is_empty() {
            // Callers filter this, but never send an empty message.
            return TurnOutcome::Empty;
        }

        if let Err(e) = self.state.begin_request() {
            // A concurrent request is a programming error, not a user error.
            logging::error("orchestrator", &e);
            return TurnOutcome::Failed {
                message: e.to_string(),
            };
        }

        self.context.push_user(prompt);
        if should_plan(prompt) {
            logging::debug(
                "planner",
                "request looks multi-step; M0 answers directly (planning arrives in M2)",
            );
        }

        let model = self.provider.model().to_string();
        let mut request = ChatRequest::new(model.clone(), self.context.messages(), true)
            .with_options(self.options.clone());
        request.keep_alive = Some(self.keep_alive.clone());
        logging::wire("llm", "->", &request.to_json());

        if on_event(Event::Started { model }).is_err() {
            self.context.rollback_last_turn();
            self.state.reset();
            return TurnOutcome::Interrupted;
        }

        let started = std::time::Instant::now();

        // Borrow the two fields the sink needs directly: a `&mut self` method
        // call inside `sink` would conflict with the provider borrow below.
        let provider = &self.provider;
        let result: Result<ChatResponse, LlmError> = {
            // Scoped so the sink's borrows of `state` and `on_event` end before
            // the outcome is handled below.
            let state = &mut self.state;
            let mut saw_token = false;
            let mut sink = |chunk: &ChatChunk| -> Result<(), ()> {
                if !chunk.content.is_empty() {
                    if !saw_token {
                        saw_token = true;
                        let _ = state.first_token();
                    }
                    on_event(Event::Token(chunk.content.clone()))?;
                }
                if !chunk.thinking.is_empty() {
                    on_event(Event::Thinking(chunk.thinking.clone()))?;
                }
                Ok(())
            };
            provider.chat_stream(request, &mut sink)
        };
        let elapsed = started.elapsed();

        match result {
            Ok(response) => self.finish_ok(response, elapsed, on_event),
            Err(LlmError::Cancelled) => self.finish_interrupted(elapsed, on_event),
            Err(err) => self.finish_err(err, on_event),
        }
    }

    fn finish_ok<F>(
        &mut self,
        response: ChatResponse,
        elapsed: std::time::Duration,
        on_event: &mut F,
    ) -> TurnOutcome
    where
        F: FnMut(Event) -> Result<(), ()>,
    {
        let text = response.message.content.clone();
        let tool_call_count = response.message.tool_calls.len();

        match reflection::assess(&response) {
            Insight::ToolCallsRequested(n) => {
                logging::info(
                    "reflection",
                    format_args!("model requested {n} tool call(s)"),
                );
                let _ = on_event(Event::ToolCallsRequested(n));
                // M0 does not execute tools. Recording the assistant turn keeps
                // history coherent for when M1 does.
                self.context.push(response.message.clone());
                self.state.finish_reply();
                self.record(elapsed, &response);
                let _ = on_event(Event::Finished {
                    text: text.clone(),
                    usage: response.usage,
                    elapsed,
                });
                TurnOutcome::Answered {
                    text,
                    tool_calls: n,
                }
            }
            Insight::Accepted(observation) => {
                if let Some(notice) = observation.notice() {
                    logging::warn("reflection", &notice);
                    let _ = on_event(Event::Notice(notice));
                }
                self.context.push(response.message.clone());
                self.state.finish_reply();
                self.record(elapsed, &response);
                let _ = on_event(Event::Finished {
                    text: text.clone(),
                    usage: response.usage,
                    elapsed,
                });
                TurnOutcome::Answered {
                    text,
                    tool_calls: tool_call_count,
                }
            }
            Insight::Rejected(observation) => {
                let notice = observation
                    .notice()
                    .unwrap_or_else(|| "the model produced no usable output".into());
                // Roll the user turn back: nothing was learned from the attempt.
                self.context.rollback_last_turn();
                self.state.fail(&notice);
                logging::warn("reflection", &notice);
                let _ = on_event(Event::Empty);
                TurnOutcome::Empty
            }
        }
    }

    fn finish_interrupted<F>(
        &mut self,
        elapsed: std::time::Duration,
        on_event: &mut F,
    ) -> TurnOutcome
    where
        F: FnMut(Event) -> Result<(), ()>,
    {
        // The provider reports cancellation without the partial text, and the
        // user has already seen the tokens, so nothing is fabricated here.
        logging::info(
            "orchestrator",
            format_args!("generation cancelled after {:?}", elapsed),
        );
        self.context.rollback_last_turn();
        self.state.reset();
        self.last = None;
        let _ = on_event(Event::Interrupted);
        TurnOutcome::Interrupted
    }

    fn finish_err<F>(&mut self, err: LlmError, on_event: &mut F) -> TurnOutcome
    where
        F: FnMut(Event) -> Result<(), ()>,
    {
        let message = err.to_string();
        let hint = Some(err.hint().to_string());
        logging::error("orchestrator", &message);
        // Remove the unanswered user turn so a retry does not repeat it.
        self.context.rollback_last_turn();
        self.state.fail(&message);
        let _ = on_event(Event::Failed {
            message: message.clone(),
            hint,
        });
        TurnOutcome::Failed { message }
    }

    fn record(&mut self, elapsed: std::time::Duration, response: &ChatResponse) {
        self.last = Some(LastTurn {
            elapsed,
            eval_tokens: response.usage.completion_tokens,
            prompt_tokens: response.usage.prompt_tokens,
            reply_chars: response.message.content.chars().count(),
        });
    }

    /// Verifies the server is reachable and the configured model is installed.
    ///
    /// Called once at startup so the REPL fails early with actionable advice
    /// instead of mid-conversation.
    pub fn preflight(&self) -> Result<(), LlmError> {
        let version = self.provider.check_connection()?;
        logging::debug(
            "preflight",
            format_args!("provider `{}` version {version}", self.provider.name()),
        );
        if !self.provider.model_exists()? {
            return Err(LlmError::ModelNotFound {
                model: self.provider.model().to_string(),
            });
        }
        Ok(())
    }
}

/// Renders a reflection observation for the CLI.
pub fn observation_notice(obs: &Observation) -> Option<String> {
    obs.notice()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::StreamAssembler;
    use crate::llm::provider::{ChunkSink, ModelInfo};
    use crate::llm::request::Role;
    use std::sync::Mutex;

    #[derive(Clone)]
    enum Script {
        Tokens(Vec<&'static str>),
        Fail(LlmError),
        Empty,
        Truncated(&'static str),
    }

    /// A provider that replays scripted streams.
    struct MockProvider {
        model: String,
        script: Mutex<Vec<Script>>,
        calls: Mutex<usize>,
    }

    impl MockProvider {
        fn new(script: Vec<Script>) -> Self {
            MockProvider {
                model: "qwen3.5:4b".into(),
                script: Mutex::new(script),
                calls: Mutex::new(0),
            }
        }
    }

    impl LlmProvider for MockProvider {
        fn name(&self) -> &'static str {
            "mock"
        }
        fn model(&self) -> &str {
            &self.model
        }
        fn check_connection(&self) -> Result<String, LlmError> {
            Ok("0.9.0".into())
        }
        fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
            Ok(vec![ModelInfo {
                name: "qwen3.5:4b".into(),
                size_bytes: 2_600_000_000,
                parameter_size: Some("4.0B".into()),
                quantization: Some("Q4_K_M".into()),
                family: Some("qwen3".into()),
            }])
        }
        fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
            unimplemented!("M0 only streams")
        }
        fn chat_stream(
            &self,
            request: ChatRequest,
            sink: &mut ChunkSink<'_>,
        ) -> Result<ChatResponse, LlmError> {
            *self.calls.lock().unwrap() += 1;
            let script = self.script.lock().unwrap().remove(0);
            let mut asm = StreamAssembler::new();
            match script {
                Script::Fail(e) => Err(e),
                Script::Empty => {
                    let done = ChatChunk {
                        done: true,
                        done_reason: Some("stop".into()),
                        ..Default::default()
                    };
                    asm.push(done.clone(), None);
                    let _ = sink(&done);
                    asm.finish(&request.model)
                }
                Script::Truncated(text) => {
                    let chunk = ChatChunk {
                        content: text.to_string(),
                        ..Default::default()
                    };
                    asm.push(chunk.clone(), None);
                    let _ = sink(&chunk);
                    let done = ChatChunk {
                        done: true,
                        done_reason: Some("length".into()),
                        ..Default::default()
                    };
                    asm.push(done.clone(), None);
                    let _ = sink(&done);
                    asm.finish(&request.model)
                }
                Script::Tokens(parts) => {
                    for part in parts {
                        let chunk = ChatChunk {
                            content: part.to_string(),
                            ..Default::default()
                        };
                        let keep = asm.push(chunk.clone(), None);
                        if sink(&chunk).is_err() {
                            return Err(LlmError::Cancelled);
                        }
                        if !keep {
                            break;
                        }
                    }
                    let last = ChatChunk {
                        done: true,
                        done_reason: Some("stop".into()),
                        ..Default::default()
                    };
                    asm.push(last.clone(), None);
                    let _ = sink(&last);
                    asm.finish(&request.model)
                }
            }
        }
    }

    fn orchestrator(script: Vec<Script>) -> Orchestrator<MockProvider> {
        Orchestrator::new(
            MockProvider::new(script),
            Context::new("you are a test", 4000),
            "10m",
        )
    }

    fn collect(out: &mut Vec<Event>) -> impl FnMut(Event) -> Result<(), ()> + '_ {
        move |e| {
            out.push(e);
            Ok(())
        }
    }

    #[test]
    fn streams_tokens_and_records_the_turn() {
        let mut o = orchestrator(vec![Script::Tokens(vec!["Hel", "lo"])]);
        let mut events = Vec::new();
        let outcome = o.send("hi", &mut collect(&mut events));

        assert_eq!(
            outcome,
            TurnOutcome::Answered {
                text: "Hello".into(),
                tool_calls: 0
            }
        );
        assert!(matches!(events[0], Event::Started { .. }));
        let tokens: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                Event::Token(t) => Some(t.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(tokens, vec!["Hel", "lo"]);
        assert!(matches!(events.last(), Some(Event::Finished { .. })));
        assert_eq!(o.context().history().len(), 2);
        assert_eq!(o.context().history()[0].role, Role::User);
        assert_eq!(o.context().history()[1].content, "Hello");
        assert_eq!(o.state(), &AgentState::Idle);
        assert!(o.last_turn().is_some());
    }

    #[test]
    fn conversation_history_accumulates_across_turns() {
        let mut o = orchestrator(vec![
            Script::Tokens(vec!["first"]),
            Script::Tokens(vec!["second"]),
        ]);
        let mut events = Vec::new();
        o.send("one", &mut collect(&mut events));
        o.send("two", &mut collect(&mut events));
        let history = o.context().history();
        assert_eq!(history.len(), 4);
        assert_eq!(history[2].content, "two");
        assert_eq!(history[3].content, "second");
    }

    #[test]
    fn failed_request_rolls_back_the_user_turn() {
        let mut o = orchestrator(vec![Script::Fail(LlmError::Connect {
            url: "http://localhost:11434".into(),
            source: "refused".into(),
        })]);
        let mut events = Vec::new();
        let outcome = o.send("hello", &mut collect(&mut events));
        assert!(matches!(outcome, TurnOutcome::Failed { .. }));
        assert!(
            o.context().history().is_empty(),
            "failed turn must not be kept"
        );
        assert!(matches!(o.state(), AgentState::Failed { .. }));
        assert!(matches!(
            events.last(),
            Some(Event::Failed { hint: Some(_), .. })
        ));
    }

    #[test]
    fn a_retry_after_failure_does_not_duplicate_the_question() {
        let mut o = orchestrator(vec![
            Script::Fail(LlmError::Io("reset".into())),
            Script::Tokens(vec!["ok"]),
        ]);
        let mut events = Vec::new();
        o.send("question", &mut collect(&mut events));
        o.send("question", &mut collect(&mut events));
        let history = o.context().history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].content, "question");
        assert_eq!(history[1].content, "ok");
    }

    #[test]
    fn empty_reply_is_rejected_and_reported() {
        let mut o = orchestrator(vec![Script::Empty]);
        let mut events = Vec::new();
        let outcome = o.send("hi", &mut collect(&mut events));
        assert_eq!(outcome, TurnOutcome::Empty);
        assert!(events.contains(&Event::Empty));
        assert!(o.context().history().is_empty());
    }

    #[test]
    fn truncation_is_surfaced_as_a_notice_but_kept() {
        let mut o = orchestrator(vec![Script::Truncated("half an ans")]);
        let mut events = Vec::new();
        let outcome = o.send("hi", &mut collect(&mut events));
        assert!(matches!(outcome, TurnOutcome::Answered { .. }));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Event::Notice(n) if n.contains("cut off")))
        );
        assert_eq!(
            o.context().history().len(),
            2,
            "partial text is still recorded"
        );
    }

    #[test]
    fn cancellation_stops_the_stream_and_keeps_history_clean() {
        let mut o = orchestrator(vec![Script::Tokens(vec!["a", "b", "c"])]);
        let mut count = 0;
        let mut on_event = |e: Event| {
            if matches!(e, Event::Token(_)) {
                count += 1;
                if count == 2 {
                    return Err(());
                }
            }
            Ok(())
        };
        let outcome = o.send("hi", &mut on_event);
        assert_eq!(outcome, TurnOutcome::Interrupted);
        assert_eq!(o.state(), &AgentState::Idle);
        assert!(
            o.context().history().is_empty(),
            "cancelled turn is rolled back"
        );
    }

    #[test]
    fn empty_prompt_sends_nothing() {
        let mut o = orchestrator(vec![]);
        let mut events = Vec::new();
        assert_eq!(o.send("   ", &mut collect(&mut events)), TurnOutcome::Empty);
        assert!(events.is_empty());
        assert_eq!(*o.provider().calls.lock().unwrap(), 0);
    }

    #[test]
    fn clear_resets_conversation_and_stats() {
        let mut o = orchestrator(vec![Script::Tokens(vec!["x"])]);
        let mut events = Vec::new();
        o.send("hi", &mut collect(&mut events));
        o.clear();
        assert!(o.context().history().is_empty());
        assert!(o.last_turn().is_none());
        assert_eq!(o.state(), &AgentState::Idle);
    }

    #[test]
    fn preflight_detects_a_missing_model() {
        struct MissingModel(MockProvider);
        impl LlmProvider for MissingModel {
            fn name(&self) -> &'static str {
                "mock"
            }
            fn model(&self) -> &str {
                self.0.model()
            }
            fn check_connection(&self) -> Result<String, LlmError> {
                self.0.check_connection()
            }
            fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
                Ok(vec![ModelInfo {
                    name: "llama3:8b".into(),
                    size_bytes: 1,
                    parameter_size: None,
                    quantization: None,
                    family: None,
                }])
            }
            fn chat(&self, r: ChatRequest) -> Result<ChatResponse, LlmError> {
                self.0.chat(r)
            }
            fn chat_stream(
                &self,
                r: ChatRequest,
                sink: &mut ChunkSink<'_>,
            ) -> Result<ChatResponse, LlmError> {
                self.0.chat_stream(r, sink)
            }
        }
        let o = Orchestrator::new(
            MissingModel(MockProvider::new(vec![])),
            Context::new("s", 1000),
            "10m",
        );
        let err = o.preflight().unwrap_err();
        assert!(matches!(err, LlmError::ModelNotFound { .. }));
    }

    #[test]
    fn tokens_per_second_guards_zero_duration() {
        let lt = LastTurn {
            elapsed: std::time::Duration::ZERO,
            eval_tokens: 10,
            prompt_tokens: 1,
            reply_chars: 5,
        };
        assert_eq!(lt.tokens_per_second(), None);
        let lt = LastTurn {
            elapsed: std::time::Duration::from_secs(2),
            eval_tokens: 20,
            prompt_tokens: 1,
            reply_chars: 5,
        };
        assert_eq!(lt.tokens_per_second(), Some(10.0));
    }
}
