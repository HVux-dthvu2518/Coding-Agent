//! The interactive REPL.
//!
//! Input comes through an [`InputSource`] trait rather than `stdin` directly, so
//! the entire session — banners, commands, streaming, cancellation, error
//! recovery — is testable without a terminal. `Session` is the unit under test;
//! `run` is a thin adapter that wires `stdin` and `stdout` into it.
//!
//! Output goes through [`Output`], which is generic over `io::Write`, so tests
//! capture exact bytes and assert on the user-visible transcript.

use crate::agent::Context;
use crate::agent::orchestrator::{Event, LastTurn, Orchestrator, TurnOutcome};
use crate::cli::commands::{self, Command, Input};
use crate::cli::ui::{self, Style};
use crate::config::Config;
use crate::llm::provider::LlmProvider;
use std::io::{BufRead, Write};

/// Where prompt lines come from.
///
/// The prompt itself is printed by [`Session`] so that the session keeps a
/// single owner of the output stream.
pub trait InputSource {
    /// Next line, or `None` at end of input (Ctrl-D).
    fn next_line(&mut self) -> Option<String>;
}

/// Line reader over any `BufRead` (a real terminal, or a test fixture).
pub struct ReaderInput<R: BufRead> {
    reader: R,
}

impl<R: BufRead> ReaderInput<R> {
    pub fn new(reader: R) -> Self {
        ReaderInput { reader }
    }
}

impl<R: BufRead> InputSource for ReaderInput<R> {
    fn next_line(&mut self) -> Option<String> {
        let mut line = String::new();
        match self.reader.read_line(&mut line) {
            Ok(0) => None,
            Ok(_) => Some(line.trim_end_matches(['\n', '\r']).to_string()),
            Err(_) => None,
        }
    }
}

/// An [`InputSource`] that replays a fixed script (tests, `--script`).
pub struct ScriptedInput {
    lines: std::collections::VecDeque<String>,
}

impl ScriptedInput {
    pub fn new<I, S>(lines: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        ScriptedInput {
            lines: lines.into_iter().map(Into::into).collect(),
        }
    }
}

impl InputSource for ScriptedInput {
    fn next_line(&mut self) -> Option<String> {
        self.lines.pop_front()
    }
}

/// Prompt label configuration, so the same session code serves `chat` and `run`.
#[derive(Debug, Clone)]
pub struct Labels {
    pub user: String,
    pub assistant: String,
    /// Whether to echo the banner and prompts (off when input is scripted).
    pub interactive: bool,
}

impl Default for Labels {
    fn default() -> Self {
        Labels {
            user: "You".into(),
            assistant: "Qwen".into(),
            interactive: true,
        }
    }
}

/// The REPL session: owns the orchestrator and the presentation state.
pub struct Session<P: LlmProvider, W: Write> {
    orchestrator: Orchestrator<P>,
    out: W,
    style: Style,
    labels: Labels,
    /// Last submitted user message, for `/retry`.
    last_prompt: Option<String>,
    /// Set by `/exit`.
    should_exit: bool,
    /// Human-readable config summary lines for the banner.
    banner: Vec<String>,
    /// Non-fatal configuration warnings shown once at startup.
    warnings: Vec<String>,
}

impl<P: LlmProvider, W: Write> Session<P, W> {
    pub fn new(orchestrator: Orchestrator<P>, out: W, style: Style) -> Self {
        Session {
            orchestrator,
            out,
            style,
            labels: Labels::default(),
            last_prompt: None,
            should_exit: false,
            banner: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn with_labels(mut self, labels: Labels) -> Self {
        self.labels = labels;
        self
    }

    pub fn with_banner(mut self, lines: Vec<String>) -> Self {
        self.banner = lines;
        self
    }

    pub fn with_warnings(mut self, warnings: Vec<String>) -> Self {
        self.warnings = warnings;
        self
    }

    pub fn should_exit(&self) -> bool {
        self.should_exit
    }

    pub fn orchestrator(&self) -> &Orchestrator<P> {
        &self.orchestrator
    }

    pub fn into_inner(self) -> (Orchestrator<P>, W) {
        (self.orchestrator, self.out)
    }

    /// Borrow the output sink (used by tests to inspect the transcript).
    pub fn output(&self) -> &W {
        &self.out
    }

    /// Test hook: mutate the provider (used to simulate a server going away).
    #[cfg(test)]
    pub fn provider_mut(&mut self) -> &mut P {
        // Safe because the orchestrator owns the provider and exposes no other
        // mutable access to it.
        self.orchestrator.provider_mut()
    }

    /// Prints the banner: name, provider, model, status.
    pub fn print_banner(&mut self) {
        if !self.labels.interactive {
            return;
        }
        let width = ui::term_width().min(60);
        let lines = self.banner.clone();
        let banner = ui::boxed(
            &self.orchestrator.provider().name().to_uppercase(),
            &lines,
            &self.style,
            width,
        );
        let _ = writeln!(self.out, "{banner}");
        let _ = writeln!(
            self.out,
            "{}",
            self.style
                .dim("Type /help for commands, /exit to quit, Ctrl-C to cancel a reply.")
        );
        let _ = writeln!(self.out);
        let _ = self.out.flush();
        for warning in &self.warnings {
            let _ = writeln!(self.out, "{} {warning}", self.style.yellow("warning:"));
        }
    }

    /// Runs the REPL until `/exit` or end of input.
    pub fn run<I: InputSource>(&mut self, input: &mut I) {
        self.print_banner();
        while !self.should_exit {
            let prompt = if self.labels.interactive {
                ui::prompt(&self.style, &self.labels.user)
            } else {
                String::new()
            };
            if !prompt.is_empty() {
                let _ = write!(self.out, "{prompt}");
                let _ = self.out.flush();
            }
            let Some(line) = input.next_line() else {
                // Ctrl-D / end of script.
                if self.labels.interactive {
                    let _ = writeln!(self.out, "\n{}", self.style.dim("Goodbye!"));
                }
                return;
            };
            self.handle_line(&line);
            let _ = self.out.flush();
        }
    }

    /// Processes one line of input: command, message, or nothing.
    pub fn handle_line(&mut self, line: &str) {
        match commands::parse(line) {
            Input::Empty => {}
            Input::Message(text) => self.submit(&text),
            Input::Command(cmd) => self.handle_command(cmd),
        }
    }

    /// Sends a message to the model and renders the streamed answer.
    fn submit(&mut self, text: &str) {
        self.last_prompt = Some(text.to_string());
        let interactive = self.labels.interactive;
        let label = self.labels.assistant.clone();
        let style = self.style;
        let out = &mut self.out;
        let mut wrote_prefix = false;
        let mut wrote_newline = false;

        // Rendering is done here, not in the orchestrator: the agent core never
        // knows a terminal exists.
        let on_event = |event: Event| -> Result<(), ()> {
            match event {
                Event::Started { .. } => {
                    // Non-interactive mode (`run`) emits the answer only, so its
                    // output can be piped straight into another tool.
                    if interactive {
                        let _ = write!(
                            out,
                            "{} ",
                            style.bold(&style.magenta(&format!("{label} >")))
                        );
                        wrote_prefix = true;
                        let _ = out.flush();
                    }
                }
                Event::Token(t) => {
                    let _ = write!(out, "{}", ui::render_markdown(&t, &style));
                    let _ = out.flush();
                }
                Event::Thinking(t) => {
                    // Reasoning traces are diagnostics, not answers.
                    if crate::logging::enabled(crate::config::LogLevel::Debug) {
                        let _ = write!(out, "{}", style.dim(&t));
                        let _ = out.flush();
                    }
                }
                Event::Finished { ref text, .. } => {
                    let _ = text;
                    ensure_newline(out, &mut wrote_newline);
                }
                Event::Interrupted => {
                    ensure_newline(out, &mut wrote_newline);
                    if interactive {
                        let _ = writeln!(out, "{}", style.dim("[cancelled]"));
                    }
                }
                Event::Empty => {
                    ensure_newline(out, &mut wrote_newline);
                }
                Event::Failed {
                    ref message,
                    ref hint,
                } => {
                    ensure_newline(out, &mut wrote_newline);
                    let mut sink = std::io::stderr();
                    let _ = writeln!(sink, "{} {message}", style.red("error:"));
                    if let Some(hint) = hint {
                        let _ = writeln!(sink, "       {}", style.dim(hint));
                    }
                }
                Event::Notice(ref notice) => {
                    ensure_newline(out, &mut wrote_newline);
                    if interactive {
                        let _ = writeln!(out, "{} {notice}", style.yellow("note:"));
                    } else {
                        // Diagnostics belong on stderr when stdout is the answer.
                        eprintln!("note: {notice}");
                    }
                }
                Event::ToolCallsRequested(n) => {
                    ensure_newline(out, &mut wrote_newline);
                    let message = format!(
                        "the model asked for {n} tool call(s); tool execution arrives in M1"
                    );
                    if interactive {
                        let _ = writeln!(out, "{} {message}", style.yellow("note:"));
                    } else {
                        eprintln!("note: {message}");
                    }
                }
            }
            Ok(())
        };

        let outcome = {
            // The closure borrows `out`; end the borrow before rendering the
            // result.
            let mut on_event = on_event;
            self.orchestrator.send(text, &mut on_event)
        };

        if wrote_prefix && !wrote_newline {
            let _ = writeln!(self.out);
        }
        match outcome {
            TurnOutcome::Answered { .. } => {
                if interactive {
                    if let Some(stats) = self.footer_stats() {
                        let _ = writeln!(self.out, "{}", self.style.dim(&stats));
                    }
                }
            }
            TurnOutcome::Empty => {
                let note = "nothing usable was returned; try again or /status";
                if interactive {
                    let _ = writeln!(self.out, "{} {note}", self.style.yellow("note:"));
                } else {
                    eprintln!("note: {note}");
                }
            }
            _ => {}
        }
        if interactive {
            let _ = writeln!(self.out);
        }
        let _ = self.out.flush();
    }

    /// "12.4 tok/s · 340 tokens · ctx 512/3000" or `None` when unknown.
    fn footer_stats(&self) -> Option<String> {
        let last: &LastTurn = self.orchestrator.last_turn()?;
        let mut parts = Vec::new();
        if let Some(tps) = last.tokens_per_second() {
            parts.push(format!("{tps:.1} tok/s"));
        }
        if last.eval_tokens > 0 {
            parts.push(format!("{} reply tokens", last.eval_tokens));
        }
        if last.prompt_tokens > 0 {
            parts.push(format!("{} prompt tokens", last.prompt_tokens));
        }
        if parts.is_empty() {
            return None;
        }
        let ctx = self.orchestrator.context().stats();
        parts.push(format!(
            "context {}/{} est.",
            ctx.estimated_tokens,
            self.orchestrator.context().budget_tokens()
        ));
        Some(parts.join(" · "))
    }

    fn handle_command(&mut self, cmd: Command) {
        match cmd {
            Command::Help => {
                let _ = writeln!(self.out, "{}", commands::help_text());
            }
            Command::Exit => {
                self.should_exit = true;
                if self.labels.interactive {
                    let _ = writeln!(self.out, "{}", self.style.dim("Goodbye!"));
                }
            }
            Command::Clear => {
                self.orchestrator.clear();
                self.last_prompt = None;
                let _ = writeln!(self.out, "{}", self.style.dim("Conversation cleared."));
            }
            Command::Model => {
                let _ = writeln!(
                    self.out,
                    "model: {}  provider: {}",
                    self.style.bold(self.orchestrator.provider().model()),
                    self.orchestrator.provider().name()
                );
            }
            Command::Models => self.list_models(),
            Command::Status => self.status(),
            Command::Stats => self.stats(),
            Command::History => self.history(),
            Command::System(arg) => self.system(arg),
            Command::Config => {
                for line in &self.banner {
                    let _ = writeln!(self.out, "  {line}");
                }
            }
            Command::Save(path) => self.save(path),
            Command::Retry => match self.last_prompt.clone() {
                Some(prompt) => {
                    let _ = writeln!(
                        self.out,
                        "{}",
                        self.style.dim(&format!("retrying: {prompt}"))
                    );
                    self.submit(&prompt);
                }
                None => {
                    let _ = writeln!(self.out, "{} nothing to retry", self.style.yellow("note:"));
                }
            },
            Command::Unknown { name, suggestion } => {
                let _ = writeln!(
                    self.out,
                    "{} unknown command {name}",
                    self.style.yellow("note:")
                );
                if let Some(s) = suggestion {
                    let _ = writeln!(self.out, "       did you mean {s}?");
                } else {
                    let _ = writeln!(self.out, "       type /help for the list");
                }
            }
        }
        let _ = self.out.flush();
    }

    fn list_models(&mut self) {
        match self.orchestrator.provider().list_models() {
            Ok(models) if models.is_empty() => {
                let _ = writeln!(
                    self.out,
                    "{} no models installed. Try `ollama pull {}`",
                    self.style.yellow("note:"),
                    self.orchestrator.provider().model()
                );
            }
            Ok(models) => {
                let active = self.orchestrator.provider().model().to_string();
                for model in &models {
                    let marker = if crate::llm::matches_model(&model.name, &active) {
                        self.style.green("*")
                    } else {
                        " ".into()
                    };
                    let mut detail = vec![model.human_size()];
                    if let Some(p) = &model.parameter_size {
                        detail.push(p.clone());
                    }
                    if let Some(q) = &model.quantization {
                        detail.push(q.clone());
                    }
                    let _ = writeln!(
                        self.out,
                        "{marker} {:<28} {}",
                        model.name,
                        self.style.dim(&detail.join(" · "))
                    );
                }
            }
            Err(err) => self.print_llm_error(&err),
        }
    }

    fn status(&mut self) {
        let health = self.orchestrator.health();
        let style = self.style;
        if health.reachable {
            let _ = writeln!(
                self.out,
                "{} ollama {} · {} model(s) installed",
                style.green("connected:"),
                health.version.clone().unwrap_or_else(|| "?".into()),
                health.model_count
            );
            if health.model_present {
                let _ = writeln!(
                    self.out,
                    "{} model {} is available",
                    style.green("ready:"),
                    style.bold(self.orchestrator.provider().model())
                );
            } else {
                let _ = writeln!(
                    self.out,
                    "{} model {} is not installed — run `ollama pull {}`",
                    style.red("missing:"),
                    self.orchestrator.provider().model(),
                    self.orchestrator.provider().model()
                );
            }
        } else {
            let _ = writeln!(self.out, "{} server unreachable", style.red("offline:"));
            if let Some(err) = &health.error {
                let _ = writeln!(self.out, "       {err}");
                let _ = writeln!(
                    self.out,
                    "       {}",
                    style.dim("Start it with `ollama serve`.")
                );
            }
        }
        let state = self.orchestrator.state();
        let ctx = self.orchestrator.context().stats();
        let _ = writeln!(
            self.out,
            "{} {} · {} message(s) · ~{} tokens of {}",
            style.dim("state:"),
            state.name(),
            ctx.messages,
            ctx.estimated_tokens,
            self.orchestrator.context().budget_tokens()
        );
        if ctx.evicted_turns > 0 {
            let _ = writeln!(
                self.out,
                "{} {} old turn(s) were dropped to fit the context budget",
                style.dim("memory:"),
                ctx.evicted_turns
            );
        }
    }

    fn stats(&mut self) {
        let style = self.style;
        match self.orchestrator.last_turn() {
            None => {
                let _ = writeln!(
                    self.out,
                    "{} no turn has completed yet",
                    style.dim("stats:")
                );
            }
            Some(last) => {
                let _ = writeln!(
                    self.out,
                    "{} {:.2?} · {} reply tokens · {} prompt tokens · {} chars",
                    style.dim("last turn:"),
                    last.elapsed,
                    last.eval_tokens,
                    last.prompt_tokens,
                    last.reply_chars
                );
                if let Some(tps) = last.tokens_per_second() {
                    let _ = writeln!(self.out, "{} {tps:.1} tokens/second", style.dim("speed:"));
                }
            }
        }
        let ctx = self.orchestrator.context().stats();
        let _ = writeln!(
            self.out,
            "{} {} message(s) · ~{} / {} tokens · {} turn(s) evicted",
            style.dim("context:"),
            ctx.messages,
            ctx.estimated_tokens,
            self.orchestrator.context().budget_tokens(),
            ctx.evicted_turns
        );
    }

    fn history(&mut self) {
        let style = self.style;
        let messages = self.orchestrator.context().history().to_vec();
        if messages.is_empty() {
            let _ = writeln!(self.out, "{}", style.dim("(no messages yet)"));
            return;
        }
        for message in messages {
            let (label, colour) = match message.role {
                crate::llm::Role::User => ("you", style.cyan("you")),
                crate::llm::Role::Assistant => ("qwen", style.magenta("qwen")),
                crate::llm::Role::System => ("sys", style.dim("sys")),
                crate::llm::Role::Tool => ("tool", style.yellow("tool")),
            };
            let _ = label;
            let body = ui::render_markdown(&message.content, &style);
            let _ = writeln!(self.out, "{colour} {}", body.trim_end());
        }
    }

    fn system(&mut self, arg: Option<String>) {
        match arg {
            Some(text) => {
                self.orchestrator
                    .context_mut()
                    .set_system_prompt(text.clone());
                let _ = writeln!(
                    self.out,
                    "{} system prompt updated ({} chars)",
                    self.style.dim("ok:"),
                    text.chars().count()
                );
            }
            None => {
                let _ = writeln!(self.out, "{}", self.orchestrator.context().system_prompt());
            }
        }
    }

    fn save(&mut self, path: Option<String>) {
        let path = path.unwrap_or_else(|| {
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            format!("qwen-agent-{stamp}.md")
        });
        let history = self.orchestrator.context().history();
        if history.is_empty() {
            let _ = writeln!(self.out, "{}", self.style.dim("nothing to save yet"));
            return;
        }
        let mut doc = String::from("# qwen-agent conversation\n\n");
        doc.push_str(&format!(
            "Model: `{}` · provider: `{}`\n\n",
            self.orchestrator.provider().model(),
            self.orchestrator.provider().name()
        ));
        for message in history {
            let heading = match message.role {
                crate::llm::Role::User => "You",
                crate::llm::Role::Assistant => "Qwen",
                crate::llm::Role::System => "System",
                crate::llm::Role::Tool => "Tool",
            };
            doc.push_str(&format!("## {heading}\n\n{}\n\n", message.content.trim()));
        }
        match std::fs::write(&path, doc) {
            Ok(()) => {
                let _ = writeln!(
                    self.out,
                    "{} saved {} message(s) to {}",
                    self.style.green("ok:"),
                    history.len(),
                    self.style.bold(&path)
                );
            }
            Err(err) => {
                let _ = writeln!(
                    self.out,
                    "{} cannot write {path}: {err}",
                    self.style.red("error:")
                );
            }
        }
    }

    fn print_llm_error(&mut self, err: &crate::error::LlmError) {
        let _ = writeln!(self.out, "{} {err}", self.style.red("error:"));
        let _ = writeln!(self.out, "       {}", self.style.dim(err.hint()));
    }
}

fn ensure_newline<W: Write>(out: &mut W, wrote_newline: &mut bool) {
    if !*wrote_newline {
        let _ = writeln!(out);
        *wrote_newline = true;
    }
}

/// Builds the banner lines shown above the prompt.
pub fn banner_lines(cfg: &Config) -> Vec<String> {
    vec![
        format!("Provider : {}", cfg.llm.provider),
        format!("Model    : {}", cfg.llm.model),
        format!("Endpoint : {}", cfg.llm.base_url),
        format!(
            "Streaming: {}   Temp: {:.1}",
            if cfg.llm.stream { "on" } else { "off" },
            cfg.llm.temperature
        ),
    ]
}

/// Composition helper used by `app.rs` so the CLI module does not need to know
/// how a provider is constructed.
pub fn context_for(cfg: &Config) -> Context {
    let budget = cfg
        .llm
        .num_ctx
        .map(|n| (n as usize / 2).max(256))
        .unwrap_or(3000);
    Context::new(cfg.agent.system_prompt.clone(), budget)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Context;
    use crate::error::LlmError;
    use crate::llm::StreamAssembler;
    use crate::llm::provider::{ChunkSink, ModelInfo};
    use crate::llm::request::ChatRequest;
    use crate::llm::response::{ChatChunk, ChatResponse};
    use std::sync::Mutex;

    /// Provider whose next reply is scripted per call.
    struct StubProvider {
        model: String,
        replies: Mutex<Vec<Reply>>,
        fail_models: Mutex<bool>,
    }

    enum Reply {
        Text(&'static str),
        Fail(&'static str),
    }

    impl StubProvider {
        fn new(replies: Vec<Reply>) -> Self {
            StubProvider {
                model: "qwen3.5:4b".into(),
                replies: Mutex::new(replies),
                fail_models: Mutex::new(false),
            }
        }
    }

    impl LlmProvider for StubProvider {
        fn name(&self) -> &'static str {
            "stub"
        }
        fn model(&self) -> &str {
            &self.model
        }
        fn check_connection(&self) -> Result<String, LlmError> {
            Ok("0.9.1".into())
        }
        fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
            if *self.fail_models.lock().unwrap() {
                return Err(LlmError::Connect {
                    url: "http://x".into(),
                    source: "refused".into(),
                });
            }
            Ok(vec![
                ModelInfo {
                    name: "qwen3.5:4b".into(),
                    size_bytes: 2_600_000_000,
                    parameter_size: Some("4.0B".into()),
                    quantization: Some("Q4_K_M".into()),
                    family: None,
                },
                ModelInfo {
                    name: "llama3:8b".into(),
                    size_bytes: 4_700_000_000,
                    parameter_size: None,
                    quantization: None,
                    family: None,
                },
            ])
        }
        fn chat(&self, _r: ChatRequest) -> Result<ChatResponse, LlmError> {
            unimplemented!()
        }
        fn chat_stream(
            &self,
            request: ChatRequest,
            sink: &mut ChunkSink<'_>,
        ) -> Result<ChatResponse, LlmError> {
            let reply = self.replies.lock().unwrap().remove(0);
            match reply {
                Reply::Fail(msg) => Err(LlmError::Connect {
                    url: "http://localhost:11434".into(),
                    source: msg.into(),
                }),
                Reply::Text(body) => {
                    let mut asm = StreamAssembler::new();
                    for chunk in body.split_inclusive(' ') {
                        let c = ChatChunk {
                            content: chunk.to_string(),
                            ..Default::default()
                        };
                        asm.push(c.clone(), None);
                        let _ = sink(&c);
                    }
                    let done = ChatChunk {
                        done: true,
                        done_reason: Some("stop".into()),
                        ..Default::default()
                    };
                    // Real providers report counters on the terminal record;
                    // mirror that so the stats footer is exercised.
                    let raw = crate::json::parse(
                        r#"{"done":true,"eval_count":42,"prompt_eval_count":31}"#,
                    )
                    .unwrap();
                    asm.push(done.clone(), Some(&raw));
                    let _ = sink(&done);
                    asm.finish(&request.model)
                }
            }
        }
    }

    fn session(lines: Vec<Reply>) -> Session<StubProvider, Vec<u8>> {
        let provider = StubProvider::new(lines);
        let orchestrator = Orchestrator::new(provider, Context::new("sys", 3000), "10m");
        Session::new(orchestrator, Vec::new(), Style::plain()).with_labels(Labels {
            user: "You".into(),
            assistant: "Qwen".into(),
            interactive: false,
        })
    }

    /// Reads the output buffer without consuming the session.
    fn transcript(session: &Session<StubProvider, Vec<u8>>) -> String {
        String::from_utf8(session.output().clone()).unwrap()
    }

    #[test]
    fn runs_a_full_conversation_from_a_script() {
        let mut s = session(vec![Reply::Text("Hello there."), Reply::Text("Fine.")]);
        let mut input = ScriptedInput::new(["hi", "how are you", "/exit"]);
        s.run(&mut input);
        let out = transcript(&s);
        assert!(out.contains("Hello there."), "{out}");
        assert!(out.contains("Fine."), "{out}");
        assert_eq!(s.orchestrator().context().history().len(), 4);
    }

    #[test]
    fn end_of_input_exits_without_error() {
        let mut s = session(vec![Reply::Text("ok")]);
        let mut input = ScriptedInput::new(["hi"]);
        s.run(&mut input);
        assert_eq!(s.orchestrator().context().history().len(), 2);
    }

    #[test]
    fn blank_lines_are_not_sent() {
        let mut s = session(vec![]);
        s.handle_line("   ");
        assert!(s.orchestrator().context().history().is_empty());
        assert!(transcript(&s).is_empty());
    }

    #[test]
    fn errors_stay_off_stdout_and_the_session_survives() {
        // stderr cannot be captured in-process, so this asserts what stdout
        // owns: the session stays usable and the answer still prints. The error
        // text itself is covered end-to-end by
        // `tests/cli_end_to_end.rs::chat_reports_an_unreachable_server_clearly`.
        let mut s = session(vec![
            Reply::Fail("connection refused"),
            Reply::Text("recovered"),
        ]);
        s.handle_line("hello");
        s.handle_line("again");
        let out = transcript(&s);
        assert!(
            !out.contains("connection refused"),
            "errors must go to stderr, not stdout: {out}"
        );
        assert!(out.contains("recovered"), "{out}");
        assert_eq!(s.orchestrator().context().history().len(), 2);
    }

    #[test]
    fn one_shot_output_is_only_the_answer() {
        let mut s = session(vec![Reply::Text("just the answer")]);
        s.handle_line("question");
        assert_eq!(transcript(&s).trim(), "just the answer");
    }

    #[test]
    fn interactive_output_has_a_prompt_and_stats() {
        let provider = StubProvider::new(vec![Reply::Text("answer")]);
        let orchestrator = Orchestrator::new(provider, Context::new("sys", 3000), "10m");
        let mut s = Session::new(orchestrator, Vec::new(), Style::plain());
        s.handle_line("question");
        let out = transcript(&s);
        assert!(out.contains("Qwen >"), "{out}");
        assert!(out.contains("context"), "stats footer: {out}");
    }

    #[test]
    fn commands_help_clear_and_unknown() {
        let mut s = session(vec![]);
        s.handle_line("/help");
        s.handle_line("/hlep");
        s.handle_line("/nonsense-xyz");
        let out = transcript(&s);
        assert!(out.contains("/status"));
        assert!(out.contains("did you mean /help"), "{out}");
        assert!(out.contains("/nonsense-xyz"), "{out}");
    }

    #[test]
    fn clear_command_resets_context() {
        let mut s = session(vec![Reply::Text("answer")]);
        s.handle_line("question");
        assert_eq!(s.orchestrator().context().history().len(), 2);
        s.handle_line("/clear");
        assert!(s.orchestrator().context().history().is_empty());
        assert!(transcript(&s).contains("cleared"));
    }

    #[test]
    fn retry_resends_the_last_message() {
        let mut s = session(vec![Reply::Fail("boom"), Reply::Text("second time lucky")]);
        s.handle_line("question");
        // A failed turn is rolled back, so the retry produces exactly one exchange.
        s.handle_line("/retry");
        let out = transcript(&s);
        assert!(out.contains("retrying: question"), "{out}");
        assert!(out.contains("second time lucky"), "{out}");
        assert_eq!(s.orchestrator().context().history().len(), 2);
    }

    #[test]
    fn retry_without_history_is_harmless() {
        let mut s = session(vec![]);
        s.handle_line("/retry");
        assert!(transcript(&s).contains("nothing to retry"));
    }

    #[test]
    fn models_lists_and_marks_the_active_one() {
        let mut s = session(vec![]);
        s.handle_line("/models");
        let out = transcript(&s);
        assert!(out.contains("qwen3.5:4b"));
        assert!(out.contains("llama3:8b"));
        assert!(out.contains("2.6 GB"), "{out}");
        assert!(out.contains('*'), "active model must be marked: {out}");
    }

    #[test]
    fn models_reports_connection_failure() {
        let mut s = session(vec![]);
        *s.provider_mut().fail_models.lock().unwrap() = true;
        s.handle_line("/models");
        let out = transcript(&s);
        assert!(out.contains("error:"), "{out}");
    }

    #[test]
    fn status_shows_ready_and_state() {
        let mut s = session(vec![]);
        s.handle_line("/status");
        let out = transcript(&s);
        assert!(out.contains("connected"), "{out}");
        assert!(out.contains("available"), "{out}");
        assert!(out.contains("state:"), "{out}");
    }

    #[test]
    fn system_command_replaces_the_prompt() {
        let mut s = session(vec![]);
        s.handle_line("/system reply only in haiku");
        s.handle_line("/system");
        let out = transcript(&s);
        assert!(out.contains("system prompt updated"), "{out}");
        assert!(out.contains("reply only in haiku"), "{out}");
        assert_eq!(
            s.orchestrator().context().system_prompt(),
            "reply only in haiku"
        );
    }

    #[test]
    fn stats_before_any_turn_is_informative() {
        let mut s = session(vec![]);
        s.handle_line("/stats");
        assert!(transcript(&s).contains("no turn has completed"));
    }

    #[test]
    fn history_prints_roles_and_content() {
        let mut s = session(vec![Reply::Text("hi back")]);
        s.handle_line("hello");
        s.handle_line("/history");
        let out = transcript(&s);
        assert!(out.contains("You") || out.contains("you"), "{out}");
        assert!(out.contains("hello"), "{out}");
        assert!(out.contains("hi back"), "{out}");
    }

    #[test]
    fn save_writes_a_markdown_transcript() {
        let dir = std::env::temp_dir().join(format!("qwen-save-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("chat.md");
        let mut s = session(vec![Reply::Text("saved answer")]);
        s.handle_line("question");
        s.handle_line(&format!("/save {}", path.display()));
        let doc = std::fs::read_to_string(&path).unwrap();
        assert!(doc.contains("# qwen-agent conversation"));
        assert!(doc.contains("## You"));
        assert!(doc.contains("saved answer"));
        assert!(transcript(&s).contains("ok:"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn exit_command_stops_the_loop() {
        let mut s = session(vec![Reply::Text("first")]);
        let mut input = ScriptedInput::new(["hi", "/exit", "never sent"]);
        s.run(&mut input);
        assert!(s.should_exit());
        assert_eq!(s.orchestrator().context().history().len(), 2);
    }

    #[test]
    fn banner_lines_describe_the_configuration() {
        let cfg = Config::default();
        let lines = banner_lines(&cfg);
        assert!(lines.iter().any(|l| l.contains(&cfg.llm.model)));
        assert!(lines.iter().any(|l| l.contains(&cfg.llm.base_url)));
        assert!(lines.iter().any(|l| l.contains("Streaming")));
    }

    #[test]
    fn context_budget_follows_num_ctx() {
        let mut cfg = Config::default();
        assert_eq!(context_for(&cfg).budget_tokens(), 3000);
        cfg.llm.num_ctx = Some(8192);
        assert_eq!(context_for(&cfg).budget_tokens(), 4096);
    }
}
