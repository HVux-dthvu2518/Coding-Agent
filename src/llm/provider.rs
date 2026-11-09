//! The provider abstraction.
//!
//! ```text
//!     Orchestrator ──► LlmProvider (trait object)
//!                          ├── OllamaProvider      (M0)
//!                          ├── LlamaCppProvider    (future)
//!                          └── OpenAiCompatProvider(future)
//! ```
//!
//! The trait is deliberately **blocking and synchronous**. M0 runs the agent on
//! a dedicated thread and streams tokens to the UI through a channel, which is
//! the same concurrency shape Tokio will provide later; making the trait `async`
//! today would force a hand-rolled `Future` implementation for no benefit.
//! Migrating to `#[async_trait]` in M1 is a mechanical change confined to the
//! `llm` module.

use crate::error::LlmError;
use crate::llm::request::{ChatRequest, Message};
use crate::llm::response::{ChatChunk, ChatResponse, NdjsonBuffer, Usage};
use std::sync::mpsc::Sender;

/// Metadata about a model installed on the server.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelInfo {
    /// e.g. `qwen3.5:4b`
    pub name: String,
    pub size_bytes: u64,
    /// e.g. `4.0B`
    pub parameter_size: Option<String>,
    pub quantization: Option<String>,
    pub family: Option<String>,
}

impl ModelInfo {
    /// Human-readable size, e.g. `2.6 GB`.
    pub fn human_size(&self) -> String {
        human_bytes(self.size_bytes)
    }
}

/// Formats a byte count using SI units, matching what `ollama list` prints.
pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1000.0 && unit + 1 < UNITS.len() {
        size /= 1000.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else if size >= 100.0 {
        format!("{size:.0} {}", UNITS[unit])
    } else {
        format!("{size:.1} {}", UNITS[unit])
    }
}

/// How a provider is doing, for the `/status` command and `doctor`.
#[derive(Debug, Clone, PartialEq)]
pub struct Health {
    pub reachable: bool,
    /// Server version string, when the provider reports one.
    pub version: Option<String>,
    /// Number of models available.
    pub model_count: usize,
    /// Whether the configured model is installed.
    pub model_present: bool,
    /// Populated when `reachable == false`.
    pub error: Option<String>,
}

impl Health {
    pub fn unreachable(error: impl Into<String>) -> Health {
        Health {
            reachable: false,
            version: None,
            model_count: 0,
            model_present: false,
            error: Some(error.into()),
        }
    }
}

/// Callback invoked for every token as it arrives.
///
/// Returning `Err(())` cancels the stream — that is how Ctrl-C and `/stop`
/// propagate into the provider without a second thread.
pub type ChunkSink<'a> = dyn FnMut(&ChatChunk) -> Result<(), ()> + 'a;

/// Everything the agent core needs from a model server.
pub trait LlmProvider: Send + Sync {
    /// Short provider name, e.g. `ollama`.
    fn name(&self) -> &'static str;

    /// The model this provider is configured to talk to.
    fn model(&self) -> &str;

    /// `GET /api/version` — is the server up?
    fn check_connection(&self) -> Result<String, LlmError>;

    /// `GET /api/tags` — what models are installed?
    fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError>;

    /// Is the configured model installed?
    fn model_exists(&self) -> Result<bool, LlmError> {
        Ok(self
            .list_models()?
            .iter()
            .any(|m| matches_model(&m.name, self.model())))
    }

    /// One-shot chat completion (`stream = false`).
    fn chat(&self, request: ChatRequest) -> Result<ChatResponse, LlmError>;

    /// Streamed chat completion. `sink` is called for each record; the returned
    /// response contains the assembled text and the final usage counters.
    fn chat_stream(
        &self,
        request: ChatRequest,
        sink: &mut ChunkSink<'_>,
    ) -> Result<ChatResponse, LlmError>;

    /// Convenience: health summary used by `doctor` and `/status`.
    fn health(&self) -> Health {
        match self.check_connection() {
            Err(e) => Health::unreachable(e.to_string()),
            Ok(version) => match self.list_models() {
                Err(e) => Health::unreachable(e.to_string()),
                Ok(models) => Health {
                    reachable: true,
                    version: Some(version),
                    model_count: models.len(),
                    model_present: models.iter().any(|m| matches_model(&m.name, self.model())),
                    error: None,
                },
            },
        }
    }
}

/// Compares model names the way Ollama does: case-insensitively, treating a
/// missing `:latest` tag as equivalent.
///
/// `qwen3.5:4b` and `Qwen3.5:4B` are the same model; `qwen3.5` and
/// `qwen3.5:latest` are the same model.
pub fn matches_model(installed: &str, wanted: &str) -> bool {
    fn norm(s: &str) -> String {
        let s = s.trim().to_ascii_lowercase();
        // Ollama accepts a bare name as the `:latest` tag.
        if s.contains(':') {
            s
        } else {
            format!("{s}:latest")
        }
    }
    if norm(installed) == norm(wanted) {
        return true;
    }
    // `qwen3.5:4b` should also be found when the user configured `qwen3.5`.
    // Compare the repository part when the installed name carries a tag and the
    // wanted name does not.
    let (inst_repo, inst_tag) = split_tag(installed);
    let (want_repo, want_tag) = split_tag(wanted);
    inst_repo.eq_ignore_ascii_case(want_repo)
        && (inst_tag.is_none()
            || want_tag.is_none()
            || inst_tag.unwrap().eq_ignore_ascii_case(want_tag.unwrap()))
}

fn split_tag(name: &str) -> (&str, Option<&str>) {
    match name.rsplit_once(':') {
        Some((repo, tag)) => (repo, Some(tag)),
        None => (name, None),
    }
}

/// Accumulates streamed chunks into a final [`ChatResponse`].
///
/// Shared by all providers so token accounting and cancellation semantics stay
/// identical no matter which backend is in use.
#[derive(Debug, Default)]
pub struct StreamAssembler {
    content: String,
    thinking: String,
    tool_calls: Vec<crate::llm::request::ToolCall>,
    done_reason: Option<String>,
    usage: Usage,
    saw_terminal: bool,
}

impl StreamAssembler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one decoded chunk. Returns `false` once the stream is finished.
    pub fn push(&mut self, chunk: ChatChunk, raw: Option<&crate::json::Value>) -> bool {
        if !chunk.content.is_empty() {
            self.content.push_str(&chunk.content);
        }
        if !chunk.thinking.is_empty() {
            self.thinking.push_str(&chunk.thinking);
        }
        if !chunk.tool_calls.is_empty() {
            self.tool_calls.extend(chunk.tool_calls.iter().cloned());
        }
        if let Some(reason) = &chunk.done_reason {
            self.done_reason = Some(reason.clone());
        }
        if let Some(raw) = raw {
            let u = Usage::from_value(raw);
            if u.prompt_tokens > 0 {
                self.usage.prompt_tokens = u.prompt_tokens;
            }
            if u.completion_tokens > 0 {
                self.usage.completion_tokens = u.completion_tokens;
            }
        }
        if chunk.done {
            self.saw_terminal = true;
            return false;
        }
        true
    }

    pub fn content(&self) -> &str {
        &self.content
    }

    pub fn thinking(&self) -> &str {
        &self.thinking
    }

    pub fn tool_calls(&self) -> &[crate::llm::request::ToolCall] {
        &self.tool_calls
    }

    pub fn saw_terminal(&self) -> bool {
        self.saw_terminal
    }

    /// Consumes the assembler, producing the final response.
    pub fn finish(self, model: &str) -> Result<ChatResponse, LlmError> {
        if !self.saw_terminal {
            return Err(LlmError::IncompleteResponse(
                "the server closed the stream before sending `done`".into(),
            ));
        }
        let mut message = Message::assistant(self.content);
        message.tool_calls = self.tool_calls;
        Ok(ChatResponse {
            model: model.to_string(),
            message,
            done_reason: self.done_reason,
            usage: self.usage,
        })
    }
}

/// Shared driver for NDJSON-based providers: feeds a raw byte stream through
/// [`NdjsonBuffer`], decodes each record, forwards it to `sink` and assembles
/// the final response.
///
/// The caller supplies `read`, which pulls the next chunk from its transport.
pub fn drive_ndjson_stream<F>(
    model: &str,
    mut read: F,
    sink: &mut ChunkSink<'_>,
) -> Result<ChatResponse, LlmError>
where
    F: FnMut() -> Result<Option<Vec<u8>>, LlmError>,
{
    let mut nd = NdjsonBuffer::new();
    let mut asm = StreamAssembler::new();

    loop {
        // Flush whatever the socket produced before asking for more.
        let chunk = read()?;
        let eof = chunk.is_none();
        let records = match chunk {
            Some(bytes) => nd.push(&bytes),
            None => nd.finish().into_iter().collect(),
        };

        for record in records {
            let value = record?;
            let parsed = ChatChunk::from_value(&value)?;
            let keep_going = asm.push(parsed.clone(), Some(&value));
            if sink(&parsed).is_err() {
                return Err(LlmError::Cancelled);
            }
            if !keep_going {
                return asm.finish(model);
            }
        }

        if eof {
            // Stream ended without a `done` record: report it as incomplete but
            // let the caller decide what to do with the partial text.
            return asm.finish(model);
        }
    }
}

/// Sending helper used by tests and by future concurrent workers.
pub struct ChannelSink<'a> {
    tx: &'a Sender<ChatChunk>,
}

impl<'a> ChannelSink<'a> {
    pub fn new(tx: &'a Sender<ChatChunk>) -> Self {
        Self { tx }
    }

    /// Returns `Err(())` when the receiver has been dropped, i.e. the UI went
    /// away and generation should stop.
    ///
    /// The unit error is deliberate: `ChunkSink` uses `Err(())` as the single
    /// cancellation signal, and a richer error would imply generation failed.
    #[allow(clippy::result_unit_err)]
    pub fn send(&self, chunk: &ChatChunk) -> Result<(), ()> {
        self.tx.send(chunk.clone()).map_err(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json;

    #[test]
    fn model_name_matching_is_forgiving() {
        assert!(matches_model("qwen3.5:4b", "Qwen3.5:4B"));
        assert!(matches_model("qwen3.5:latest", "qwen3.5"));
        assert!(matches_model("qwen3.5:4b", "qwen3.5"));
        assert!(!matches_model("llama3:8b", "qwen3.5:4b"));
        assert!(!matches_model("qwen2.5:4b", "qwen3.5:4b"));
    }

    #[test]
    fn human_bytes_is_si() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(999), "999 B");
        assert_eq!(human_bytes(1000), "1.0 KB");
        assert_eq!(human_bytes(2_600_000_000), "2.6 GB");
        assert_eq!(human_bytes(150_000_000_000), "150 GB");
    }

    #[test]
    fn assembler_concatenates_and_reports_incompleteness() {
        let mut asm = StreamAssembler::new();
        let a = ChatChunk {
            content: "Hel".into(),
            ..Default::default()
        };
        let b = ChatChunk {
            content: "lo".into(),
            ..Default::default()
        };
        assert!(asm.push(a, None));
        assert!(asm.push(b, None));
        // No terminal record yet: finishing must fail loudly.
        assert!(matches!(
            asm.finish("m"),
            Err(LlmError::IncompleteResponse(_))
        ));
    }

    #[test]
    fn assembler_collects_usage_from_terminal_record() {
        let mut asm = StreamAssembler::new();
        let first = json::parse(r#"{"message":{"content":"hi"},"done":false}"#).unwrap();
        let last = json::parse(r#"{"done":true,"eval_count":9,"prompt_eval_count":4}"#).unwrap();
        asm.push(ChatChunk::from_value(&first).unwrap(), Some(&first));
        let keep = asm.push(ChatChunk::from_value(&last).unwrap(), Some(&last));
        assert!(!keep);
        let resp = asm.finish("qwen").unwrap();
        assert_eq!(resp.content(), "hi");
        assert_eq!(resp.usage.completion_tokens, 9);
        assert_eq!(resp.usage.prompt_tokens, 4);
        assert_eq!(resp.done_reason, None);
    }

    #[test]
    fn drive_reports_incomplete_when_transport_ends_early() {
        let mut chunks = vec![
            Ok(Some(
                b"{\"message\":{\"content\":\"par\"},\"done\":false}\n".to_vec(),
            )),
            Ok(None),
        ]
        .into_iter();
        let mut sink = |_: &ChatChunk| Ok(());
        let err = drive_ndjson_stream("m", || chunks.next().unwrap(), &mut sink).unwrap_err();
        assert!(matches!(err, LlmError::IncompleteResponse(_)));
    }

    #[test]
    fn drive_stops_on_cancel() {
        let texts: Vec<Vec<u8>> = vec![
            b"{\"message\":{\"content\":\"a\"},\"done\":false}\n".to_vec(),
            b"{\"message\":{\"content\":\"b\"},\"done\":false}\n".to_vec(),
            b"{\"done\":true}\n".to_vec(),
        ];
        let mut i = 0;
        let mut sink = |c: &ChatChunk| if c.content == "b" { Err(()) } else { Ok(()) };
        let err = drive_ndjson_stream(
            "m",
            || {
                let out = texts
                    .get(i)
                    .cloned()
                    .map(|t| Ok(Some(t)))
                    .unwrap_or(Ok(None));
                i += 1;
                out
            },
            &mut sink,
        )
        .unwrap_err();
        assert_eq!(err, LlmError::Cancelled);
    }
}
