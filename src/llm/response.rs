//! Normalized chat responses and the NDJSON stream parser.
//!
//! Ollama answers `/api/chat` with newline-delimited JSON: one record per line,
//! terminated by a record with `"done": true`. TCP chunks do **not** line up
//! with records, so [`NdjsonBuffer`] keeps partial lines between reads — this is
//! the single most common source of "it works until the tokens get fast" bugs.

use crate::error::JsonError;
use crate::json::{self, Cursor, Value};
use crate::llm::request::{Message, ToolCall};

/// Incremental handler for a newline-delimited JSON stream.
///
/// Feed it whatever the socket returned with [`NdjsonBuffer::push`]; it returns
/// every **complete** record. Call [`NdjsonBuffer::finish`] at EOF to flush a
/// final record that lacks a trailing newline.
#[derive(Debug, Default)]
pub struct NdjsonBuffer {
    buf: Vec<u8>,
}

impl NdjsonBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a chunk and returns the complete lines it contains.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<Result<Value, JsonError>> {
        self.buf.extend_from_slice(chunk);
        self.drain_complete()
    }

    /// Flushes any remaining bytes as a final record (used at stream EOF).
    pub fn finish(&mut self) -> Option<Result<Value, JsonError>> {
        if self.buf.iter().all(|b| b.is_ascii_whitespace()) {
            self.buf.clear();
            return None;
        }
        let rest = std::mem::take(&mut self.buf);
        Some(parse_line(&rest))
    }

    /// Bytes held back for an incomplete record. Exposed for diagnostics.
    pub fn pending(&self) -> &[u8] {
        &self.buf
    }

    fn drain_complete(&mut self) -> Vec<Result<Value, JsonError>> {
        let mut out = Vec::new();
        while let Some(idx) = self.buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=idx).collect();
            let line = &line[..line.len() - 1];
            let line = trim_cr(line);
            if line.iter().all(|b| b.is_ascii_whitespace()) {
                continue;
            }
            out.push(parse_line(line));
        }
        out
    }
}

fn trim_cr(line: &[u8]) -> &[u8] {
    match line.last() {
        Some(b'\r') => &line[..line.len() - 1],
        _ => line,
    }
}

fn parse_line(line: &[u8]) -> Result<Value, JsonError> {
    json::parse_bytes(line)
}

/// One record from the stream.
/// Used by the M1 tool loop to tell "the model wants a tool" from "the model is
/// answering" without buffering the stream first.
#[derive(Debug, Clone, Default)]
pub struct ChatChunk {
    /// Incremental content since the previous chunk.
    pub content: String,
    /// Reasoning trace, when the model emits one.
    pub thinking: String,
    /// Tool calls requested by the model.
    pub tool_calls: Vec<ToolCall>,
    /// True on the terminating record.
    pub done: bool,
    /// Why generation stopped: `stop`, `length`, `load`, ...
    pub done_reason: Option<String>,
}

impl ChatChunk {
    /// Decodes one NDJSON record.
    pub fn from_value(v: &Value) -> Result<ChatChunk, JsonError> {
        let c = Cursor::new(v, "chat chunk");
        // Errors reported mid-stream arrive as `{"error": "..."}`.
        if let Some(error) = c.optional_str("error")? {
            return Err(JsonError {
                offset: 0,
                message: format!("ollama stream error: {error}"),
            });
        }
        let message = v.get("message");
        let mut chunk = ChatChunk::default();
        if let Some(message) = message {
            if !message.is_null() {
                let m = Message::from_value(message);
                chunk.content = m.content;
                chunk.tool_calls = m.tool_calls;
                chunk.thinking = c.optional_str("thinking")?.unwrap_or_default().to_string();
            }
        }
        chunk.done = c.optional_bool("done")?.unwrap_or(false);
        chunk.done_reason = c.optional_str("done_reason")?.map(str::to_string);
        Ok(chunk)
    }

    pub fn is_terminal(&self) -> bool {
        self.done
    }
}

/// Usage counters reported on the final record.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

impl Usage {
    pub fn from_value(v: &Value) -> Usage {
        let c = Cursor::new(v, "chat response");
        Usage {
            prompt_tokens: c
                .optional_u64("prompt_eval_count")
                .ok()
                .flatten()
                .unwrap_or(0),
            completion_tokens: c.optional_u64("eval_count").ok().flatten().unwrap_or(0),
        }
    }
}

/// A complete (non-streamed) chat response.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatResponse {
    pub model: String,
    pub message: Message,
    pub done_reason: Option<String>,
    pub usage: Usage,
}

impl ChatResponse {
    pub fn content(&self) -> &str {
        &self.message.content
    }

    pub fn from_value(v: &Value) -> Result<ChatResponse, JsonError> {
        let c = Cursor::new(v, "chat response");
        if let Some(error) = c.optional_str("error")? {
            return Err(JsonError {
                offset: 0,
                message: format!("ollama error: {error}"),
            });
        }
        let message = match v.get("message") {
            Some(m) => Message::from_value(m),
            None => Message::assistant(c.optional_str("response")?.unwrap_or_default()),
        };
        Ok(ChatResponse {
            model: c.optional_str("model")?.unwrap_or_default().to_string(),
            message,
            done_reason: c.optional_str("done_reason")?.map(str::to_string),
            usage: Usage::from_value(v),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(results: Vec<Result<Value, JsonError>>) -> Vec<Value> {
        results
            .into_iter()
            .map(|r| r.expect("valid record"))
            .collect()
    }

    #[test]
    fn splits_records_regardless_of_chunk_boundaries() {
        let mut nd = NdjsonBuffer::new();
        // Same payload, split at every possible boundary: every split must
        // yield nothing until the newline arrives, then exactly one record.
        let stream = b"{\"a\":1}\n{\"b\":2}\n";
        assert!(values(nd.push(&stream[..3])).is_empty());
        let first = values(nd.push(&stream[3..8]));
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].get("a").unwrap().as_f64(), Some(1.0));
        let rest = values(nd.push(&stream[8..]));
        assert_eq!(rest.len(), 1);
        assert_eq!(rest[0].get("b").unwrap().as_f64(), Some(2.0));
        assert!(nd.pending().is_empty());
    }

    #[test]
    fn handles_crlf_and_blank_lines() {
        let mut nd = NdjsonBuffer::new();
        let recs = values(nd.push(b"{\"a\":1}\r\n\r\n{\"b\":2}\r\n"));
        assert_eq!(recs.len(), 2);
    }

    #[test]
    fn reports_bad_record_without_losing_the_next_one() {
        let mut nd = NdjsonBuffer::new();
        let recs = nd.push(b"not json\n{\"ok\":true}\n");
        assert_eq!(recs.len(), 2);
        assert!(recs[0].is_err());
        assert!(recs[1].is_ok());
    }

    #[test]
    fn finish_flushes_trailing_record_without_newline() {
        let mut nd = NdjsonBuffer::new();
        assert!(values(nd.push(b"{\"done\":true}")).is_empty());
        let last = nd.finish().expect("flushed").expect("valid");
        assert_eq!(last.get("done").unwrap().as_bool(), Some(true));
        assert!(nd.finish().is_none());
    }

    #[test]
    fn decodes_content_and_done_flag() {
        let v = json::parse(
            r#"{"model":"m","message":{"role":"assistant","content":"Hel"},"done":false}"#,
        )
        .unwrap();
        let chunk = ChatChunk::from_value(&v).unwrap();
        assert_eq!(chunk.content, "Hel");
        assert!(!chunk.done);

        let v = json::parse(r#"{"done":true,"done_reason":"stop","eval_count":7}"#).unwrap();
        let chunk = ChatChunk::from_value(&v).unwrap();
        assert!(chunk.done);
        assert_eq!(chunk.done_reason.as_deref(), Some("stop"));
        assert_eq!(Usage::from_value(&v).completion_tokens, 7);
    }

    #[test]
    fn surfaces_midstream_errors() {
        let v = json::parse(r#"{"error":"model requires more system memory"}"#).unwrap();
        let err = ChatChunk::from_value(&v).unwrap_err();
        assert!(err.message.contains("more system memory"), "{err}");
    }

    #[test]
    fn decodes_non_streamed_response() {
        let v = json::parse(
            r#"{"model":"qwen","message":{"role":"assistant","content":"hi"},"done":true,"prompt_eval_count":12,"eval_count":3}"#,
        )
        .unwrap();
        let r = ChatResponse::from_value(&v).unwrap();
        assert_eq!(r.content(), "hi");
        assert_eq!(r.usage.prompt_tokens, 12);
        assert_eq!(r.usage.completion_tokens, 3);
    }

    #[test]
    fn legacy_generate_style_response_is_tolerated() {
        let v = json::parse(r#"{"response":"hello","done":true}"#).unwrap();
        let r = ChatResponse::from_value(&v).unwrap();
        assert_eq!(r.content(), "hello");
    }
}
