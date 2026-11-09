//! Model communication: provider abstraction, request/response types, and the
//! Ollama implementation.
//!
//! The rest of the crate depends on [`provider::LlmProvider`], never on
//! [`ollama::OllamaProvider`] directly (except at the composition root in
//! `app.rs`).

pub mod ollama;
pub mod provider;
pub mod request;
pub mod response;

pub use ollama::{ModelSpec, OllamaConfig, OllamaProvider};
pub use provider::{
    ChunkSink, Health, LlmProvider, ModelInfo, StreamAssembler, human_bytes, matches_model,
};
pub use request::{ChatRequest, Message, Options, Role, ToolCall};
pub use response::{ChatChunk, ChatResponse, NdjsonBuffer, Usage};
