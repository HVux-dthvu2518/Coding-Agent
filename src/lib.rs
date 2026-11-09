//! # qwen-agent
//!
//! A local-first CLI agent for Qwen3.5:4B served by Ollama.
//!
//! The crate is organised in six subsystems, matching the architecture document:
//!
//! | Module | Responsibility |
//! |--------|----------------|
//! | [`llm`] | provider abstraction + Ollama implementation, streaming, NDJSON |
//! | [`agent`] | context, state machine, planner/reflection seams, orchestrator |
//! | [`cli`] | argument parsing, REPL, slash commands, `doctor`, terminal styling |
//! | [`config`] | TOML + env + flag resolution and validation |
//! | [`http`] | minimal HTTP/1.1 client (chunked bodies, timeouts) |
//! | [`json`] / [`toml`] | dependency-free parsers used by the wire protocol |
//!
//! ## Data flow
//!
//! ```text
//! main ─► cli::args::parse_from ─► app::run ─► Session ─► Orchestrator ─► LlmProvider
//!                                                    │                        │
//!                                                    │◄──── Event stream ◄─────┘
//!                                                    ▼
//!                                              ui::render (stdout)
//! ```
//!
//! ## Invariants worth preserving
//!
//! * `llm` never depends on `agent` or `cli`; the provider is swappable.
//! * `agent` never writes to a terminal: it emits [`agent::Event`]s.
//! * model output goes to stdout; logs and diagnostics go to stderr.
//! * a failed turn is rolled back out of the conversation context.
//!
//! ## Example
//!
//! ```text
//! use qwen_agent::json::{self, Cursor};
//!
//! // One NDJSON record from an Ollama chat stream.
//! let record = json::parse(r#"{"message":{"content":"hi"},"done":false}"#).unwrap();
//! let cursor = Cursor::new(&record, "chat chunk");
//! assert_eq!(cursor.optional_bool("done").unwrap(), Some(false));
//! ```
//!
//! The same code is executed by `tests/api_usage.rs`, so the example cannot rot
//! even though `rustdoc` is unavailable in the target toolchain.

pub mod agent;
pub mod app;
pub mod cli;
pub mod config;
pub mod config_template;
pub mod error;
pub mod http;
pub mod json;
pub mod llm;
pub mod logging;
pub mod server;
pub mod toml;

/// Crate version, re-exported for banners and `--version`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
