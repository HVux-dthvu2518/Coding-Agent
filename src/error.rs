//! Error types for the agent.
//!
//! The design follows the common Rust split:
//!
//! * subsystem errors are **typed** (`ConfigError`, `HttpError`, `JsonError`,
//!   ...), enumerated so callers can react differently to each case;
//! * the application boundary uses [`AppError`], which implements
//!   `std::error::Error` and exposes a human-readable [`AppError::hint`]
//!   telling the user what to do next.
//!
//! Nothing in this crate calls `process::exit` outside `main`, so every error
//! is testable.

use std::fmt;

/// Errors produced while locating, reading or validating configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// No configuration file was found at any of the probed locations.
    Missing,
    /// The file exists but could not be read.
    Unreadable(String),
    /// The file could not be parsed.
    Parse(String),
    /// A value parsed but is outside the accepted domain.
    Invalid { field: String, reason: String },
    /// Same as [`ConfigError::Invalid`], but attributed to a file.
    InvalidInFile {
        path: String,
        field: String,
        reason: String,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::Missing => write!(f, "no configuration file found"),
            ConfigError::Unreadable(e) => write!(f, "configuration is not readable: {e}"),
            ConfigError::Parse(e) => write!(f, "configuration is malformed: {e}"),
            ConfigError::Invalid { field, reason } => {
                write!(f, "invalid configuration value for `{field}`: {reason}")
            }
            ConfigError::InvalidInFile {
                path,
                field,
                reason,
            } => {
                write!(f, "{path}: invalid value for `{field}`: {reason}")
            }
        }
    }
}

impl std::error::Error for ConfigError {}

/// Why a model call failed.
///
/// These variants are deliberately distinct: "Ollama is not running", "Ollama
/// is running but the model is not pulled" and "the model returned something
/// we could not parse" need very different user-facing advice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlmError {
    /// Could not open a TCP connection to the configured base URL.
    Connect { url: String, source: String },
    /// Connected, but the server did not answer with a complete HTTP response.
    Protocol(String),
    /// The server answered with a non-2xx status.
    Http { status: u16, body: String },
    /// The server answered 2xx with a body we could not deserialize.
    Decode(String),
    /// The configured model is not present on the server.
    ModelNotFound { model: String },
    /// The response stream ended before the terminating record arrived.
    IncompleteResponse(String),
    /// The caller cancelled the request (e.g. Ctrl-C).
    Cancelled,
    /// I/O failure while reading or writing a socket.
    Io(String),
}

impl LlmError {
    /// Short, actionable advice shown under the error banner.
    pub fn hint(&self) -> &'static str {
        match self {
            LlmError::Connect { .. } => {
                "Is Ollama running? Start it with `ollama serve`, then retry."
            }
            LlmError::Protocol(_) => {
                "The endpoint answered, but not with HTTP/1.1. Check `llm.base_url`."
            }
            LlmError::Http { status: 404, .. } => {
                "Ollama returned 404 for the chat endpoint. Check that `llm.base_url` has no path suffix."
            }
            LlmError::Http { .. } => "Ollama rejected the request. See the response body above.",
            LlmError::Decode(_) => "Ollama returned a body that is not valid JSON.",
            LlmError::ModelNotFound { .. } => {
                "Pull the model first: `ollama pull Qwen3.5:4B` (or run `qwen-agent models`)."
            }
            LlmError::IncompleteResponse(_) => {
                "The stream stopped early. The reply shown above is partial."
            }
            LlmError::Cancelled => "Request cancelled.",
            LlmError::Io(_) => "Local socket error. Check the Ollama host/port.",
        }
    }

    /// True when retrying the same request could plausibly succeed.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            LlmError::Connect { .. } | LlmError::Io(_) | LlmError::IncompleteResponse(_)
        )
    }
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlmError::Connect { url, source } => {
                write!(f, "cannot reach Ollama at {url} ({source})")
            }
            LlmError::Protocol(m) => write!(f, "malformed HTTP response: {m}"),
            LlmError::Http { status, body } => {
                if body.is_empty() {
                    write!(f, "Ollama returned HTTP {status}")
                } else {
                    write!(f, "Ollama returned HTTP {status}: {body}")
                }
            }
            LlmError::Decode(m) => write!(f, "could not decode Ollama response: {m}"),
            LlmError::ModelNotFound { model } => {
                write!(
                    f,
                    "model `{model}` is not installed on this Ollama instance"
                )
            }
            LlmError::IncompleteResponse(m) => write!(f, "incomplete response: {m}"),
            LlmError::Cancelled => write!(f, "request cancelled"),
            LlmError::Io(m) => write!(f, "socket error: {m}"),
        }
    }
}

impl std::error::Error for LlmError {}

/// Errors from the module that transports bytes to Ollama.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HttpError {
    /// TCP/TLS connect failed.
    Connect(String),
    /// The peer closed or the bytes did not form a valid HTTP/1.1 message.
    Protocol(String),
    /// Writing the request failed.
    Write(String),
    /// Reading the response failed.
    Read(String),
    /// The URL was not an absolute `http://host[:port][/path]`.
    BadUrl(String),
    /// A configured timeout elapsed.
    Timeout(String),
}

impl fmt::Display for HttpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HttpError::Connect(m) => write!(f, "connect failed: {m}"),
            HttpError::Protocol(m) => write!(f, "HTTP protocol error: {m}"),
            HttpError::Write(m) => write!(f, "request write failed: {m}"),
            HttpError::Read(m) => write!(f, "response read failed: {m}"),
            HttpError::BadUrl(m) => write!(f, "invalid URL: {m}"),
            HttpError::Timeout(m) => write!(f, "timeout: {m}"),
        }
    }
}

impl std::error::Error for HttpError {}

/// Errors from [`crate::json`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonError {
    /// Byte offset into the input where parsing failed.
    pub offset: usize,
    /// Human readable reason.
    pub message: String,
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "JSON error at byte {}: {}", self.offset, self.message)
    }
}

impl std::error::Error for JsonError {}

/// Errors from [`crate::toml`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TomlError {
    /// 1-based line number.
    pub line: usize,
    /// Human readable reason.
    pub message: String,
}

impl fmt::Display for TomlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TOML error on line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for TomlError {}

/// Top-level error for the binary.
#[derive(Debug)]
pub enum AppError {
    Config(ConfigError),
    Llm(LlmError),
    Io(std::io::Error),
    /// The user asked for something the CLI does not support.
    Usage(String),
    /// Anything surfaced from a subsystem that already formatted itself.
    Other(String),
}

impl AppError {
    /// One-line advice for the user, or `None` when the message speaks for itself.
    pub fn hint(&self) -> Option<&str> {
        match self {
            AppError::Config(ConfigError::Missing) => {
                Some("Run `qwen-agent config --init` to write a default config file.")
            }
            AppError::Config(ConfigError::Invalid { .. }) => {
                Some("Run `qwen-agent config` to inspect it.")
            }
            AppError::Config(_) => None,
            AppError::Llm(e) => Some(e.hint()),
            AppError::Io(_) => None,
            AppError::Usage(_) => Some("Run `qwen-agent --help`."),
            AppError::Other(_) => None,
        }
    }
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::Config(e) => write!(f, "{e}"),
            AppError::Llm(e) => write!(f, "{e}"),
            AppError::Io(e) => write!(f, "{e}"),
            AppError::Usage(m) => write!(f, "{m}"),
            AppError::Other(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for AppError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AppError::Config(e) => Some(e),
            AppError::Llm(e) => Some(e),
            AppError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ConfigError> for AppError {
    fn from(e: ConfigError) -> Self {
        AppError::Config(e)
    }
}

impl From<LlmError> for AppError {
    fn from(e: LlmError) -> Self {
        AppError::Llm(e)
    }
}

impl From<std::io::Error> for AppError {
    fn from(e: std::io::Error) -> Self {
        AppError::Io(e)
    }
}

impl From<HttpError> for LlmError {
    fn from(e: HttpError) -> Self {
        match e {
            HttpError::Connect(m) => LlmError::Connect {
                url: String::new(), // filled in by the caller, which knows the URL
                source: m,
            },
            HttpError::BadUrl(m) => LlmError::Protocol(m),
            HttpError::Protocol(m) => LlmError::Protocol(m),
            HttpError::Write(m) | HttpError::Read(m) => LlmError::Io(m),
            HttpError::Timeout(m) => LlmError::Io(format!("timed out: {m}")),
        }
    }
}

impl From<JsonError> for LlmError {
    fn from(e: JsonError) -> Self {
        LlmError::Decode(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_not_found_gets_pull_hint() {
        let e = LlmError::ModelNotFound {
            model: "qwen3.5:4b".into(),
        };
        assert!(e.hint().contains("ollama pull"));
        assert!(!e.is_retryable());
    }

    #[test]
    fn connect_is_retryable() {
        let e = LlmError::Connect {
            url: "http://x".into(),
            source: "refused".into(),
        };
        assert!(e.is_retryable());
    }

    #[test]
    fn app_error_exposes_hint_through_wrapper() {
        let e = AppError::Llm(LlmError::ModelNotFound { model: "m".into() });
        assert!(e.hint().is_some());
    }
}
