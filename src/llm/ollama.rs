//! Ollama implementation of [`LlmProvider`].
//!
//! Endpoints used:
//!
//! | Endpoint          | Method | Purpose                  |
//! |-------------------|--------|--------------------------|
//! | `/api/version`    | GET    | liveness / version       |
//! | `/api/tags`       | GET    | installed models         |
//! | `/api/chat`       | POST   | chat completion (NDJSON) |
//! | `/api/show`       | POST   | model metadata (M2)      |
//!
//! Error mapping is the important part here: every transport failure is
//! classified into a distinct [`LlmError`] so the CLI can print advice the user
//! can act on instead of a generic "request failed".

use crate::error::{HttpError, LlmError};
use crate::http::{self, Response, Url};
use crate::json::{self, Cursor, Value};
use crate::llm::provider::{ChunkSink, LlmProvider, ModelInfo, drive_ndjson_stream};
use crate::llm::request::ChatRequest;
use crate::llm::response::ChatResponse;

/// Configuration for the Ollama backend.
#[derive(Debug, Clone, PartialEq)]
pub struct OllamaConfig {
    /// e.g. `http://localhost:11434`
    pub base_url: String,
    /// e.g. `qwen3.5:4b`
    pub model: String,
}

/// A provider that talks to a local Ollama server over HTTP/1.1.
#[derive(Debug, Clone)]
pub struct OllamaProvider {
    base: Url,
    base_url: String,
    model: String,
}

impl OllamaProvider {
    /// Validates the base URL up front so `doctor` can fail before the first
    /// chat request.
    pub fn new(config: &OllamaConfig) -> Result<OllamaProvider, LlmError> {
        let base = Url::parse(&config.base_url).map_err(|e| LlmError::Protocol(e.to_string()))?;
        Ok(OllamaProvider {
            base,
            base_url: config.base_url.clone(),
            model: config.model.clone(),
        })
    }

    /// The configured base URL, as typed by the user.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    fn url(&self, endpoint: &str) -> Url {
        self.base.join(endpoint)
    }

    /// Maps a transport failure onto `LlmError`, filling in the URL where the
    /// lower layer could not know it.
    fn map_http_err(&self, err: HttpError) -> LlmError {
        match err {
            HttpError::Connect(m) => LlmError::Connect {
                url: self.base_url.clone(),
                source: m,
            },
            other => LlmError::from(other),
        }
    }

    fn get_json(&self, endpoint: &str) -> Result<Value, LlmError> {
        let url = self.url(endpoint);
        let mut resp = http::get(&url).map_err(|e| self.map_http_err(e))?;
        self.decode_response(endpoint, &mut resp)
    }

    fn post_json(&self, endpoint: &str, body: &str) -> Result<Response, LlmError> {
        let url = self.url(endpoint);
        http::post_json(&url, body.as_bytes()).map_err(|e| self.map_http_err(e))
    }

    /// Reads a complete (non-streaming) response body, mapping status codes.
    fn decode_response(&self, endpoint: &str, resp: &mut Response) -> Result<Value, LlmError> {
        let status = resp.status();
        let body = resp.read_to_string().map_err(|e| self.map_http_err(e))?;
        if !(200..300).contains(&status) {
            return Err(self.status_error(status, &body));
        }
        json::parse(&body).map_err(|e| {
            LlmError::Decode(format!("{endpoint}: {e} (body: {})", truncate(&body, 200)))
        })
    }

    /// Turns a non-2xx response into the most specific error we can justify.
    ///
    /// Ollama reports a missing model as `404 {"error":"model 'x' not found ..."}`,
    /// but also uses 404 for unknown routes; the body is what disambiguates.
    fn status_error(&self, status: u16, body: &str) -> LlmError {
        let parsed_error = json::parse(body)
            .ok()
            .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(str::to_string));
        let message = parsed_error.unwrap_or_else(|| truncate(body, 300).to_string());
        let lowered = message.to_ascii_lowercase();
        if lowered.contains("not found") && lowered.contains("model") {
            // Extract the model name Ollama quoted, falling back to ours.
            let model = message
                .split('\'')
                .nth(1)
                .map(str::to_string)
                .unwrap_or_else(|| self.model.clone());
            return LlmError::ModelNotFound { model };
        }
        LlmError::Http {
            status,
            body: message,
        }
    }

    /// `GET /api/tags`
    pub fn list_models_verbose(&self) -> Result<Vec<ModelInfo>, LlmError> {
        let value = self.get_json("/api/tags")?;
        let c = Cursor::new(&value, "tags response");
        let mut models = Vec::new();
        for item in c.object_array("models")? {
            let name = match item
                .get("name")
                .or_else(|| item.get("model"))
                .and_then(Value::as_str)
            {
                Some(name) => name.to_string(),
                None => continue, // a model entry without a name is unusable
            };
            let ic = Cursor::new(item, "model entry");
            let details = item.get("details");
            let detail = |key: &str| {
                details
                    .and_then(|d| d.get(key))
                    .and_then(Value::as_str)
                    .map(str::to_string)
            };
            models.push(ModelInfo {
                name,
                size_bytes: ic.optional_u64("size").ok().flatten().unwrap_or(0),
                parameter_size: detail("parameter_size"),
                quantization: detail("quantization_level"),
                family: detail("family"),
            });
        }
        Ok(models)
    }

    /// `POST /api/show` — model template and capabilities. Used by `doctor` to
    /// warn when the configured model cannot do tool calling (an M1 concern
    /// that is cheaper to surface now).
    pub fn show(&self, model: &str) -> Result<ModelSpec, LlmError> {
        let body = json::to_string(&Value::object([("model", Value::str(model))]));
        let mut resp = self.post_json("/api/show", &body)?;
        let value = self.decode_response("/api/show", &mut resp)?;
        let c = Cursor::new(&value, "show response");
        let detail = |key: &str| {
            value
                .get("details")
                .and_then(|d| d.get(key))
                .and_then(Value::as_str)
                .map(str::to_string)
        };
        Ok(ModelSpec {
            family: detail("family"),
            parameter_size: detail("parameter_size"),
            capabilities: c.str_array("capabilities")?,
            template: c.optional_str("template")?.map(str::to_string),
        })
    }
}

/// What `/api/show` told us about a model.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelSpec {
    pub family: Option<String>,
    pub parameter_size: Option<String>,
    pub capabilities: Vec<String>,
    pub template: Option<String>,
}

impl ModelSpec {
    /// True when the server advertises tool-calling support for this model.
    /// Absence of the list is treated as "unknown", not "no".
    pub fn supports_tools(&self) -> Option<bool> {
        if self.capabilities.is_empty() {
            None
        } else {
            Some(self.capabilities.iter().any(|c| c == "tools"))
        }
    }
}

impl LlmProvider for OllamaProvider {
    fn name(&self) -> &'static str {
        "ollama"
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn check_connection(&self) -> Result<String, LlmError> {
        let value = self.get_json("/api/version")?;
        let c = Cursor::new(&value, "version response");
        Ok(c.optional_str("version")?.unwrap_or("unknown").to_string())
    }

    fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        OllamaProvider::list_models_verbose(self)
    }

    fn chat(&self, mut request: ChatRequest) -> Result<ChatResponse, LlmError> {
        request.stream = false;
        let mut resp = self.post_json("/api/chat", &request.to_json())?;
        let value = self.decode_response("/api/chat", &mut resp)?;
        let mut response = ChatResponse::from_value(&value)?;
        if response.model.is_empty() {
            response.model = request.model;
        }
        Ok(response)
    }

    fn chat_stream(
        &self,
        mut request: ChatRequest,
        sink: &mut ChunkSink<'_>,
    ) -> Result<ChatResponse, LlmError> {
        request.stream = true;
        let model = request.model.clone();
        let mut resp = self.post_json("/api/chat", &request.to_json())?;

        let status = resp.status();
        if !(200..300).contains(&status) {
            let body = resp.read_to_string().unwrap_or_default();
            return Err(self.status_error(status, &body));
        }

        // Read raw bytes (not lines) so the NDJSON framer is exercised exactly
        // as it would be with a socket whose chunking is arbitrary.
        let mut buf = [0u8; 8192];
        let mut read = || -> Result<Option<Vec<u8>>, LlmError> {
            match resp.read(&mut buf) {
                Ok(0) => Ok(None),
                Ok(n) => Ok(Some(buf[..n].to_vec())),
                Err(e) => Err(LlmError::from(e)),
            }
        };

        drive_ndjson_stream(&model, &mut read, sink)
    }
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider(model: &str) -> OllamaProvider {
        OllamaProvider::new(&OllamaConfig {
            base_url: "http://localhost:11434".into(),
            model: model.into(),
        })
        .unwrap()
    }

    #[test]
    fn builds_endpoint_urls_from_base() {
        let p = provider("qwen3.5:4b");
        assert_eq!(p.url("/api/chat").path(), "/api/chat");
        assert_eq!(p.base_url(), "http://localhost:11434");
    }

    #[test]
    fn rejects_invalid_base_url() {
        let err = OllamaProvider::new(&OllamaConfig {
            base_url: "localhost:11434".into(),
            model: "m".into(),
        })
        .unwrap_err();
        assert!(matches!(err, LlmError::Protocol(_)));
    }

    #[test]
    fn missing_model_404_maps_to_model_not_found() {
        let p = provider("qwen3.5:4b");
        let err = p.status_error(
            404,
            r#"{"error":"model 'qwen3.5:4b' not found, try pulling it first"}"#,
        );
        match err {
            LlmError::ModelNotFound { model } => assert_eq!(model, "qwen3.5:4b"),
            other => panic!("expected ModelNotFound, got {other:?}"),
        }
    }

    #[test]
    fn other_errors_keep_status_and_body() {
        let p = provider("m");
        let err = p.status_error(500, r#"{"error":"out of memory"}"#);
        match err {
            LlmError::Http { status, body } => {
                assert_eq!(status, 500);
                assert_eq!(body, "out of memory");
            }
            other => panic!("expected Http, got {other:?}"),
        }
    }

    #[test]
    fn non_json_error_body_is_preserved() {
        let p = provider("m");
        let err = p.status_error(502, "<html>bad gateway</html>");
        assert!(matches!(err, LlmError::Http { status: 502, .. }));
    }

    #[test]
    fn model_spec_tool_support_is_tri_state() {
        let mut spec = ModelSpec::default();
        assert_eq!(spec.supports_tools(), None);
        spec.capabilities = vec!["completion".into(), "tools".into()];
        assert_eq!(spec.supports_tools(), Some(true));
        spec.capabilities = vec!["completion".into()];
        assert_eq!(spec.supports_tools(), Some(false));
    }

    #[test]
    fn list_models_parses_tag_payload() {
        // The decoder is exercised directly; the transport is covered by the
        // integration tests against the fake server.
        let raw = json::parse(
            r#"{"models":[
                {"name":"qwen3.5:4b","size":2600000000,
                 "details":{"parameter_size":"4.0B","quantization_level":"Q4_K_M","family":"qwen3"}},
                {"name":"llama3:8b","size":4700000000}
            ]}"#,
        )
        .unwrap();
        let c = Cursor::new(&raw, "tags response");
        let entries = c.object_array("models").unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(
            entries[0]
                .get("details")
                .unwrap()
                .get("parameter_size")
                .unwrap()
                .as_str(),
            Some("4.0B")
        );
    }

    #[test]
    fn truncate_respects_char_boundaries() {
        let s = "héllo wörld";
        let t = truncate(s, 2);
        assert!(s.is_char_boundary(t.len()));
        assert_eq!(truncate("abc", 10), "abc");
    }
}
