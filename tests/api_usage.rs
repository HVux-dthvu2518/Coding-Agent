//! Executable documentation.
//!
//! `Cargo.toml` disables doctests because `rustdoc` is not part of the offline
//! toolchain this project targets. The examples from `src/lib.rs` and the README
//! therefore live here, where `cargo test` actually runs them — documentation
//! that cannot rot.

use qwen_agent::agent::{Context, Orchestrator};
use qwen_agent::config::{Config, Overrides};
use qwen_agent::json::{self, Cursor, Value};
use qwen_agent::llm::provider::{LlmProvider, ModelInfo, matches_model};
use qwen_agent::llm::request::{ChatRequest, Message, Role};

/// From `src/lib.rs`: decoding one NDJSON record.
#[test]
fn ndjson_record_decoding() {
    let record = json::parse(r#"{"message":{"content":"hi"},"done":false}"#).unwrap();
    let cursor = Cursor::new(&record, "chat chunk");
    assert_eq!(cursor.optional_bool("done").unwrap(), Some(false));
}

/// From the README: building a request by hand.
#[test]
fn building_a_chat_request() {
    let mut request = ChatRequest::new(
        "qwen3.5:4b",
        vec![Message::system("be terse"), Message::user("hello")],
        true,
    );
    request.keep_alive = Some("10m".into());
    let wire = request.to_json();
    let parsed = json::parse(&wire).unwrap();
    assert_eq!(parsed.get("model").unwrap().as_str(), Some("qwen3.5:4b"));
    assert_eq!(parsed.get("stream").unwrap().as_bool(), Some(true));
}

/// From the README: a conversation context with a token budget.
#[test]
fn bounded_conversation_context() {
    let mut ctx = Context::new("you are a careful local assistant", 200);
    ctx.push_user("summarise this file");
    ctx.push_assistant("it is a Rust HTTP client");
    assert_eq!(ctx.messages().first().unwrap().role, Role::System);
    assert!(ctx.estimated_tokens() <= 200);
}

/// From the README: resolving configuration from a file, then validating it.
#[test]
fn configuration_loading() {
    let cfg = Config::from_toml_str(
        r#"
        [llm]
        model = "Qwen3.5:4B"
        temperature = 0.3
        "#,
    )
    .unwrap();
    assert_eq!(cfg.llm.base_url, "http://localhost:11434");
    assert!(cfg.validate().is_ok());
    // An explicit missing path is an error rather than a silent fallback.
    let err = Config::load(&Overrides {
        config_path: Some("/nonexistent/config.toml".into()),
        ..Default::default()
    })
    .unwrap_err();
    assert!(err.to_string().contains("does not exist"));
}

/// From the README: adding a provider means implementing one trait.
///
/// This is the extension point the architecture document asks for: the agent
/// core depends on `LlmProvider`, never on Ollama.
#[test]
fn a_new_provider_is_one_trait_impl() {
    struct EchoProvider;

    impl LlmProvider for EchoProvider {
        fn name(&self) -> &'static str {
            "echo"
        }
        fn model(&self) -> &str {
            "echo-1b"
        }
        fn check_connection(&self) -> Result<String, qwen_agent::error::LlmError> {
            Ok("1.0".into())
        }
        fn list_models(&self) -> Result<Vec<ModelInfo>, qwen_agent::error::LlmError> {
            Ok(vec![ModelInfo {
                name: "echo-1b".into(),
                size_bytes: 1024,
                parameter_size: None,
                quantization: None,
                family: None,
            }])
        }
        fn chat(
            &self,
            request: ChatRequest,
        ) -> Result<qwen_agent::llm::ChatResponse, qwen_agent::error::LlmError> {
            let text = request
                .messages
                .iter()
                .rev()
                .find(|m| m.role == Role::User)
                .map(|m| m.content.clone())
                .unwrap_or_default();
            Ok(qwen_agent::llm::ChatResponse {
                model: self.model().to_string(),
                message: Message::assistant(format!("echo: {text}")),
                done_reason: Some("stop".into()),
                usage: Default::default(),
            })
        }
        fn chat_stream(
            &self,
            request: ChatRequest,
            sink: &mut qwen_agent::llm::ChunkSink<'_>,
        ) -> Result<qwen_agent::llm::ChatResponse, qwen_agent::error::LlmError> {
            let response = self.chat(request)?;
            let chunk = qwen_agent::llm::ChatChunk {
                content: response.message.content.clone(),
                ..Default::default()
            };
            let _ = sink(&chunk);
            let done = qwen_agent::llm::ChatChunk {
                done: true,
                ..Default::default()
            };
            let _ = sink(&done);
            Ok(response)
        }
    }

    // The orchestrator is generic over the provider, so it works unchanged.
    let provider = EchoProvider;
    assert_eq!(provider.health().model_count, 1);
    assert!(matches_model("echo-1b", "Echo-1B"));

    let mut orchestrator = Orchestrator::new(provider, Context::new("sys", 1000), "10m");
    let mut tokens = String::new();
    let outcome = orchestrator.send("ping", &mut |event| {
        if let qwen_agent::agent::Event::Token(t) = event {
            tokens.push_str(&t);
        }
        Ok(())
    });
    assert!(outcome.is_success());
    assert_eq!(tokens, "echo: ping");
    assert_eq!(orchestrator.context().history().len(), 2);
}

/// From the README: the orchestrator is UI-agnostic — events, not print calls.
#[test]
fn orchestrator_emits_events_in_order() {
    let server = qwen_agent::server::FakeOllama::start().unwrap();
    let provider = qwen_agent::llm::OllamaProvider::new(&qwen_agent::llm::OllamaConfig {
        base_url: server.base_url(),
        model: "qwen3.5:4b".into(),
    })
    .unwrap();
    let mut orchestrator = Orchestrator::new(provider, Context::new("sys", 3000), "10m");

    let mut kinds: Vec<&'static str> = Vec::new();
    let mut text = String::new();
    orchestrator.send("ECHO events", &mut |event| {
        match &event {
            qwen_agent::agent::Event::Started { .. } => kinds.push("started"),
            qwen_agent::agent::Event::Token(t) => {
                kinds.push("token");
                text.push_str(t);
            }
            qwen_agent::agent::Event::Finished { .. } => kinds.push("finished"),
            _ => kinds.push("other"),
        }
        Ok(())
    });
    assert_eq!(kinds.first(), Some(&"started"));
    assert_eq!(kinds.last(), Some(&"finished"));
    assert!(kinds.iter().filter(|k| **k == "token").count() > 1);
    assert!(text.contains("You said: ECHO events"));
}

/// From the README: JSON values are the currency of tool definitions (M1).
#[test]
fn tool_definition_schema_is_plain_json() {
    let schema = json::parse(
        r#"{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}"#,
    )
    .unwrap();
    assert_eq!(schema.get("type").unwrap().as_str(), Some("object"));
    let required = schema.get("required").unwrap().as_array().unwrap();
    assert_eq!(required[0].as_str(), Some("path"));
    // Values round-trip through the serializer unchanged.
    assert_eq!(json::parse(&json::to_string(&schema)).unwrap(), schema);
    assert!(matches!(Value::Null, Value::Null));
}
