//! Provider-level integration tests against the scripted server.
//!
//! These exercise the layer between the CLI and the wire: NDJSON framing over
//! chunked transfer encoding, tool-call parsing, error classification, and
//! cancellation. The fake server splits its output at arbitrary byte offsets, so
//! the tests genuinely cover "HTTP chunks do not align with JSON records".

use qwen_agent::error::LlmError;
use qwen_agent::llm::provider::{LlmProvider, drive_ndjson_stream};
use qwen_agent::llm::request::{ChatRequest, Message};
use qwen_agent::llm::{OllamaConfig, OllamaProvider};
use qwen_agent::server::FakeOllama;

fn provider(server: &FakeOllama, model: &str) -> OllamaProvider {
    OllamaProvider::new(&OllamaConfig {
        base_url: server.base_url(),
        model: model.to_string(),
    })
    .expect("valid base url")
}

fn request(model: &str, prompt: &str) -> ChatRequest {
    ChatRequest::new(
        model,
        vec![Message::system("be brief"), Message::user(prompt)],
        true,
    )
}

#[test]
fn reports_the_server_version() {
    let server = FakeOllama::start().unwrap();
    let version = provider(&server, "qwen3.5:4b").check_connection().unwrap();
    assert_eq!(version, "0.9.9-fake");
}

#[test]
fn lists_models_with_metadata() {
    let server = FakeOllama::start().unwrap();
    let models = provider(&server, "qwen3.5:4b").list_models().unwrap();
    assert_eq!(models.len(), 2);
    let qwen = models.iter().find(|m| m.name == "qwen3.5:4b").unwrap();
    assert_eq!(qwen.parameter_size.as_deref(), Some("4.0B"));
    assert_eq!(qwen.quantization.as_deref(), Some("Q4_K_M"));
    assert_eq!(qwen.human_size(), "2.6 GB");
}

#[test]
fn model_exists_matches_case_and_tags() {
    let server = FakeOllama::start().unwrap();
    assert!(provider(&server, "Qwen3.5:4B").model_exists().unwrap());
    assert!(provider(&server, "qwen3.5").model_exists().unwrap());
    assert!(!provider(&server, "qwen3.5:72b").model_exists().unwrap());
}

#[test]
fn streams_content_token_by_token() {
    let server = FakeOllama::start().unwrap();
    let provider = provider(&server, "qwen3.5:4b");
    let mut chunks = Vec::new();
    let reply = provider
        .chat_stream(request("qwen3.5:4b", "ECHO streamed"), &mut |chunk| {
            if !chunk.content.is_empty() {
                chunks.push(chunk.content.clone());
            }
            Ok(())
        })
        .unwrap();

    assert!(chunks.len() > 3, "expected several records, got {chunks:?}");
    assert_eq!(chunks.concat(), reply.content());
    assert!(reply.content().contains("You said: ECHO streamed"));
    assert_eq!(reply.done_reason.as_deref(), Some("stop"));
    assert_eq!(reply.usage.completion_tokens, 42);
    assert_eq!(reply.usage.prompt_tokens, 31);
}

#[test]
fn non_streaming_chat_is_not_supported_by_the_fake_server_but_errors_cleanly() {
    let server = FakeOllama::start().unwrap();
    let provider = provider(&server, "qwen3.5:4b");
    // The fake always streams; the provider asks for stream=false but the
    // server's chunks still arrive — this asserts we do not panic or hang and
    // that the decoded body is reported, not silently dropped.
    let mut req = request("qwen3.5:4b", "hello");
    req.stream = false;
    match provider.chat(req) {
        Ok(response) => assert!(!response.model.is_empty()),
        Err(LlmError::Decode(_)) => { /* multi-record body: acceptable, still typed */ }
        Err(other) => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn missing_model_is_classified_precisely() {
    let server = FakeOllama::start().unwrap();
    let provider = provider(&server, "qwen3.5:4b");
    let err = provider
        .chat_stream(request("not-a-model:1b", "hi"), &mut |_| Ok(()))
        .unwrap_err();
    match err {
        LlmError::ModelNotFound { model } => assert_eq!(model, "not-a-model:1b"),
        other => panic!("expected ModelNotFound, got {other:?}"),
    }
}

#[test]
fn server_errors_keep_status_and_body() {
    let server = FakeOllama::start().unwrap();
    let provider = provider(&server, "qwen3.5:4b");
    let err = provider
        .chat_stream(request("qwen3.5:4b", "FAIL_500 please"), &mut |_| Ok(()))
        .unwrap_err();
    match err {
        LlmError::Http { status, body } => {
            assert_eq!(status, 500);
            assert_eq!(body, "out of memory");
        }
        other => panic!("expected Http, got {other:?}"),
    }
}

#[test]
fn midstream_error_records_are_surfaced() {
    let server = FakeOllama::start().unwrap();
    let provider = provider(&server, "qwen3.5:4b");
    let err = provider
        .chat_stream(request("qwen3.5:4b", "EMIT_ERROR"), &mut |_| Ok(()))
        .unwrap_err();
    assert!(err.to_string().contains("model crashed"), "{err}");
}

#[test]
fn tool_calls_are_parsed_from_the_stream() {
    let server = FakeOllama::start().unwrap();
    let provider = provider(&server, "qwen3.5:4b");
    let reply = provider
        .chat_stream(request("qwen3.5:4b", "TOOLS"), &mut |_| Ok(()))
        .unwrap();
    assert_eq!(reply.message.tool_calls.len(), 1);
    let call = &reply.message.tool_calls[0];
    assert_eq!(call.name, "read_file");
    assert_eq!(
        call.arguments.get("path").unwrap().as_str(),
        Some("src/main.rs")
    );
}

#[test]
fn cancelled_streams_stop_early_and_report_cancellation() {
    let server = FakeOllama::start().unwrap();
    let provider = provider(&server, "qwen3.5:4b");
    let mut seen = 0;
    let err = provider
        .chat_stream(request("qwen3.5:4b", "ECHO cancel me"), &mut |chunk| {
            if !chunk.content.is_empty() {
                seen += 1;
                if seen == 2 {
                    return Err(());
                }
            }
            Ok(())
        })
        .unwrap_err();
    assert_eq!(err, LlmError::Cancelled);
    assert_eq!(seen, 2);
}

#[test]
fn a_stream_without_a_done_record_is_incomplete() {
    let server = FakeOllama::start().unwrap();
    let provider = provider(&server, "qwen3.5:4b");
    let err = provider
        .chat_stream(request("qwen3.5:4b", "ABORT"), &mut |_| Ok(()))
        .unwrap_err();
    assert!(matches!(err, LlmError::IncompleteResponse(_)), "{err:?}");
    assert!(err.is_retryable());
}

#[test]
fn malformed_records_fail_with_a_decode_error() {
    let server = FakeOllama::start().unwrap();
    let provider = provider(&server, "qwen3.5:4b");
    let err = provider
        .chat_stream(request("qwen3.5:4b", "BADJSON"), &mut |_| Ok(()))
        .unwrap_err();
    assert!(matches!(err, LlmError::Decode(_)), "{err:?}");
}

#[test]
fn capabilities_come_from_api_show() {
    let server = FakeOllama::start().unwrap();
    let spec = provider(&server, "qwen3.5:4b").show("qwen3.5:4b").unwrap();
    assert_eq!(spec.supports_tools(), Some(true));
    assert_eq!(spec.parameter_size.as_deref(), Some("4.0B"));
    assert!(spec.template.is_some());
}

#[test]
fn connection_failures_are_reported_as_connect_errors() {
    let provider = OllamaProvider::new(&OllamaConfig {
        base_url: "http://127.0.0.1:1".into(),
        model: "qwen3.5:4b".into(),
    })
    .unwrap();
    let err = provider.check_connection().unwrap_err();
    assert!(err.is_retryable());
    match err {
        LlmError::Connect { url, .. } => assert_eq!(url, "http://127.0.0.1:1"),
        other => panic!("expected Connect, got {other:?}"),
    }
}

#[test]
fn health_summarises_a_working_server() {
    let server = FakeOllama::start().unwrap();
    let health = provider(&server, "qwen3.5:4b").health();
    assert!(health.reachable);
    assert!(health.model_present);
    assert_eq!(health.model_count, 2);
    assert_eq!(health.version.as_deref(), Some("0.9.9-fake"));

    let missing = provider(&server, "qwen3.5:32b").health();
    assert!(missing.reachable);
    assert!(
        !missing.model_present,
        "version and model detection are independent"
    );
}

#[test]
fn the_ndjson_driver_round_trips_a_whole_stream() {
    // Drives the shared streaming scaffold with a hand-made transport closure,
    // proving that providers other than Ollama can reuse it.
    let records = [
        b"{\"message\":{\"content\":\"one \"},\"done\":false}\n".to_vec(),
        b"{\"message\":{\"content\":\"two\"},\"done\":false}\n".to_vec(),
        b"{\"done\":true,\"done_reason\":\"stop\"}\n".to_vec(),
    ];
    let mut i = 0;
    let reply = drive_ndjson_stream(
        "test-model",
        || {
            let out: Result<Option<Vec<u8>>, LlmError> = match records.get(i) {
                Some(bytes) => Ok(Some(bytes.clone())),
                None => Ok(None),
            };
            i += 1;
            out
        },
        &mut |_| Ok(()),
    )
    .unwrap();
    assert_eq!(reply.content(), "one two");
    assert_eq!(reply.model, "test-model");
}
