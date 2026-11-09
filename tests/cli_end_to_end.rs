//! End-to-end tests: the real binary, the real HTTP client, a scripted Ollama.
//!
//! These cover the M0 acceptance criteria from the spec:
//!
//! | Test | Criterion |
//! |------|-----------|
//! | `doctor_reports_a_healthy_environment` | Ollama detected, model detected |
//! | `chat_streams_and_keeps_conversation_history` | streaming + in-session history |
//! | `chat_handles_missing_model` | clear model-not-found message |
//! | `chat_handles_unreachable_server` | clear connection error |
//! | `chat_handles_malformed_records` | controlled error on bad JSON |
//! | `chat_reports_interrupted_streams` | incomplete response reported |
//! | `run_prints_only_the_answer` | scriptable one-shot output |
//! | `models_lists_installed_models` | CLI command |
//! | `config_check_validates` | CLI command |
//! | `unknown_flags_exit_with_usage_error` | CLI errors |

use qwen_agent::server::FakeOllama;
use std::io::Write;
use std::process::{Command, Output, Stdio};

/// Environment for a test binary invocation: isolated HOME, its own CWD, no
/// ambient Ollama variables.
fn command(args: &[&str], base_url: &str, cwd: &std::path::Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_qwen-agent"));
    cmd.args(args)
        .current_dir(cwd)
        .env_clear()
        .env("PATH", std::env::var("PATH").unwrap_or_default())
        .env("HOME", cwd)
        .env("QWEN_AGENT_BASE_URL", base_url)
        .env("NO_COLOR", "1")
        .env("TERM", "dumb")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    cmd
}

fn run_with_stdin(mut cmd: Command, input: &str) -> Output {
    let mut child = cmd.spawn().expect("spawn qwen-agent");
    child
        .stdin
        .as_mut()
        .expect("stdin piped")
        .write_all(input.as_bytes())
        .expect("write stdin");
    // Dropping stdin closes it, which ends the REPL.
    drop(child.stdin.take());
    child.wait_with_output().expect("wait for qwen-agent")
}

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("qwen-agent-it-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).to_string()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).to_string()
}

#[test]
fn doctor_reports_a_healthy_environment() {
    let server = FakeOllama::start().unwrap();
    let dir = temp_dir("doctor-ok");
    let out = command(&["doctor", "--verbose"], &server.base_url(), &dir)
        .output()
        .expect("run doctor");
    let text = stdout(&out);
    assert_eq!(
        out.status.code(),
        Some(0),
        "stdout: {text} stderr: {}",
        stderr(&out)
    );
    assert!(text.contains("ollama"), "{text}");
    assert!(text.contains("0.9.9-fake"), "{text}");
    assert!(text.contains("qwen3.5:4b"), "{text}");
    assert!(text.contains("tools supported"), "{text}");
    assert!(text.contains("ready"), "{text}");
}

#[test]
fn doctor_fails_when_the_model_is_missing() {
    let server = FakeOllama::start().unwrap();
    let dir = temp_dir("doctor-model");
    let out = command(
        &["--model", "qwen3.5:32b", "doctor"],
        &server.base_url(),
        &dir,
    )
    .output()
    .expect("run doctor");
    let text = stdout(&out);
    assert_eq!(out.status.code(), Some(1));
    assert!(text.contains("not installed"), "{text}");
    assert!(text.contains("ollama pull qwen3.5:32b"), "{text}");
}

#[test]
fn doctor_fails_when_ollama_is_unreachable() {
    let dir = temp_dir("doctor-down");
    // Port 1 is never listening; connect is refused immediately.
    let out = command(&["doctor"], "http://127.0.0.1:1", &dir)
        .output()
        .expect("run doctor");
    let text = stdout(&out);
    assert_eq!(out.status.code(), Some(1));
    assert!(text.contains("ollama serve"), "{text}");
}

#[test]
fn chat_streams_and_keeps_conversation_history() {
    let server = FakeOllama::start().unwrap();
    let dir = temp_dir("chat-history");
    let out = run_with_stdin(
        command(&["chat"], &server.base_url(), &dir),
        "first question\n/history\n/exit\n",
    );
    let text = stdout(&out);
    assert_eq!(out.status.code(), Some(0), "stderr: {}", stderr(&out));
    // The streamed answer arrived.
    assert!(text.contains("Hello from the fake Ollama"), "{text}");
    assert!(text.contains("first question"), "{text}");
    // /history replay shows both roles.
    assert!(text.contains("you") && text.contains("qwen"), "{text}");
}

#[test]
fn chat_carries_context_between_turns() {
    let server = FakeOllama::start().unwrap();
    let dir = temp_dir("chat-context");
    // ECHO makes the fake reply with the prompt, so we can prove the second
    // request was made at all and that both turns are in history.
    let out = run_with_stdin(
        command(&["chat"], &server.base_url(), &dir),
        "ECHO one\nECHO two\n/history\n/exit\n",
    );
    let text = stdout(&out);
    assert!(text.contains("You said: ECHO one"), "{text}");
    assert!(text.contains("You said: ECHO two"), "{text}");
    assert!(server.requests.load(std::sync::atomic::Ordering::Relaxed) >= 3);
}

#[test]
fn chat_reports_a_missing_model_with_a_pull_hint() {
    let server = FakeOllama::start().unwrap();
    let dir = temp_dir("chat-missing-model");
    let out = run_with_stdin(
        command(
            &["--model", "qwen3.5:32b", "chat"],
            &server.base_url(),
            &dir,
        ),
        "hello\n/exit\n",
    );
    let text = stdout(&out) + &stderr(&out);
    // Preflight refuses to start a conversation with an unusable model.
    assert_eq!(out.status.code(), Some(1), "{text}");
    assert!(
        text.contains("not installed") || text.contains("not found"),
        "{text}"
    );
    assert!(text.contains("ollama pull"), "{text}");
}

#[test]
fn chat_reports_an_unreachable_server_clearly() {
    let dir = temp_dir("chat-down");
    let out = run_with_stdin(command(&["chat"], "http://127.0.0.1:1", &dir), "hello\n");
    let text = stdout(&out) + &stderr(&out);
    assert_eq!(out.status.code(), Some(1));
    assert!(text.contains("cannot reach Ollama"), "{text}");
    assert!(text.contains("ollama serve"), "{text}");
}

#[test]
fn chat_survives_a_malformed_stream_record() {
    let server = FakeOllama::start().unwrap();
    let dir = temp_dir("chat-badjson");
    let out = run_with_stdin(
        command(&["chat"], &server.base_url(), &dir),
        "BADJSON\n/exit\n",
    );
    let text = stdout(&out) + &stderr(&out);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the session must survive: {text}"
    );
    assert!(text.to_lowercase().contains("json"), "{text}");
}

#[test]
fn chat_reports_an_interrupted_stream() {
    let server = FakeOllama::start().unwrap();
    let dir = temp_dir("chat-abort");
    let out = run_with_stdin(
        command(&["chat"], &server.base_url(), &dir),
        "ABORT\n/exit\n",
    );
    let text = stdout(&out) + &stderr(&out);
    assert_eq!(out.status.code(), Some(0), "{text}");
    assert!(
        text.contains("incomplete") || text.contains("stream"),
        "an aborted stream must be reported: {text}"
    );
}

#[test]
fn chat_reports_an_empty_reply() {
    let server = FakeOllama::start().unwrap();
    let dir = temp_dir("chat-empty");
    let out = run_with_stdin(
        command(&["chat"], &server.base_url(), &dir),
        "EMPTY\n/exit\n",
    );
    let text = stdout(&out);
    assert_eq!(out.status.code(), Some(0), "{text}");
    assert!(text.contains("nothing usable"), "{text}");
}

#[test]
fn chat_surfaces_tool_call_requests_without_executing_them() {
    let server = FakeOllama::start().unwrap();
    let dir = temp_dir("chat-tools");
    let out = run_with_stdin(
        command(&["chat"], &server.base_url(), &dir),
        "TOOLS\n/exit\n",
    );
    let text = stdout(&out);
    assert_eq!(out.status.code(), Some(0), "{text}");
    assert!(text.contains("tool call"), "{text}");
    assert!(
        text.contains("M1"),
        "M0 must say tool execution is not implemented: {text}"
    );
}

#[test]
fn run_prints_only_the_answer() {
    let server = FakeOllama::start().unwrap();
    let dir = temp_dir("run-once");
    let out = command(&["run", "ECHO from-a-script"], &server.base_url(), &dir)
        .output()
        .expect("run once");
    let text = stdout(&out);
    assert_eq!(out.status.code(), Some(0), "stderr: {}", stderr(&out));
    assert!(text.contains("You said: ECHO from-a-script"), "{text}");
    assert!(
        !text.contains('\u{256d}'),
        "one-shot output has no banner: {text}"
    );
}

#[test]
fn run_reads_the_prompt_from_stdin() {
    let server = FakeOllama::start().unwrap();
    let dir = temp_dir("run-stdin");
    let out = run_with_stdin(
        command(&["run", "--stdin"], &server.base_url(), &dir),
        "ECHO piped prompt\n",
    );
    let text = stdout(&out);
    assert_eq!(out.status.code(), Some(0), "stderr: {}", stderr(&out));
    assert!(text.contains("You said: ECHO piped prompt"), "{text}");
}

#[test]
fn run_exits_nonzero_when_the_request_fails() {
    let dir = temp_dir("run-fail");
    let out = command(&["run", "FAIL_500"], "http://127.0.0.1:1", &dir)
        .output()
        .expect("run");
    assert_eq!(out.status.code(), Some(1));
}

#[test]
fn models_lists_installed_models_as_text_and_json() {
    let server = FakeOllama::start().unwrap();
    let dir = temp_dir("models");

    let text_out = command(&["models"], &server.base_url(), &dir)
        .output()
        .unwrap();
    let text = stdout(&text_out);
    assert_eq!(
        text_out.status.code(),
        Some(0),
        "stderr: {}",
        stderr(&text_out)
    );
    assert!(text.contains("qwen3.5:4b"), "{text}");
    assert!(text.contains("llama3:8b"), "{text}");
    assert!(text.contains('*'), "the active model is marked: {text}");

    let json_out = command(&["models", "--json"], &server.base_url(), &dir)
        .output()
        .unwrap();
    let json_text = stdout(&json_out);
    let parsed = qwen_agent::json::parse(json_text.trim()).expect("valid JSON");
    let models = parsed.as_array().expect("array");
    assert_eq!(models.len(), 2);
    assert_eq!(models[0].get("name").unwrap().as_str(), Some("qwen3.5:4b"));
    assert_eq!(models[0].get("active").unwrap().as_bool(), Some(true));
}

#[test]
fn config_check_validates_a_config_file() {
    let dir = temp_dir("config-check");
    std::fs::write(
        dir.join("qwen-agent.toml"),
        "[llm]\nmodel = \"qwen3.5:4b\"\n",
    )
    .unwrap();
    let out = command(&["config", "--check"], "http://127.0.0.1:1", &dir)
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", stdout(&out));
    assert!(stdout(&out).contains("valid"));

    // A bad value must fail with the field name.
    std::fs::write(dir.join("qwen-agent.toml"), "[llm]\ntemperature = 9.5\n").unwrap();
    let out = command(&["config", "--check"], "http://127.0.0.1:1", &dir)
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(stdout(&out).contains("llm.temperature"), "{}", stdout(&out));
}

#[test]
fn config_init_writes_a_file_that_parses() {
    let dir = temp_dir("config-init");
    let out = command(&["config", "--init"], "http://127.0.0.1:1", &dir)
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(0));
    let written = std::fs::read_to_string(dir.join("qwen-agent.toml")).expect("file written");
    qwen_agent::config::Config::from_toml_str(&written).expect("generated config must parse");

    // Refuses to clobber an existing file.
    let out = command(&["config", "--init"], "http://127.0.0.1:1", &dir)
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
}

#[test]
fn config_toml_round_trips_through_the_loader() {
    let dir = temp_dir("config-toml");
    let out = command(&["config", "--toml"], "http://127.0.0.1:1", &dir)
        .output()
        .unwrap();
    let text = stdout(&out);
    assert_eq!(out.status.code(), Some(0));
    let parsed = qwen_agent::config::Config::from_toml_str(&text).expect("valid TOML");
    assert_eq!(parsed.llm.base_url, "http://127.0.0.1:1");
}

#[test]
fn command_line_flags_override_the_config_file() {
    let server = FakeOllama::start().unwrap();
    let dir = temp_dir("precedence");
    std::fs::write(
        dir.join("qwen-agent.toml"),
        "[llm]\nmodel = \"llama3:8b\"\nbase_url = \"http://127.0.0.1:1\"\n",
    )
    .unwrap();
    let out = command(
        &["--model", "qwen3.5:4b", "config", "--toml"],
        &server.base_url(),
        &dir,
    )
    .output()
    .unwrap();
    let text = stdout(&out);
    // env beats file, flag beats env's neighbour field: model from the flag.
    assert!(text.contains("model = \"qwen3.5:4b\""), "{text}");
}

#[test]
fn unknown_flags_and_commands_exit_with_usage_errors() {
    let dir = temp_dir("usage");
    let out = command(&["--nope"], "http://127.0.0.1:1", &dir)
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr(&out).contains("unknown option"));
    assert!(stderr(&out).contains("--help"));

    let out = command(&["frobnicate"], "http://127.0.0.1:1", &dir)
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr(&out).contains("unknown command"));
}

#[test]
fn help_and_version_are_available_without_a_server() {
    let dir = temp_dir("help");
    let out = command(&["--help"], "http://127.0.0.1:1", &dir)
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(0));
    let text = stdout(&out);
    for needle in ["chat", "run", "models", "config", "doctor", "--base-url"] {
        assert!(text.contains(needle), "help is missing {needle}");
    }

    let out = command(&["--version"], "http://127.0.0.1:1", &dir)
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(0));
    assert!(stdout(&out).contains("qwen-agent"));
}

#[test]
fn log_level_flag_enables_diagnostics_on_stderr_only() {
    let server = FakeOllama::start().unwrap();
    let dir = temp_dir("logging");
    let out = command(
        &["--log-level", "debug", "run", "ECHO logged"],
        &server.base_url(),
        &dir,
    )
    .output()
    .unwrap();
    assert_eq!(out.status.code(), Some(0));
    // stdout stays clean for piping.
    assert!(!stdout(&out).contains(":: debug"), "{}", stdout(&out));
    assert!(stderr(&out).contains(":: debug"), "{}", stderr(&out));
}
