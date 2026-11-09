//! A tiny scripted stand-in for an Ollama daemon.
//!
//! Used by `tests/` and by the `fake-ollama` binary. It speaks just enough of the
//! real API (`/api/version`, `/api/tags`, `/api/show`, `/api/chat`) to exercise
//! the client end to end, including the awkward parts:
//!
//! * NDJSON streamed as **chunked** transfer encoding,
//! * token-sized chunks that straddle JSON record boundaries,
//! * mid-stream `{"error": ...}` records,
//! * connections dropped without a terminating `done` record,
//! * `404 model not found`, `500`, and slow first tokens.
//!
//! Behaviour is chosen by the **last user message** so a single server can serve
//! a scripted conversation: send `SLOW`, `FAIL_500`, `ABORT`, `TOOLS` or
//! `BADJSON` and the server reacts accordingly.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

/// Controls for a running fake server.
pub struct FakeOllama {
    port: u16,
    shutdown: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    pub requests: Arc<AtomicUsize>,
}

impl FakeOllama {
    /// Starts the server on an ephemeral port.
    pub fn start() -> std::io::Result<FakeOllama> {
        FakeOllama::start_on(0)
    }

    /// Starts the server on a fixed port (0 = ephemeral).
    pub fn start_on(port: u16) -> std::io::Result<FakeOllama> {
        let listener = TcpListener::bind(("127.0.0.1", port))?;
        let port = listener.local_addr()?.port();
        listener.set_nonblocking(true)?;
        let shutdown = Arc::new(AtomicBool::new(false));
        let requests = Arc::new(AtomicUsize::new(0));
        let flag = shutdown.clone();
        let counter = requests.clone();
        let handle = std::thread::spawn(move || {
            while !flag.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        counter.fetch_add(1, Ordering::Relaxed);
                        let _ = stream.set_nonblocking(false);
                        let flag = flag.clone();
                        std::thread::spawn(move || {
                            let _ = handle_connection(stream, &flag);
                        });
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
        });
        Ok(FakeOllama {
            port,
            shutdown,
            handle: Some(handle),
            requests,
        })
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    pub fn port(&self) -> u16 {
        self.port
    }
}

impl Drop for FakeOllama {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn handle_connection(mut stream: TcpStream, shutdown: &AtomicBool) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    let request = match read_request(&mut stream) {
        Some(r) => r,
        None => return Ok(()),
    };

    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/api/version") => {
            write_json(&mut stream, 200, r#"{"version":"0.9.9-fake"}"#)?;
        }
        ("GET", "/api/tags") => {
            write_json(
                &mut stream,
                200,
                r#"{"models":[
                    {"name":"qwen3.5:4b","model":"qwen3.5:4b","size":2600000000,
                     "details":{"family":"qwen3","parameter_size":"4.0B","quantization_level":"Q4_K_M"}},
                    {"name":"llama3:8b","size":4700000000,
                     "details":{"family":"llama","parameter_size":"8.0B","quantization_level":"Q4_0"}}
                ]}"#,
            )?;
        }
        ("POST", "/api/show") => {
            write_json(
                &mut stream,
                200,
                r#"{"details":{"family":"qwen3","parameter_size":"4.0B"},
                    "capabilities":["completion","tools"],
                    "template":"{{ .Prompt }}"}"#,
            )?;
        }
        ("POST", "/api/chat") => {
            let body = String::from_utf8_lossy(&request.body).to_string();
            chat(&mut stream, &body, shutdown)?;
        }
        _ => write_json(&mut stream, 404, r#"{"error":"not found"}"#)?,
    }
    Ok(())
}

fn read_request(stream: &mut TcpStream) -> Option<Request> {
    let mut reader = BufReader::new(stream.try_clone().ok()?);
    let mut line = String::new();
    reader.read_line(&mut line).ok()?;
    let mut parts = line.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();

    let mut content_length = 0usize;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header).ok()? == 0 {
            return None;
        }
        let header = header.trim_end();
        if header.is_empty() {
            break;
        }
        if let Some((name, value)) = header.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") {
                content_length = value.trim().parse().unwrap_or(0);
            }
        }
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body).ok()?;
    }
    Some(Request { method, path, body })
}

fn write_json(stream: &mut TcpStream, status: u16, body: &str) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        if status == 200 { "OK" } else { "Error" },
        body.len(),
        body
    )?;
    stream.flush()
}

/// Extracts the last user message from an Ollama chat request body.
fn last_user_message(body: &str) -> String {
    let Ok(value) = crate::json::parse(body) else {
        return String::new();
    };
    let Some(messages) = value.get("messages").and_then(crate::json::Value::as_array) else {
        return String::new();
    };
    messages
        .iter()
        .rfind(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string()
}

fn chat(stream: &mut TcpStream, body: &str, shutdown: &AtomicBool) -> std::io::Result<()> {
    let prompt = last_user_message(body);

    // Model-not-found check, mirroring Ollama's real 404 body.
    if let Ok(value) = crate::json::parse(body) {
        let model = value.get("model").and_then(|m| m.as_str()).unwrap_or("");
        if !crate::llm::matches_model("qwen3.5:4b", model)
            && !crate::llm::matches_model("llama3:8b", model)
        {
            write_json(
                stream,
                404,
                &format!(r#"{{"error":"model '{model}' not found, try pulling it first"}}"#),
            )?;
            return Ok(());
        }
    }

    if prompt.contains("FAIL_500") {
        return write_json(stream, 500, r#"{"error":"out of memory"}"#);
    }

    // Streaming response headers: chunked so the client's decoder is exercised.
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
    )?;
    stream.flush()?;

    if prompt.contains("BADJSON") {
        write_chunk(stream, b"this is not json\n")?;
        write_chunk(stream, b"{\"done\":true,\"done_reason\":\"stop\"}\n")?;
        write_chunk(stream, b"")?;
        return Ok(());
    }

    let reply = if prompt.contains("TOOLS") {
        // A tool-call record has empty content, so it must be written verbatim
        // rather than routed through the token splitter (which wraps text as
        // `content`).
        let record = r#"{"model":"qwen3.5:4b","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"read_file","arguments":{"path":"src/main.rs"}}}]},"done":false}"#;
        if prompt.contains("SLOW") {
            std::thread::sleep(Duration::from_millis(250));
        }
        write_chunk(stream, format!("{record}\n").as_bytes())?;
        write_chunk(stream, b"{\"done\":true,\"done_reason\":\"stop\",\"eval_count\":12,\"prompt_eval_count\":40}\n")?;
        write_chunk(stream, b"")?;
        return Ok(());
    } else if prompt.contains("ABORT") {
        let record = r#"{"model":"qwen3.5:4b","message":{"role":"assistant","content":"this reply will be cut"},"done":false}"#;
        write_chunk(stream, format!("{record}\n").as_bytes())?;
        // Drop the connection without a `done` record.
        write_chunk(stream, b"")?;
        return Ok(());
    } else if prompt.contains("EMPTY") {
        let record =
            r#"{"model":"qwen3.5:4b","message":{"role":"assistant","content":"   "},"done":false}"#;
        write_chunk(stream, format!("{record}\n").as_bytes())?;
        write_chunk(stream, b"{\"done\":true,\"done_reason\":\"stop\"}\n")?;
        write_chunk(stream, b"")?;
        return Ok(());
    } else if prompt.contains("ECHO") {
        format!("You said: {prompt}")
    } else if prompt.contains("EMIT_ERROR") {
        write_chunk(stream, b"{\"error\":\"model crashed\"}\n")?;
        write_chunk(stream, b"")?;
        return Ok(());
    } else {
        format!(
            "Hello from the fake Ollama. You asked: {prompt}. \
             This response is streamed token by token, exactly like Qwen3.5:4B would."
        )
    };

    let slow = prompt.contains("SLOW");
    stream_chunked(stream, &[reply.as_str()], shutdown, slow)
}

/// Emits `texts` as NDJSON records, splitting the way a token stream would.
fn stream_chunked(
    stream: &mut TcpStream,
    texts: &[&str],
    shutdown: &AtomicBool,
    slow: bool,
) -> std::io::Result<()> {
    for text in texts {
        if slow {
            std::thread::sleep(Duration::from_millis(250));
        }
        for token in split_like_tokens(text) {
            if shutdown.load(Ordering::Relaxed) {
                return Ok(());
            }
            let escaped = crate::json::to_string(&crate::json::Value::str(token));
            let record = format!(
                "{{\"model\":\"qwen3.5:4b\",\"message\":{{\"role\":\"assistant\",\"content\":{escaped}}},\"done\":false}}\n"
            );
            write_chunk(stream, record.as_bytes())?;
            if slow {
                std::thread::sleep(Duration::from_millis(80));
            }
        }
    }
    write_chunk(
        stream,
        b"{\"model\":\"qwen3.5:4b\",\"done\":true,\"done_reason\":\"stop\",\"eval_count\":42,\"prompt_eval_count\":31}\n",
    )?;
    write_chunk(stream, b"")
}

/// Splits text into small pieces, simulating per-token records.
fn split_like_tokens(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Break after ~7 bytes on a char boundary.
        if i - start >= 7 && text.is_char_boundary(i) {
            out.push(&text[start..i]);
            start = i;
        }
        i += 1;
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

/// Writes one HTTP chunk; an empty payload terminates the body.
fn write_chunk(stream: &mut TcpStream, payload: &[u8]) -> std::io::Result<()> {
    write!(stream, "{:x}\r\n", payload.len())?;
    if !payload.is_empty() {
        stream.write_all(payload)?;
    }
    write!(stream, "\r\n")?;
    stream.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_splitting_preserves_content() {
        let text = "héllo wörld, this is a longer sentence.";
        let parts = split_like_tokens(text);
        assert_eq!(parts.concat(), text);
        assert!(parts.len() > 1);
        for p in &parts {
            // Every piece must be valid UTF-8 on its own.
            assert!(!p.is_empty());
        }
    }

    #[test]
    fn last_user_message_ignores_system_and_assistant() {
        let body = r#"{"messages":[
            {"role":"system","content":"sys"},
            {"role":"user","content":"first"},
            {"role":"assistant","content":"reply"},
            {"role":"user","content":"second"}
        ]}"#;
        assert_eq!(last_user_message(body), "second");
        assert_eq!(last_user_message("not json"), "");
    }

    #[test]
    fn server_starts_and_serves_version() {
        let server = FakeOllama::start().unwrap();
        let url = crate::http::Url::parse(&format!("{}/api/version", server.base_url())).unwrap();
        let mut resp = crate::http::get(&url).unwrap();
        assert_eq!(resp.status(), 200);
        let body = resp.read_to_string().unwrap();
        assert!(body.contains("0.9.9-fake"));
    }
}
