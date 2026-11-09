//! `fake-ollama` — run the scripted stand-in server from the command line.
//!
//! ```text
//! $ cargo run --features testing-server --bin fake-ollama -- --port 11434
//! $ qwen-agent --base-url http://127.0.0.1:11434 chat
//! ```
//!
//! Useful for demos and for developing the agent without a GPU. Behaviour is
//! controlled by the last user message: `SLOW`, `ABORT`, `FAIL_500`, `TOOLS`,
//! `BADJSON`, `EMPTY`, `EMIT_ERROR`, `ECHO`.

use qwen_agent::server::FakeOllama;

fn main() {
    let mut port = 11434;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--port" | "-p" => {
                i += 1;
                match args.get(i).and_then(|v| v.parse::<u16>().ok()) {
                    Some(p) => port = p,
                    None => {
                        eprintln!("error: --port needs a number");
                        std::process::exit(2);
                    }
                }
            }
            "--help" | "-h" => {
                println!("usage: fake-ollama [--port N]  (default 11434)");
                println!();
                println!("Serves /api/version, /api/tags, /api/show and /api/chat.");
                println!("Last user message controls the reply: SLOW, ABORT, FAIL_500,");
                println!("TOOLS, BADJSON, EMPTY, EMIT_ERROR, ECHO.");
                return;
            }
            other => {
                eprintln!("error: unknown argument `{other}`");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    match FakeOllama::start_on(port) {
        Ok(server) => {
            eprintln!("fake-ollama listening on {}", server.base_url());
            eprintln!("press Ctrl-C to stop");
            // Park forever; the listener thread owns the work.
            loop {
                std::thread::sleep(std::time::Duration::from_secs(60));
            }
        }
        Err(err) => {
            eprintln!("error: cannot bind port {port}: {err}");
            std::process::exit(1);
        }
    }
}
