//! `qwen-agent` — a local-first CLI agent for Qwen3.5:4B via Ollama.
//!
//! M0 scope: talk to Ollama, stream tokens, keep a bounded conversation,
//! validate configuration and diagnose the environment. Tool calling (M1) and
//! the autonomous loop (M2) plug into the seams left in `agent/`.
//!
//! This file is intentionally tiny: everything of substance lives in the library
//! target (`src/lib.rs`), so it can be unit- and integration-tested.

use qwen_agent::{app, cli};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match cli::args::parse_from(args) {
        cli::args::ParseOutcome::Run(cli) => app::run(*cli),
        cli::args::ParseOutcome::Help(text) => {
            print!("{text}");
            app::exit::OK
        }
        cli::args::ParseOutcome::Version(text) => {
            println!("{text}");
            app::exit::OK
        }
        cli::args::ParseOutcome::Error { message, usage } => {
            eprintln!("error: {message}");
            if usage {
                eprintln!("\nRun `qwen-agent --help` for usage.");
            }
            app::exit::USAGE
        }
    };
    std::process::exit(code);
}
