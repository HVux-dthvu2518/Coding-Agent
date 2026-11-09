# qwen-agent

A local-first CLI agent for **[Qwen3.5:4B]** served by **[Ollama]**, written in Rust.

This repository implements **M0 — Foundation** of the architecture proposal: a
working local AI CLI that talks to Ollama, streams tokens, keeps a bounded
in-session conversation, validates its configuration, diagnoses its environment,
and fails with advice instead of panics. Tool calling (M1), the autonomous loop
(M2) and the capability milestones (M3–M5) plug into typed seams that are
already in place — see [Roadmap](#roadmap).

```text
$ qwen-agent chat

╭──────────────────────────────────────────────────────────╮
│                          OLLAMA                          │
│ Provider : ollama                                        │
│ Model    : Qwen3.5:4B                                    │
│ Endpoint : http://localhost:11434                        │
│ Streaming: on   Temp: 0.7                                │
│ Status   : Connected                                     │
╰──────────────────────────────────────────────────────────╯

You > Explain Rust ownership in two sentences.
Qwen > Ownership is a set of rules the compiler checks: every value has exactly
       one owner, and when that owner goes out of scope the value is dropped…
24.7 tok/s · 42 reply tokens · 31 prompt tokens · context 84/3000 est.
```

---

## Status

| Area | State |
|------|-------|
| Ollama provider (`/api/version`, `/api/tags`, `/api/show`, `/api/chat`) | done |
| Streaming NDJSON, robust to arbitrary chunk boundaries | done |
| Conversation context with turn-aware budget eviction | done |
| Configuration: TOML + env + flags, validated, provenance shown | done |
| CLI: `chat`, `run`, `models`, `config`, `doctor` | done |
| REPL commands (`/help`, `/status`, `/stats`, `/history`, `/save`, …) | done |
| Deterministic reflection (empty/truncated/scaffolding/repetition) | done |
| Typed error taxonomy with actionable hints | done |
| Tool calling (M1) | seam only: tool calls are parsed, surfaced, **not executed** |
| Planning + autonomous loop (M2) | seam only: state machine + plan types |
| Filesystem/shell/git tools (M3), SQLite memory (M4), browser (M5) | not started |

241 tests (196 unit + 45 integration/doc) pass with `cargo test`; `cargo clippy
--all-targets` is clean; `cargo fmt --check` is clean.

---

## Dependency strategy (read this first)

The environment this was built in can reach `github.com`, `npmjs.org` and
`pypi.org`, but **not `crates.io` or `static.crates.io`**, so `cargo` cannot
resolve `tokio`, `serde`, `reqwest`, `clap` or any other registry crate. Rather
than ship code nobody could build there, **M0 has zero dependencies** and is
implemented on `std` alone:

| Proposal said | This build | Why it is equivalent |
|---------------|-----------|----------------------|
| `reqwest` + `tokio` | `src/http.rs`: blocking HTTP/1.1 client with connect/read timeouts, chunked decoding | Only `POST /api/chat` against `localhost` is needed; the socket read loop is the same |
| `serde` + `serde_json` | `src/json.rs`: parser, serializer, typed `Cursor` | Same wire types, precise byte-offset errors |
| `toml` | `src/toml.rs`: subset reader for the config schema | The config format is fixed; unsupported syntax is a named error, not silence |
| `clap` | `src/cli/args.rs`: hand-written parser with clap-compatible flags | Same interface (`--flag=value`, `-h`, `-V`, `--`), fully unit-tested |
| `tracing` | `src/logging.rs`: levelled logger to stderr | Same "logs never touch stdout" invariant |
| `anyhow`/`thiserror` | `src/error.rs`: typed enums + `std::error::Error` | Same split: typed subsystems, one app-level error |

**Migration path.** Nothing outside `http.rs`, `json.rs`, `toml.rs` and
`logging.rs` knows how those services are implemented. When a registry is
available:

1. add the dependencies to `Cargo.toml`;
2. reimplement the four leaf modules (`http` → `reqwest`, `json` → `serde_json`,
   `toml` → `toml`, `logging` → `tracing`), keeping their public functions;
3. make `LlmProvider` methods `async` (they are `fn` today; see
   `src/llm/provider.rs` for why this is a contained change);
4. delete the shims. No call site in `agent/`, `cli/` or `app.rs` changes.

The design deliberately keeps the **agent loop** independent of the HTTP client,
so this is a refactor of the edge, not of the core.

---

## Build and run

### With a normal Rust toolchain and crates.io access

```bash
cargo build --release
./target/release/qwen-agent --help
```

Requires Rust 1.85+ (edition 2024).

### Offline toolchain (as used here)

The bundled toolchain (`rustc`/`cargo`/`clippy`/`rustfmt`, no `rustdoc`)
builds this crate with no registry access:

```bash
export PATH=/path/to/rust-1.97.0-x86_64-unknown-linux-gnu/bin:$PATH
cargo build --release   # no network, no vendored crates
cargo test
```

Because `rustdoc` is absent, `Cargo.toml` sets `doctest = false`; the examples
from this README and the crate docs are executed by `tests/api_usage.rs`
instead, so they cannot rot.

### Prerequisites for real use

```bash
ollama serve            # in one terminal
ollama pull Qwen3.5:4B  # ~2.6 GB
qwen-agent doctor       # confirm everything lines up
qwen-agent chat
```

### Try it without Ollama

A scripted stand-in server ships with the crate:

```bash
cargo run --features testing-server --bin fake-ollama -- --port 11434 &
qwen-agent --base-url http://127.0.0.1:11434 chat
```

The fake replies to `/api/version`, `/api/tags`, `/api/show` and `/api/chat`
(including chunked NDJSON, mid-stream errors and aborted streams). Its last
user message selects the behaviour:

| Send this | The fake does |
|-----------|---------------|
| `ECHO <text>` | replies `You said: <text>` |
| `SLOW` | trickles tokens with 80 ms gaps (testing Ctrl-C) |
| `TOOLS` | returns a `read_file` tool call |
| `ABORT` | drops the connection without a `done` record |
| `FAIL_500` | replies `500 {"error":"out of memory"}` |
| `EMPTY` | returns whitespace only |
| `EMIT_ERROR` | sends `{"error":"model crashed"}` mid-stream |
| `BADJSON` | sends a malformed record |

---

## CLI reference

```text
qwen-agent [GLOBAL OPTIONS] [COMMAND]

COMMANDS
  chat                Interactive conversation (default)
  run <prompt>        Answer one prompt and exit
  models              List installed Ollama models
  config              Show the effective configuration
  doctor              Check Ollama, the model and the config

GLOBAL OPTIONS
  -c, --config <path>     Configuration file to use
      --base-url <url>    Ollama endpoint (default http://localhost:11434)
      --model <name>      Model name (default qwen3.5:4b)
      --temperature <n>   Sampling temperature, 0.0–2.0
      --no-stream         Disable streaming output
      --log-level <lvl>   error|warn|info|debug|trace
      --color <when>      auto|always|never
  -h, --help              Show help
  -V, --version           Show the version

COMMAND OPTIONS
  run --stdin             Read the prompt from stdin
  models --json           Machine-readable output
  config --toml           Print the effective config as TOML
  config --check          Validate the configuration and exit
  config --init [path]    Write a default configuration file
  doctor --verbose        Probe model capabilities via /api/show
```

Exit codes: `0` success, `1` failure (bad config, unreachable server, missing
model, failed request), `2` usage error.

`run` writes the answer to stdout and diagnostics to stderr, so it composes:

```bash
# piped input replaces the prompt…
cat src/http.rs | qwen-agent run > notes.md
# …or is appended to it as context
git diff | qwen-agent run "review this change for bugs"
# …and a prompt alone needs no stdin at all
qwen-agent run "explain Rust's ? operator" | pbcopy
```

Because stdin is only read when it is **not** a terminal, `qwen-agent run
"explain ownership"` run by hand never waits for EOF.

### REPL commands

| Command | Behaviour |
|---------|-----------|
| `/help` | list commands |
| `/exit`, `/quit`, Ctrl-D | leave |
| `/clear` | forget the conversation, keep the system prompt |
| `/model` | show the active model |
| `/models` | list installed models, marking the active one |
| `/status` | server reachability, model presence, context usage |
| `/stats` | last-turn timing, tokens, tokens/second, context budget |
| `/history` | print the conversation so far |
| `/system [text]` | show or replace the system prompt |
| `/config` | print the effective configuration |
| `/save [path]` | write the conversation to a Markdown file |
| `/retry` | re-send the previous message |

Typos get a suggestion (`/hlep` → `did you mean /help?`).

---

## Configuration

Precedence, highest first:

1. command-line flags
2. environment: `QWEN_AGENT_CONFIG`, `QWEN_AGENT_BASE_URL`, `QWEN_AGENT_MODEL`,
   `QWEN_AGENT_TEMPERATURE`, `QWEN_AGENT_LOG`, plus `OLLAMA_HOST` (honoured when
   `QWEN_AGENT_BASE_URL` is unset) and `NO_COLOR`
3. the first file found walking up from the working directory:
   `qwen-agent.toml`, then `config/default.toml`, then `.qwen-agent.toml`,
   then `$XDG_CONFIG_HOME`-style `~/.config/qwen-agent/config.toml`
4. built-in defaults

```toml
[agent]
name = "Qwen Agent"
max_iterations = 10
keep_alive = "10m"           # keeps the 4B model resident between turns
system_prompt = "You are Qwen, a careful local AI assistant…"

[llm]
provider = "ollama"          # only ollama is implemented in this build
base_url = "http://localhost:11434"
model = "qwen3.5:4b"         # case-insensitive; a missing tag means ":latest"
temperature = 0.7            # 0.0 – 2.0
stream = true
num_ctx = 0                  # 0 = server default; also sets the context budget
num_predict = -1             # -1 = no explicit cap

[logging]
level = "warn"               # logs always go to stderr

[workspace]
root = "."
```

Every value is validated at load time with the field name, and a validation
failure names the file it came from:

```text
$ qwen-agent config --check
invalid: /home/me/qwen-agent.toml: invalid value for `llm.temperature`:
         9.5 is outside 0.0..=2.0 (0 is deterministic, 2 is very random)
```

Unknown keys or sections produce warnings rather than silent no-ops.

---

## Testing

```bash
cargo test                      # unit + integration + executable docs
cargo clippy --all-targets      # clean
cargo fmt --check               # clean
```

| Suite | Covers |
|-------|--------|
| `src/**` unit tests (196) | JSON/TOML/HTTP parsers, NDJSON framing, context eviction, state machine, reflection, argument parsing, REPL transcripts, doctor |
| `tests/ollama_provider.rs` (16) | the provider against the scripted server: streaming, tool-call parsing, error classification, cancellation, incomplete streams |
| `tests/cli_end_to_end.rs` (22) | the real binary: every command, every M0 acceptance criterion, exit codes, stdout/stderr separation |
| `tests/api_usage.rs` (7) | the README examples, executed |

The M0 acceptance criteria from the proposal map one-to-one onto
`tests/cli_end_to_end.rs`, including the awkward ones — malformed JSON,
interrupted streams and empty replies.

---

## Architecture in one screen

```text
                 ┌──────────────┐
   stdin ───────►│ cli::repl    │──► stdout (answers, prompts)
                 │ Session      │──► stderr (errors, logs, notices)
                 └──────┬───────┘
                        │ Event stream
                 ┌──────▼───────┐        ┌───────────────────┐
                 │ agent        │───────►│ llm::LlmProvider  │
                 │ Orchestrator │◄───────│ (trait)           │
                 │ Context      │ chunks │  └ OllamaProvider │
                 │ StateMachine │        └─────────┬─────────┘
                 │ planner /    │                  │
                 │ reflection   │        ┌─────────▼─────────┐
                 └──────────────┘        │ http (chunked)    │
                                         │ json / NDJSON     │
                                         └───────────────────┘
```

Invariants:

* `llm` never depends on `agent` or `cli`; the provider is swappable.
* `agent` never writes to a terminal — it emits `Event`s.
* model output → stdout; logs and diagnostics → stderr.
* a failed turn is rolled back out of the conversation, so `/retry` cannot
  duplicate the question.
* the LLM proposes nothing that Rust does not validate; in M0 the only action
  is "talk".

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design: module
map, state machine, streaming algorithm, error taxonomy, context budgeting, and
the security posture M1 must inherit.

---

## Roadmap

| Milestone | What lands | Where it plugs in |
|-----------|-----------|-------------------|
| **M0** (this repo) | Ollama + streaming + history + CLI + doctor | — |
| M1 — Tools | tool registry, JSON-schema validation, execution, result messages | `agent/workers.rs` allow-lists, `llm::request::ToolCall` (already parsed), `Event::ToolCallsRequested` |
| M2 — Intelligence | planner, orchestrator loop, iteration limits, retries, completion checks | `agent/planner.rs` (`Plan`, `should_plan`), `agent/state.rs` (`Planning`/`Executing`/`Observing`), `config.agent.max_iterations` |
| M3 — Computer tools | filesystem, shell, git, approvals | `agent/workers.rs` (`may_mutate`), a `security/` module |
| M4 — Memory | SQLite conversation and task history | `agent/context.rs` (a repository trait slots under it) |
| M5 — Advanced | browser, research, parallel workers | `LlmProvider` + worker profiles |

Deliberate non-goals for M0, per the proposal: no planner in the hot path, no
multi-agent runtime, no vector store, no browser engine.

---

## Known limitations

* **HTTP/1.1 only, plaintext only.** Ollama is expected on `localhost`; TLS,
  proxies and redirects are not implemented (they arrive with `reqwest`).
* **Single conversation.** No persistence yet — `/save` writes Markdown, but
  history does not survive a restart (that is M4).
* **Token counts are estimates.** The context budget uses a ~4-chars-per-token
  heuristic with headroom; the server's real counts are shown in `/stats`.
* **Tool calls are surfaced, not executed.** A model that asks for a tool gets a
  notice, and M0 records the request so M1 can act on it.
* **No Ctrl-C handling yet.** Cancellation is plumbed end to end (`ChunkSink`
  returns `Err(())`), but the REPL does not yet install a signal handler; Ctrl-C
  terminates the process, which is a clean exit but not yet a "stop this reply"
  gesture.
* `Qwen3.5:4B` is treated as a model name string; M0 does not verify that the
  installed model is genuinely Qwen or that its template emits tool calls — run
  `qwen-agent doctor --verbose` to see what `/api/show` reports.

---

## License

MIT.

[Qwen3.5:4B]: https://ollama.com/library/qwen3
[Ollama]: https://ollama.com
