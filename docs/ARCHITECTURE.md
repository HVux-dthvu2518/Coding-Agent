# qwen-agent — architecture

This document describes what M0 implements, why the seams are where they are,
and what the later milestones must inherit. It follows the structure of the
original proposal, marking each decision as **implemented**, **seam** (types and
tests exist, behaviour is deferred) or **deferred**.

---

## 1. Objectives and principles

| Principle | How this build honours it |
|-----------|---------------------------|
| Rust-first | Core orchestration, HTTP, JSON, TOML and CLI are Rust; no external runtime |
| Local-first | Only `localhost:11434` is contacted; no telemetry, no cloud fallback |
| Modular | Six subsystems with one-way dependencies (see §2) |
| Async-shaped | Blocking I/O behind a channel-shaped `ChunkSink`, ready to become `async` |
| Type-safe | Every wire type is a struct/enum; JSON is only touched at the edge |
| Observable | `tracing`-style levelled logs to stderr, per-turn timing, `/stats` |
| Permission-aware | No model-suggested action is executed in M0; M1 inherits the gate |
| Testable | 241 tests; the REPL, provider and CLI are driven without a terminal |

Deliberate non-goals for M0: no planner in the hot path, no LangChain-style
frameworks, no unrestricted tool execution.

---

## 2. Module map and dependency direction

```text
main.rs ──► cli::args ──► app ──► cli::repl ──► agent::orchestrator ──► llm::provider
             │             │         │                    │                    │
             │             │         │                    ▼                    ▼
             │             │         │             agent::context         llm::ollama
             │             │         │             agent::state               │
             │             │         │             agent::reflection          ▼
             │             │         ▼                                       http
             │             │   cli::ui, cli::commands, cli::doctor          json / toml
             │             ▼
             │        config, logging, error
             ▼
        error, config
```

Rules enforced by review and by the module graph:

1. `llm` knows nothing about `agent` or `cli`.
2. `agent` knows nothing about terminals: it emits `Event`s.
3. `cli` owns every byte written to a terminal.
4. `app` is the only composition root; it is the only module that names
   `OllamaProvider` and `Session` together.
5. `http`, `json`, `toml` and `logging` are leaves: they can be swapped for
   `reqwest`, `serde_json`, `toml` and `tracing` without touching anything else.

### File-by-file

| File | Responsibility | Notes |
|------|----------------|-------|
| `main.rs` | argv → `parse_from` → `app::run` → exit code | 30 lines; nothing worth testing |
| `lib.rs` | module declarations, crate docs | exists so integration tests can import the crate |
| `app.rs` | command dispatch, composition, config fallbacks | every branch returns an exit code |
| `config.rs` | TOML + env + flag resolution, validation, provenance | `ConfigError::InvalidInFile` names the file |
| `config_template.rs` | the annotated `config --init` file | validated by a unit test so it cannot drift |
| `error.rs` | `ConfigError`, `LlmError`, `HttpError`, `JsonError`, `TomlError`, `AppError` | each `LlmError` carries a user-facing `hint()` |
| `http.rs` | HTTP/1.1 client: connect/read timeouts, `Content-Length`/chunked/until-EOF | body framing is a state machine, not a buffered read |
| `json.rs` | parser, serializer, `Value`, typed `Cursor` | byte-offset errors; no schema validation |
| `toml.rs` | subset reader for the configuration schema | unsupported syntax errors with the line number |
| `logging.rs` | levelled logger to stderr, wire-level request tracing | `debug` logs bodies, truncated on a char boundary |
| `llm/request.rs` | `Message`, `Role`, `ToolCall`, `Options`, `ChatRequest` | tool-call fields exist now so M1 adds behaviour, not structure |
| `llm/response.rs` | `NdjsonBuffer`, `ChatChunk`, `ChatResponse`, `Usage` | the streaming framer |
| `llm/provider.rs` | `LlmProvider` trait, `ModelInfo`, `Health`, `StreamAssembler`, `drive_ndjson_stream` | shared streaming scaffold for future providers |
| `llm/ollama.rs` | Ollama endpoints, error classification | maps 404+body → `ModelNotFound` |
| `agent/context.rs` | bounded conversation, turn-aware eviction | never separates a tool call from its result |
| `agent/state.rs` | the state machine | M2 states exist and are gated |
| `agent/planner.rs` | `Plan`, `PlanStep`, `should_plan` | deterministic heuristic only |
| `agent/reflection.rs` | post-hoc checks on a reply | empty / truncated / scaffolding / repetition |
| `agent/workers.rs` | worker profiles (allow-lists, mutation policy) | M1/M5 policy lives here |
| `agent/orchestrator.rs` | the loop: send, stream, reflect, record, roll back | emits `Event`s |
| `cli/args.rs` | hand-written parser | clap-compatible surface |
| `cli/commands.rs` | slash-command parsing, typo suggestions | Levenshtein distance ≤ 2 |
| `cli/repl.rs` | `Session`, `InputSource`, one-shot vs interactive rendering | transcript is assertable |
| `cli/ui.rs` | ANSI styling, markdown-lite, wrapping, boxes, width detection | no colour when not a TTY |
| `cli/doctor.rs` | independent environment checks with hints | failures accumulate, not short-circuit |
| `server.rs` + `bin/fake_ollama.rs` | scripted Ollama stand-in | powers integration tests and demos |

---

## 3. The request lifecycle

```text
Session::handle_line
  ├─ commands::parse ──► Empty | Message | Command
  ├─ Command ─────────► handled in place (/status, /models, …)
  └─ Message ─────────► Session::submit
                          │
                          ├─ Orchestrator::send
                          │    ├─ state.begin_request()        (Idle → Thinking)
                          │    ├─ context.push_user(prompt)
                          │    ├─ planner::should_plan()       (debug note only)
                          │    ├─ ChatRequest::new(... keep_alive, options)
                          │    ├─ provider.chat_stream(request, sink)
                          │    │     └─ sink → Event::Token → stdout
                          │    ├─ reflection::assess(response)
                          │    └─ finish_ok / finish_interrupted / finish_err
                          │
                          └─ render notices, stats footer
```

### Why an event stream and not callbacks into the UI

`Orchestrator` is generic over the provider and takes `FnMut(Event) -> Result<(),
()>`. That gives three properties for free:

* **cancellation** — returning `Err(())` from the callback aborts generation;
* **testability** — tests collect events into a `Vec` and assert on order;
* **UI independence** — the REPL, a future TUI, or an HTTP server render the same
  events differently.

### Failure handling

| Situation | Context effect | User sees |
|-----------|----------------|-----------|
| Connection refused / HTTP error / bad JSON | user turn rolled back | `error:` + hint on stderr, session stays alive |
| Empty reply | user turn rolled back | `note: nothing usable was returned` |
| Cancelled mid-stream | user turn rolled back | `[cancelled]` (terminal); silent in `run` |
| Truncated (`done_reason = length`) | reply kept | `note: response was cut off …` |
| Scaffolding or repetition detected | reply kept | `note: …` |

Rolling the user turn back is what makes `/retry` safe: the question is never
present twice in the context.

---

## 4. Streaming: the part that usually breaks

`POST /api/chat` with `"stream": true` returns newline-delimited JSON. Two
independent hazards exist, and both are covered:

### 4.1 HTTP chunks ≠ JSON records

`NdjsonBuffer` accumulates bytes and only emits complete lines:

```rust
let mut nd = NdjsonBuffer::new();
for chunk in socket_reads {
    for record in nd.push(&chunk) { … }   // 0..n complete records
}
nd.finish();                               // flush a final unterminated line
```

Unit tests split a two-record payload at every possible boundary and assert that
records appear exactly once, in order.

### 4.2 Parsing the head consumes body bytes

The status line and headers are read through a `BufReader`, which legitimately
reads *past* the blank line into the body. Dropping that buffer silently loses
the first bytes of the stream — the classic "first record is truncated" bug, and
one that no amount of NDJSON buffering can repair.

`http::Prefixed` replays the leftover bytes before continuing on the socket:

```rust
let leftover = reader.buffer().to_vec();
let transport = Prefixed { prefix: leftover, inner: stream };
```

This was found by an integration test against the fake server, not by reading
the code; `prefixed_replays_buffered_body_bytes` now guards it.

### 4.3 Termination

A stream is only successful if a record with `"done": true` arrives.
`StreamAssembler::finish` returns `LlmError::IncompleteResponse` otherwise, so a
dropped connection can never be mistaken for a complete answer. `body.close` in
Ollama is the common cause and the hint tells the user the reply above is
partial.

### 4.4 Body framing

All three framings real servers use are decoded: `Content-Length`, `chunked`
(with extension parameters, trailers, and chunks larger than the caller's
buffer), and read-until-EOF.

---

## 5. Provider abstraction

```rust
pub trait LlmProvider: Send + Sync {
    fn name(&self) -> &'static str;
    fn model(&self) -> &str;
    fn check_connection(&self) -> Result<String, LlmError>;
    fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError>;
    fn model_exists(&self) -> Result<bool, LlmError> { … }
    fn chat(&self, request: ChatRequest) -> Result<ChatResponse, LlmError>;
    fn chat_stream(&self, request: ChatRequest, sink: &mut ChunkSink<'_>)
        -> Result<ChatResponse, LlmError>;
    fn health(&self) -> Health { … }
}
```

The proposal sketched an `async fn` trait. It is `fn` here for one reason: the
crate has no async runtime, and hand-rolling `Future`s for a socket read loop
would add risk without adding capability. The *shape* is preserved —
`chat_stream` takes a push-based sink and can be cancelled from the sink — so
making it `async` (or spawning it on a Tokio task and pushing into an `mpsc`
channel) is a change confined to `llm/` and one line in the orchestrator.

`drive_ndjson_stream` is the shared scaffold: a new provider supplies a
`FnMut() -> Result<Option<Vec<u8>>, LlmError>` reader and inherits framing,
decoding, cancellation and assembly. `tests/api_usage.rs` implements a complete
second provider in ~40 lines to prove it.

Adding a provider therefore means:

1. implement `LlmProvider`;
2. construct it in `app.rs`;
3. nothing else.

---

## 6. Context management for a 4B model

`agent::context::Context` is not a `Vec<Message>`:

* **system prompt first, always** — some Ollama templates drop a system message
  that arrives mid-conversation;
* **budget in estimated tokens** — ~4 chars/token with a floor of one token per
  word, because agents spend context on dense JSON and code, not prose;
* **turn-aware eviction** — a turn is a user message plus everything up to the
  next user message, so a tool call is never separated from its result (which
  providers reject);
* **no half-turn at the head** — after eviction the oldest surviving message is
  a user message;
* **empty assistant turns are dropped** — residue of a cancelled stream confuses
  small models;
* **near-limit warning** — `/status` and `/stats` show usage against the budget so
  the user sees memory pressure before answers degrade.

Budget defaults to 3000 estimated tokens, or half of `llm.num_ctx` when set.

---

## 7. State machine

```text
                 ┌──────────── complete ─────────────┐
                 ▼                                   │
   Idle ──begin──► Thinking ──first token──► Responding ──finish──► Idle
     ▲                │                          │
     │                └──── error ──► Failed ─────┘
     │                                 │
     └──────────── begin_request ──────┘     (a failure never locks the user out)
```

M2 states — `Planning`, `Executing`, `Observing`, `Completed`, `Blocked` — are
modelled and transition-checked now. Rejected transitions return a
`TransitionError` value instead of panicking, so an M2 bug is a test failure
rather than a runtime surprise.

Two behaviours are deliberately tested because they are easy to get wrong:

* a second concurrent request is refused and is **not counted**;
* a failed request does not require a manual reset: asking again is legal.

---

## 8. Reflection: deterministic before clever

M0 reflection is free and always correct:

| Check | Result | Kept? |
|-------|--------|-------|
| empty / whitespace-only | `Empty` | no, turn rolled back |
| `done_reason = length` | `Truncated` | yes, with a notice |
| chat-template tokens (`<\|im_start\|>`, `<tool_call>`) | `Scaffolding` | yes, with a notice |
| role prefixes at line start (`assistant:`) | `Scaffolding` | yes, with a notice |
| same line repeated 4+ times | `Repetition` | yes, with a notice |

The LLM-driven "did this answer the objective?" check is an M2 concern and costs
a round trip, which a 4B model on a laptop should not pay for a greeting.

False positives were treated as bugs: prose such as *"the line `user:` marks the
speaker"* must not be flagged, and a test enforces it.

---

## 9. Error taxonomy

```text
AppError
 ├─ Config(ConfigError)   Missing | Unreadable | Parse | Invalid | InvalidInFile
 ├─ Llm(LlmError)         Connect | Protocol | Http{status,body} | Decode
 │                        ModelNotFound | IncompleteResponse | Cancelled | Io
 ├─ Io(std::io::Error)
 ├─ Usage(String)
 └─ Other(String)
```

Two rules:

1. **messages carry the fix** — `LlmError::hint()` returns the sentence shown
   under the error (`is Ollama running? Start it with ollama serve`).
2. **`is_retryable()`** distinguishes transient failures (connect, I/O, dropped
   stream) from permanent ones (missing model, malformed response), which is the
   hook M2's retry policy needs.

`doctor` does not short-circuit: a missing workspace and an unreachable server
are reported together.

---

## 10. Security posture

M0 executes nothing on the model's behalf. The pieces M1–M3 must inherit:

* **`agent/workers.rs`** — per-role tool allow-lists (`fs.read`, `web.`, …) and
  `may_mutate`. Only the coder role may write, and that is data, not a comment.
* **`ToolCall` parsing already rejects malformed calls** (missing `function`, or
  a name that is not a string) instead of passing `None` onwards.
* **`Event::ToolCallsRequested`** is the single place the CLI learns that the
  model wants to act; M1 will replace the notice with a permission prompt.
* **`config.agent.max_iterations`** exists so M2's loop has a hard stop from the
  first commit that runs it.

Still required before M3: a `security/` module enforcing workspace-relative path
resolution (after symlink canonicalisation), command allow/deny policies,
output size caps and approval prompts. The spec's rule is binding: *the LLM
proposes, Rust decides.*

---

## 11. Testing strategy

| Layer | Technique | Why |
|-------|-----------|-----|
| Parsers (`json`, `toml`, `http`) | table tests, malformed inputs, boundary splits | they are the trust boundary |
| Streaming | split the same payload at every offset | chunking bugs are intermittent otherwise |
| Context | budget floods, tool-call adjacency | eviction rules are subtle |
| State machine | legal and illegal transitions, retry after failure | cheap and catches regressions |
| Orchestrator | `MockProvider` replaying scripted streams, errors, cancellation | no sockets, deterministic |
| REPL | `ScriptedInput` + `Vec<u8>` sink, assert transcripts | the session is the product |
| CLI | `Command` against the real binary | exit codes and stream separation |
| Provider | real HTTP against `server.rs` | chunked framing, error bodies, aborts |
| Docs | `tests/api_usage.rs` | examples cannot rot (no `rustdoc` offline) |

The fake server is a first-class test fixture: it splits its output into
token-sized records, so every provider test is also a chunk-boundary test.

---

## 12. Decisions and their rationale

| Decision | Choice | Alternative rejected |
|----------|--------|----------------------|
| Dependencies | zero for M0 | vendoring a registry snapshot: unreviewable, size-prohibitive |
| Concurrency | blocking + push sink | Tokio: no runtime available offline; the trait shape already fits |
| HTTP | hand-written, HTTP/1.1 plaintext | shelling out to `curl`: no streaming control, no timeout semantics |
| Streaming | byte-level NDJSON framer | line reads on the socket: misses records split across chunks |
| Head parsing | `Prefixed` replay | reading byte-wise: correct but pathologically slow on large headers |
| Context | estimated tokens + turn eviction | eager truncation: breaks tool-call adjacency |
| Planning | heuristic gate only | LLM planner per turn: doubles latency for greetings |
| Reflection | deterministic checks | model-as-judge: a round trip per reply |
| Commands | slash commands | flags-only: no way to inspect state mid-conversation |
| Colour | `NO_COLOR` + TTY + `TERM=dumb` + `--color` | always-on ANSI: breaks pipes and CI logs |
| Errors | typed + `hint()` | single opaque `String`: the user cannot act |

---

## 13. What M1 must change (and what it must not)

Add:

1. `tools/` — `ToolDefinition` (name, description, JSON schema), a `Tool` trait,
   a `ToolRegistry` (`register`, `get`, `list`, `execute`), and schema validation
   of arguments before dispatch (the proposal's hard requirement).
2. Send `tools` in `ChatRequest` (the field already exists and is omitted when
   empty).
3. In `Orchestrator`, replace the `Event::ToolCallsRequested` notice with: check
   `workers::may_use_tool`, validate arguments, execute, append a
   `Message::tool(...)`, then loop — bounded by `agent.max_iterations`.

Do not touch: the provider trait, the streaming scaffold, the context eviction
rules, or the REPL's rendering contract. Those are the parts M0 paid for.

---

## 14. Honest limitations

* HTTP/1.1 plaintext only; no TLS, proxies or redirects (arrives with `reqwest`).
* One conversation, no persistence (M4).
* Token estimates are heuristic; the server's real counts appear in `/stats`.
* No SIGINT handler yet: cancellation is plumbed, the terminal gesture is not.
* `Qwen3.5:4B` is a name string; nothing verifies the model's template really
  emits tool calls until M1 exercises it (`doctor --verbose` shows what
  `/api/show` claims).
* The 4B model's actual behaviour is untested here: every test uses a scripted
  server, so prompt quality and tool-call reliability remain to be measured on
  real hardware.
