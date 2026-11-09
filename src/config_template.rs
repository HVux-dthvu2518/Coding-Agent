//! The annotated configuration file written by `config --init`.
//!
//! Kept as a literal rather than generated so that it can carry comments, and
//! validated by `default_toml_is_valid` so it cannot drift from the loader.

/// Contents of the default `qwen-agent.toml`.
pub const DEFAULT_TOML: &str = r#"# qwen-agent configuration
#
# Precedence: CLI flags > QWEN_AGENT_* environment variables > this file.
# Inspect the effective values with `qwen-agent config`.

[agent]
name = "Qwen Agent"
# Upper bound on loop iterations once the autonomous loop lands in M2.
max_iterations = 10
# Ollama keep_alive; keeps the 4B model resident between turns.
keep_alive = "10m"
# First message sent on every request.
system_prompt = "You are Qwen, a careful local AI assistant running on the user's machine. Be concise and concrete. If you are unsure, say so."

[llm]
provider = "ollama"
base_url = "http://localhost:11434"
# Case-insensitive; a missing tag means ":latest".
model = "qwen3.5:4b"
# 0.0 is deterministic, 2.0 is very random.
temperature = 0.7
stream = true
# Context window requested from Ollama. Leave 0 to use the server default.
num_ctx = 0
# Token cap per reply; -1 means "no explicit cap".
num_predict = -1

[logging]
# error | warn | info | debug | trace  (logs go to stderr)
level = "warn"

[workspace]
root = "."
"#;
