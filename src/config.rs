//! Configuration: TOML file + environment + command-line, in that order.
//!
//! Precedence (highest wins):
//!
//! 1. command-line flags (`--model`, `--base-url`, `--config`)
//! 2. environment (`QWEN_AGENT_MODEL`, `OLLAMA_HOST`, `QWEN_AGENT_CONFIG`, …)
//! 3. the closest `default.toml` found while walking up from the CWD
//! 4. hard-coded defaults
//!
//! Every value is validated here rather than at first use, so a typo like
//! `temperature = "hot"` fails immediately with the field name and file.

use crate::error::ConfigError;
use crate::llm::request::Options;
use crate::toml::{self, Table, TomlValue};
use std::path::{Path, PathBuf};

pub const CONFIG_ENV: &str = "QWEN_AGENT_CONFIG";

/// Fully resolved configuration.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Config {
    pub agent: AgentConfig,
    pub llm: LlmConfig,
    pub logging: LoggingConfig,
    pub workspace: WorkspaceConfig,
    /// Where the configuration came from; `None` means defaults only.
    pub source: Option<PathBuf>,
    /// Non-fatal notes produced while loading (unknown keys, overrides).
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AgentConfig {
    pub name: String,
    pub max_iterations: u32,
    /// Appended to every conversation as the first system message.
    pub system_prompt: String,
    /// Ollama `keep_alive` value; keeps the 4B model resident between turns.
    pub keep_alive: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlmConfig {
    pub provider: String,
    pub base_url: String,
    pub model: String,
    pub temperature: f64,
    pub stream: bool,
    pub num_ctx: Option<u64>,
    pub num_predict: Option<i64>,
    /// Per-request budget for a non-streaming call, in seconds.
    pub timeout_secs: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoggingConfig {
    /// `error` | `warn` | `info` | `debug` | `trace`
    pub level: LogLevel,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WorkspaceConfig {
    pub root: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl LogLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            LogLevel::Error => "error",
            LogLevel::Warn => "warn",
            LogLevel::Info => "info",
            LogLevel::Debug => "debug",
            LogLevel::Trace => "trace",
        }
    }

    pub fn parse(s: &str) -> Option<LogLevel> {
        match s.trim().to_ascii_lowercase().as_str() {
            "error" => Some(LogLevel::Error),
            "warn" | "warning" => Some(LogLevel::Warn),
            "info" => Some(LogLevel::Info),
            "debug" => Some(LogLevel::Debug),
            "trace" => Some(LogLevel::Trace),
            _ => None,
        }
    }

    /// Effectively-verbose levels; used to decide whether to log request bodies.
    pub fn is_verbose(self) -> bool {
        self >= LogLevel::Debug
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        AgentConfig {
            name: "Qwen Agent".into(),
            max_iterations: 10,
            system_prompt: "You are Qwen, a careful local AI assistant running on the user's \
                            machine. Be concise and concrete. If you are unsure, say so."
                .into(),
            keep_alive: "10m".into(),
        }
    }
}

impl Default for LlmConfig {
    fn default() -> Self {
        LlmConfig {
            provider: "ollama".into(),
            base_url: "http://localhost:11434".into(),
            model: "qwen3.5:4b".into(),
            temperature: 0.7,
            stream: true,
            num_ctx: None,
            num_predict: None,
            timeout_secs: 600,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        LoggingConfig {
            level: LogLevel::Warn,
        }
    }
}

impl Default for WorkspaceConfig {
    fn default() -> Self {
        WorkspaceConfig {
            root: PathBuf::from("."),
        }
    }
}

/// Command-line overrides, collected by the CLI parser.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Overrides {
    pub config_path: Option<PathBuf>,
    pub base_url: Option<String>,
    pub model: Option<String>,
    pub temperature: Option<f64>,
    pub log_level: Option<String>,
    pub no_stream: bool,
}

/// Keys we accept in each section; anything else produces a warning so that a
/// misspelled option does not silently do nothing.
const KNOWN_KEYS: &[(&str, &[&str])] = &[
    (
        "agent",
        &["name", "max_iterations", "system_prompt", "keep_alive"],
    ),
    (
        "llm",
        &[
            "provider",
            "base_url",
            "model",
            "temperature",
            "stream",
            "num_ctx",
            "num_predict",
            "timeout_secs",
        ],
    ),
    ("logging", &["level"]),
    ("workspace", &["root"]),
];

impl Config {
    /// Loads configuration from the first source that exists.
    ///
    /// Never fails when no file is present (defaults are used); fails when a
    /// file exists but is unreadable, unparsable or invalid.
    pub fn load(overrides: &Overrides) -> Result<Config, ConfigError> {
        let (table, source) = match locate(overrides)? {
            Some(path) => {
                let text = std::fs::read_to_string(&path)
                    .map_err(|e| ConfigError::Unreadable(format!("{}: {e}", path.display())))?;
                let table = toml::parse(&text)
                    .map_err(|e| ConfigError::Parse(format!("{}: {e}", path.display())))?;
                (table, Some(path))
            }
            None => (Table::default(), None),
        };

        let mut cfg = Config {
            source: source.clone(),
            ..Config::default()
        };
        cfg.apply_table(&table)?;
        cfg.apply_env()?;
        cfg.apply_overrides(overrides)?;
        // Attribute validation failures to the file, so a user with three
        // config files in play knows which one to fix.
        if let Err(err) = cfg.validate() {
            return Err(match (err, &source) {
                (ConfigError::Invalid { field, reason }, Some(path)) => {
                    ConfigError::InvalidInFile {
                        path: path.display().to_string(),
                        field,
                        reason,
                    }
                }
                (other, _) => other,
            });
        }
        cfg.finish_loading(table);
        Ok(cfg)
    }

    /// Loads from an explicit TOML string. Used by tests and `config --check`.
    pub fn from_toml_str(text: &str) -> Result<Config, ConfigError> {
        let table = toml::parse(text).map_err(|e| ConfigError::Parse(e.to_string()))?;
        let mut cfg = Config::default();
        cfg.apply_table(&table)?;
        cfg.validate()?;
        cfg.warn_unknown(&table);
        Ok(cfg)
    }

    fn finish_loading(&mut self, table: Table) {
        self.warn_unknown(&table);
        if self.logging.level.is_verbose() && self.llm.stream {
            self.warnings.push(
                "debug logging with streaming enabled may interleave with model output".into(),
            );
        }
    }

    fn warn_unknown(&mut self, table: &Table) {
        for (section, entries) in KNOWN_KEYS.iter() {
            for (key, _) in table.entries(section) {
                if !entries.contains(&key.as_str()) {
                    self.warnings
                        .push(format!("unknown key `{section}.{key}` (ignored)"));
                }
            }
        }
        for name in table.section_names() {
            if !KNOWN_KEYS.iter().any(|(s, _)| *s == name) {
                self.warnings
                    .push(format!("unknown section `[{name}]` (ignored)"));
            }
        }
    }

    fn apply_table(&mut self, table: &Table) -> Result<(), ConfigError> {
        if let Some(v) = table.get("agent", "name") {
            self.agent.name = string_value("agent.name", v)?.to_string();
        }
        if let Some(v) = table.get("agent", "max_iterations") {
            let n = int_value("agent.max_iterations", v)?;
            self.agent.max_iterations = u32::try_from(n).map_err(|_| ConfigError::Invalid {
                field: "agent.max_iterations".into(),
                reason: format!("{n} is not a positive integer that fits in 32 bits"),
            })?;
        }
        if let Some(v) = table.get("agent", "system_prompt") {
            self.agent.system_prompt = string_value("agent.system_prompt", v)?.to_string();
        }
        if let Some(v) = table.get("agent", "keep_alive") {
            self.agent.keep_alive = string_value("agent.keep_alive", v)?.to_string();
        }

        if let Some(v) = table.get("llm", "provider") {
            self.llm.provider = string_value("llm.provider", v)?.to_string();
        }
        if let Some(v) = table.get("llm", "base_url") {
            self.llm.base_url = string_value("llm.base_url", v)?.to_string();
        }
        if let Some(v) = table.get("llm", "model") {
            self.llm.model = string_value("llm.model", v)?.to_string();
        }
        if let Some(v) = table.get("llm", "temperature") {
            self.llm.temperature = number_value("llm.temperature", v)?;
        }
        if let Some(v) = table.get("llm", "stream") {
            self.llm.stream = bool_value("llm.stream", v)?;
        }
        if let Some(v) = table.get("llm", "num_ctx") {
            let n = int_value("llm.num_ctx", v)?;
            // 0 means "let the server decide", matching the generated config.
            self.llm.num_ctx = if n > 0 { Some(n as u64) } else { None };
        }
        if let Some(v) = table.get("llm", "num_predict") {
            let n = int_value("llm.num_predict", v)?;
            self.llm.num_predict = if n >= 0 { Some(n) } else { None };
        }
        if let Some(v) = table.get("llm", "timeout_secs") {
            self.llm.timeout_secs = int_value("llm.timeout_secs", v)? as u64;
        }

        if let Some(v) = table.get("logging", "level") {
            let raw = string_value("logging.level", v)?;
            self.logging.level = LogLevel::parse(raw).ok_or_else(|| ConfigError::Invalid {
                field: "logging.level".into(),
                reason: format!("`{raw}` is not one of error/warn/info/debug/trace"),
            })?;
        }

        if let Some(v) = table.get("workspace", "root") {
            self.workspace.root = PathBuf::from(string_value("workspace.root", v)?);
        }
        Ok(())
    }

    fn apply_env(&mut self) -> Result<(), ConfigError> {
        // `OLLAMA_HOST` is honoured because that is what the Ollama ecosystem
        // uses; the QWEN_AGENT_* names win when both are set.
        if let Some(host) = non_empty_env("OLLAMA_HOST") {
            let normalized = normalize_ollama_host(&host);
            if self.llm.base_url == LlmConfig::default().base_url {
                self.llm.base_url = normalized;
            }
        }
        if let Some(v) = non_empty_env("QWEN_AGENT_BASE_URL") {
            self.llm.base_url = v;
        }
        if let Some(v) = non_empty_env("QWEN_AGENT_MODEL") {
            self.llm.model = v;
        }
        if let Some(v) = non_empty_env("QWEN_AGENT_TEMPERATURE") {
            let parsed: f64 = v.parse().map_err(|_| ConfigError::Invalid {
                field: "QWEN_AGENT_TEMPERATURE".into(),
                reason: format!("`{v}` is not a number"),
            })?;
            self.llm.temperature = parsed;
        }
        if let Some(v) = non_empty_env("QWEN_AGENT_LOG") {
            self.logging.level = LogLevel::parse(&v).ok_or_else(|| ConfigError::Invalid {
                field: "QWEN_AGENT_LOG".into(),
                reason: format!("`{v}` is not one of error/warn/info/debug/trace"),
            })?;
        }
        Ok(())
    }

    fn apply_overrides(&mut self, o: &Overrides) -> Result<(), ConfigError> {
        if let Some(url) = &o.base_url {
            self.llm.base_url = url.clone();
        }
        if let Some(model) = &o.model {
            self.llm.model = model.clone();
        }
        if let Some(t) = o.temperature {
            self.llm.temperature = t;
        }
        if o.no_stream {
            self.llm.stream = false;
        }
        if let Some(level) = &o.log_level {
            self.logging.level = LogLevel::parse(level).ok_or_else(|| ConfigError::Invalid {
                field: "--log-level".into(),
                reason: format!("`{level}` is not one of error/warn/info/debug/trace"),
            })?;
        }
        Ok(())
    }

    /// Semantic validation that cannot be expressed by types alone.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if !(0.0..=2.0).contains(&self.llm.temperature) {
            return Err(ConfigError::Invalid {
                field: "llm.temperature".into(),
                reason: format!(
                    "{} is outside 0.0..=2.0 (0 is deterministic, 2 is very random)",
                    self.llm.temperature
                ),
            });
        }
        if self.llm.model.trim().is_empty() {
            return Err(ConfigError::Invalid {
                field: "llm.model".into(),
                reason: "must not be empty".into(),
            });
        }
        if self.llm.base_url.trim().is_empty() {
            return Err(ConfigError::Invalid {
                field: "llm.base_url".into(),
                reason: "must not be empty".into(),
            });
        }
        if self.agent.max_iterations == 0 {
            return Err(ConfigError::Invalid {
                field: "agent.max_iterations".into(),
                reason: "must be at least 1".into(),
            });
        }
        if self.llm.provider != "ollama" {
            return Err(ConfigError::Invalid {
                field: "llm.provider".into(),
                reason: format!(
                    "`{}` is not supported in this build; only `ollama` is implemented",
                    self.llm.provider
                ),
            });
        }
        if self.llm.timeout_secs == 0 {
            return Err(ConfigError::Invalid {
                field: "llm.timeout_secs".into(),
                reason: "must be at least 1 second".into(),
            });
        }
        Ok(())
    }

    /// Sampling options derived from the config.
    pub fn options(&self) -> Options {
        Options {
            temperature: self.llm.temperature,
            num_ctx: self.llm.num_ctx,
            num_predict: self.llm.num_predict,
            seed: None,
            stop: Vec::new(),
        }
    }

    /// The annotated configuration file written by `config --init`.
    pub fn default_toml() -> String {
        crate::config_template::DEFAULT_TOML.to_string()
    }

    /// Search order for the config file, for `doctor` and error messages.
    pub fn search_paths(overrides: &Overrides) -> Vec<PathBuf> {
        let mut paths = Vec::new();
        if let Some(p) = overrides
            .config_path
            .clone()
            .or_else(|| non_empty_env(CONFIG_ENV).map(PathBuf::from))
        {
            paths.push(p);
        } else {
            if let Ok(cwd) = std::env::current_dir() {
                for dir in cwd.ancestors() {
                    for name in ["qwen-agent.toml", "config/default.toml", ".qwen-agent.toml"] {
                        paths.push(dir.join(name));
                    }
                }
            }
            if let Some(home) = non_empty_env("HOME") {
                paths.push(PathBuf::from(home).join(".config/qwen-agent/config.toml"));
            }
        }
        paths
    }
}

fn locate(overrides: &Overrides) -> Result<Option<PathBuf>, ConfigError> {
    let explicit = overrides
        .config_path
        .clone()
        .or_else(|| non_empty_env(CONFIG_ENV).map(PathBuf::from));
    if let Some(path) = explicit {
        if path.is_file() {
            return Ok(Some(path));
        }
        return Err(ConfigError::Unreadable(format!(
            "{} does not exist",
            path.display()
        )));
    }
    Ok(Config::search_paths(overrides)
        .into_iter()
        .find(|p| p.is_file()))
}

/// `OLLAMA_HOST` is normally `127.0.0.1:11434`; accept both forms.
fn normalize_ollama_host(raw: &str) -> String {
    let raw = raw.trim();
    if raw.starts_with("http://") || raw.starts_with("https://") {
        raw.to_string()
    } else {
        format!("http://{raw}")
    }
}

fn non_empty_env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

fn string_value<'a>(field: &str, v: &'a TomlValue) -> Result<&'a str, ConfigError> {
    v.as_str().ok_or_else(|| ConfigError::Invalid {
        field: field.into(),
        reason: format!("expected a string, found {}", v.type_name()),
    })
}

fn int_value(field: &str, v: &TomlValue) -> Result<i64, ConfigError> {
    v.as_i64().ok_or_else(|| ConfigError::Invalid {
        field: field.into(),
        reason: format!("expected an integer, found {}", v.type_name()),
    })
}

fn number_value(field: &str, v: &TomlValue) -> Result<f64, ConfigError> {
    v.as_f64().ok_or_else(|| ConfigError::Invalid {
        field: field.into(),
        reason: format!("expected a number, found {}", v.type_name()),
    })
}

fn bool_value(field: &str, v: &TomlValue) -> Result<bool, ConfigError> {
    v.as_bool().ok_or_else(|| ConfigError::Invalid {
        field: field.into(),
        reason: format!("expected a boolean, found {}", v.type_name()),
    })
}

/// True when `path` looks like a configuration file we can read.
pub fn is_readable_config(path: &Path) -> bool {
    path.is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
[agent]
name = "Qwen Agent"
max_iterations = 10

[llm]
provider = "ollama"
base_url = "http://localhost:11434"
model = "Qwen3.5:4B"
temperature = 0.7
stream = true

[logging]
level = "info"

[workspace]
root = "."
"#;

    #[test]
    fn parses_the_documented_sample_config() {
        let cfg = Config::from_toml_str(SAMPLE).unwrap();
        assert_eq!(cfg.agent.name, "Qwen Agent");
        assert_eq!(cfg.agent.max_iterations, 10);
        assert_eq!(cfg.llm.model, "Qwen3.5:4B");
        assert_eq!(cfg.llm.temperature, 0.7);
        assert!(cfg.llm.stream);
        assert_eq!(cfg.logging.level, LogLevel::Info);
        assert_eq!(cfg.workspace.root, PathBuf::from("."));
        assert!(
            cfg.warnings.is_empty(),
            "unexpected warnings: {:?}",
            cfg.warnings
        );
    }

    #[test]
    fn defaults_are_valid() {
        Config::default().validate().unwrap();
        let cfg = Config::from_toml_str("").unwrap();
        assert_eq!(cfg.llm.model, LlmConfig::default().model);
    }

    #[test]
    fn generated_default_toml_round_trips() {
        let text = Config::default_toml();
        let cfg = Config::from_toml_str(&text).expect("generated TOML must parse");
        assert_eq!(
            cfg,
            Config {
                source: None,
                warnings: Vec::new(),
                ..Config::default()
            }
        );
        assert!(cfg.warnings.is_empty(), "{:?}", cfg.warnings);
        // The template documents the 0 / -1 sentinels.
        assert_eq!(cfg.llm.num_ctx, None);
        assert_eq!(cfg.llm.num_predict, None);
    }

    #[test]
    fn rejects_out_of_range_temperature() {
        let err = Config::from_toml_str("[llm]\ntemperature = 5.0\n").unwrap_err();
        assert!(matches!(err, ConfigError::Invalid { field, .. } if field == "llm.temperature"));
        let err = Config::from_toml_str("[llm]\ntemperature = -1.0\n").unwrap_err();
        assert!(err.to_string().contains("outside"));
    }

    #[test]
    fn rejects_wrong_types_with_field_names() {
        let err = Config::from_toml_str("[llm]\nstream = \"yes\"\n").unwrap_err();
        assert!(err.to_string().contains("llm.stream"), "{err}");
        assert!(err.to_string().contains("boolean"), "{err}");

        let err = Config::from_toml_str("[llm]\nmodel = 4\n").unwrap_err();
        assert!(err.to_string().contains("llm.model"), "{err}");
    }

    #[test]
    fn rejects_unknown_provider_and_bad_level() {
        assert!(Config::from_toml_str("[llm]\nprovider = \"openai\"\n").is_err());
        let err = Config::from_toml_str("[logging]\nlevel = \"loud\"\n").unwrap_err();
        assert!(err.to_string().contains("loud"), "{err}");
    }

    #[test]
    fn warns_about_unknown_keys_but_still_loads() {
        let cfg = Config::from_toml_str("[llm]\nmodel = \"m\"\nmodle = \"typo\"\n").unwrap();
        assert_eq!(cfg.llm.model, "m");
        assert!(
            cfg.warnings.iter().any(|w| w.contains("llm.modle")),
            "{:?}",
            cfg.warnings
        );

        let cfg = Config::from_toml_str("[bogus]\nx = 1\n").unwrap();
        assert!(
            cfg.warnings.iter().any(|w| w.contains("[bogus]")),
            "{:?}",
            cfg.warnings
        );
    }

    #[test]
    fn zero_iterations_is_rejected() {
        let err = Config::from_toml_str("[agent]\nmax_iterations = 0\n").unwrap_err();
        assert!(err.to_string().contains("max_iterations"), "{err}");
    }

    #[test]
    fn negative_iterations_are_rejected() {
        let err = Config::from_toml_str("[agent]\nmax_iterations = -3\n").unwrap_err();
        assert!(matches!(err, ConfigError::Invalid { .. }), "{err}");
    }

    #[test]
    fn overrides_beat_file_values() {
        let mut cfg = Config::from_toml_str(SAMPLE).unwrap();
        cfg.apply_overrides(&Overrides {
            model: Some("qwen3.5:8b".into()),
            temperature: Some(0.1),
            no_stream: true,
            log_level: Some("debug".into()),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(cfg.llm.model, "qwen3.5:8b");
        assert_eq!(cfg.llm.temperature, 0.1);
        assert!(!cfg.llm.stream);
        assert_eq!(cfg.logging.level, LogLevel::Debug);
    }

    #[test]
    fn bad_override_level_is_reported() {
        let mut cfg = Config::default();
        let err = cfg
            .apply_overrides(&Overrides {
                log_level: Some("chatty".into()),
                ..Default::default()
            })
            .unwrap_err();
        assert!(err.to_string().contains("--log-level"), "{err}");
    }

    #[test]
    fn ollama_host_env_is_normalized() {
        assert_eq!(
            normalize_ollama_host("127.0.0.1:11434"),
            "http://127.0.0.1:11434"
        );
        assert_eq!(
            normalize_ollama_host("http://box:11434"),
            "http://box:11434"
        );
    }

    #[test]
    fn explicit_missing_config_path_is_an_error() {
        let err = Config::load(&Overrides {
            config_path: Some(PathBuf::from("/nonexistent/qwen-agent.toml")),
            ..Default::default()
        })
        .unwrap_err();
        assert!(matches!(err, ConfigError::Unreadable(_)), "{err}");
    }

    #[test]
    fn search_paths_always_start_with_explicit_path() {
        let paths = Config::search_paths(&Overrides {
            config_path: Some(PathBuf::from("/tmp/x.toml")),
            ..Default::default()
        });
        assert_eq!(paths, vec![PathBuf::from("/tmp/x.toml")]);
    }

    #[test]
    fn log_level_ordering_supports_verbosity_checks() {
        assert!(LogLevel::Debug.is_verbose());
        assert!(!LogLevel::Info.is_verbose());
        assert!(LogLevel::Trace > LogLevel::Warn);
        assert_eq!(LogLevel::parse("WARNING"), Some(LogLevel::Warn));
        assert_eq!(LogLevel::parse("???"), None);
    }
}
