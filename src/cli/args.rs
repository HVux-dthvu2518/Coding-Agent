//! Argument parsing.
//!
//! Hand-written rather than derived, for the same reason as everything else in
//! this build: no crates. The shape mirrors the clap interface the spec asks for
//! (`chat`, `run`, `models`, `config`, `doctor`) with clap-compatible long/short
//! flags, `--flag=value` and `--flag value` forms, `--` end-of-flags, and
//! `-h/--help` / `-V/--version`.
//!
//! Parsing is pure: `parse_from` takes an iterator of arguments and returns a
//! value or a [`ParseOutcome`], so every CLI path is unit-tested without a
//! subprocess.

use crate::config::Overrides;
use std::path::PathBuf;

/// Top-level command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// Interactive conversation.
    Chat {
        /// Run these lines before handing over to stdin (non-interactive demos).
        script: Vec<String>,
        /// Exit after the script instead of reading stdin.
        no_repl: bool,
    },
    /// One-shot task from the command line or stdin.
    Run {
        prompt: Option<String>,
        /// Read the prompt from stdin.
        stdin: bool,
    },
    /// List installed models.
    Models {
        /// Print machine-readable JSON.
        json: bool,
    },
    /// Inspect configuration.
    Config {
        /// Print the effective configuration as TOML.
        toml: bool,
        /// Write a default configuration file and exit.
        init: Option<PathBuf>,
        /// Validate the configuration and exit.
        check: bool,
    },
    /// Check the environment end to end.
    Doctor {
        /// Also probe model capabilities via `/api/show`.
        verbose: bool,
    },
}

/// Global options that apply to every command.
#[derive(Debug, Clone, PartialEq)]
pub struct GlobalOptions {
    pub overrides: Overrides,
    pub color: String,
}

/// A parsed invocation.
#[derive(Debug, Clone, PartialEq)]
pub struct Cli {
    pub command: Command,
    pub global: GlobalOptions,
}

/// Result of parsing: run something, or print and exit 0/1.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseOutcome {
    Run(Box<Cli>),
    /// Text to print on stdout, exit code 0.
    Help(String),
    /// Text to print on stdout, exit code 0.
    Version(String),
    /// Text to print on stderr, exit code 2.
    Error {
        message: String,
        usage: bool,
    },
}

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const BIN: &str = "qwen-agent";

/// Parses process arguments (excluding argv[0]).
pub fn parse_from<I, S>(args: I) -> ParseOutcome
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let args: Vec<String> = args.into_iter().map(Into::into).collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        return ParseOutcome::Help(help_text());
    }
    if args.iter().any(|a| a == "--version" || a == "-V") {
        return ParseOutcome::Version(format!("{BIN} {VERSION}"));
    }

    let mut overrides = Overrides::default();
    let mut color = "auto".to_string();
    let mut positional: Vec<String> = Vec::new();
    let mut script: Vec<String> = Vec::new();
    let mut no_repl = false;
    let mut stdin = false;
    let mut json = false;
    let mut toml = false;
    let mut check = false;
    let mut verbose = false;
    let mut init: Option<PathBuf> = None;

    let mut iter = args.into_iter().peekable();
    let mut no_more_flags = false;
    while let Some(arg) = iter.next() {
        if no_more_flags {
            positional.push(arg);
            continue;
        }
        if arg == "--" {
            no_more_flags = true;
            continue;
        }

        // Long flags, with optional `=value`.
        if let Some(rest) = arg.strip_prefix("--") {
            let (name, inline) = match rest.split_once('=') {
                Some((n, v)) => (n, Some(v.to_string())),
                None => (rest, None),
            };
            // A small helper: `?` cannot be used here because the function
            // returns `ParseOutcome`, not `Result`.
            macro_rules! value_of {
                ($flag:expr) => {
                    match take_value(&mut iter, inline.clone(), $flag) {
                        Ok(v) => v,
                        Err(outcome) => return outcome,
                    }
                };
            }
            match name {
                "model" => overrides.model = Some(value_of!("model")),
                "base-url" => overrides.base_url = Some(value_of!("base-url")),
                "config" => overrides.config_path = Some(PathBuf::from(value_of!("config"))),
                "temperature" => {
                    let raw = value_of!("temperature");
                    match raw.parse::<f64>() {
                        Ok(v) => overrides.temperature = Some(v),
                        Err(_) => {
                            return ParseOutcome::Error {
                                message: format!("`--temperature` expects a number, got `{raw}`"),
                                usage: true,
                            };
                        }
                    }
                }
                "log-level" | "log" => overrides.log_level = Some(value_of!("log-level")),
                "color" => color = value_of!("color"),
                "no-stream" => overrides.no_stream = true,
                "no-repl" => no_repl = true,
                "stdin" => stdin = true,
                "json" => json = true,
                "toml" => toml = true,
                "init" => {
                    // `--init[=path]`: the value is optional. A following
                    // non-flag argument is taken as the path; otherwise the
                    // default file name is used.
                    let path = match inline.clone() {
                        Some(v) => v,
                        None => match iter.peek() {
                            Some(next) if !next.starts_with('-') => iter.next().unwrap_or_default(),
                            _ => "qwen-agent.toml".to_string(),
                        },
                    };
                    init = Some(PathBuf::from(path))
                }
                "check" => check = true,
                "verbose" => verbose = true,
                "script" => script.push(value_of!("script")),
                other => {
                    return ParseOutcome::Error {
                        message: format!("unknown option `--{other}`"),
                        usage: true,
                    };
                }
            }
            continue;
        }

        // Short flags. Only those that take no value are supported as clusters.
        if arg.len() > 1 && arg.starts_with('-') && arg != "-" {
            let mut recognised = true;
            for ch in arg.chars().skip(1) {
                match ch {
                    'n' => no_repl = true,
                    'v' => verbose = true,
                    'j' => json = true,
                    _ => {
                        recognised = false;
                        break;
                    }
                }
            }
            if recognised {
                continue;
            }
            return ParseOutcome::Error {
                message: format!("unknown option `{arg}`"),
                usage: true,
            };
        }

        positional.push(arg);
    }

    if crate::cli::ui::ColorMode::parse(&color).is_none() {
        return ParseOutcome::Error {
            message: format!("`--color` expects auto|always|never, got `{color}`"),
            usage: true,
        };
    }

    let (command_name, rest) = match positional.split_first() {
        Some((name, rest)) => (name.as_str(), rest.to_vec()),
        None => ("chat", Vec::new()),
    };

    let command = match command_name {
        "chat" => {
            // Any leftover positional arguments in `chat` are a mistake.
            if let Some(extra) = rest.first() {
                return ParseOutcome::Error {
                    message: format!(
                        "unexpected argument `{extra}` for `chat` (use `run` for one-shot prompts)"
                    ),
                    usage: true,
                };
            }
            Command::Chat { script, no_repl }
        }
        "run" => {
            let prompt = if rest.is_empty() {
                None
            } else {
                Some(rest.join(" "))
            };
            if prompt.is_none() && !stdin {
                return ParseOutcome::Error {
                    message: "`run` needs a prompt, e.g. `qwen-agent run \"explain this repo\"`, or --stdin"
                        .into(),
                    usage: true,
                };
            }
            Command::Run { prompt, stdin }
        }
        "models" => Command::Models { json },
        "config" => {
            if check && init.is_some() {
                return ParseOutcome::Error {
                    message: "`--check` and `--init` cannot be combined".into(),
                    usage: true,
                };
            }
            Command::Config { toml, init, check }
        }
        "doctor" => Command::Doctor { verbose },
        other => {
            return ParseOutcome::Error {
                message: format!("unknown command `{other}`"),
                usage: true,
            };
        }
    };

    // CLI values win over env, which wins over the file (see `config.rs`).
    let global = GlobalOptions { overrides, color };
    ParseOutcome::Run(Box::new(Cli { command, global }))
}

/// Reads a long flag's value: inline (`--flag=v`) or the next argument.
fn take_value<I>(
    iter: &mut std::iter::Peekable<I>,
    inline: Option<String>,
    flag: &str,
) -> Result<String, ParseOutcome>
where
    I: Iterator<Item = String>,
{
    if let Some(v) = inline {
        return Ok(v);
    }
    iter.next().ok_or_else(|| ParseOutcome::Error {
        message: format!("`--{flag}` needs a value"),
        usage: true,
    })
}

pub fn help_text() -> String {
    format!(
        "\
{BIN} {VERSION} — local-first CLI agent for Qwen3.5:4B via Ollama

USAGE:
    {BIN} [GLOBAL OPTIONS] [COMMAND]

COMMANDS:
    chat                Interactive conversation (default)
    run <prompt>        Answer one prompt and exit
    models              List installed Ollama models
    config              Show the effective configuration
    doctor              Check Ollama, the model and the config
    help                Show this help

GLOBAL OPTIONS:
    -c, --config <path>     Configuration file to use
        --base-url <url>    Ollama endpoint (default http://localhost:11434)
        --model <name>      Model name (default qwen3.5:4b)
        --temperature <n>   Sampling temperature, 0.0–2.0
        --no-stream         Disable streaming output
        --log-level <lvl>   error|warn|info|debug|trace
        --color <when>      auto|always|never
    -h, --help              Show this help
    -V, --version           Show the version

COMMAND OPTIONS:
    run --stdin             Read the prompt from stdin
    models --json           Machine-readable output
    config --toml           Print the effective config as TOML
    config --check          Validate the configuration and exit
    config --init [path]    Write a default configuration file
    doctor --verbose        Probe model capabilities

ENVIRONMENT:
    QWEN_AGENT_CONFIG       Configuration file path
    QWEN_AGENT_BASE_URL     Ollama endpoint
    QWEN_AGENT_MODEL        Model name
    QWEN_AGENT_TEMPERATURE  Sampling temperature
    QWEN_AGENT_LOG          Log level
    OLLAMA_HOST             Ollama endpoint (if QWEN_AGENT_BASE_URL is unset)
    NO_COLOR                Disable coloured output

EXAMPLES:
    {BIN} chat
    {BIN} run \"what does the ? operator do in Rust?\"
    {BIN} --model qwen3.5:8b run \"summarise src/http.rs\"
    echo \"explain this stack trace\" | {BIN} run --stdin
    {BIN} doctor
"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(args: &[&str]) -> Cli {
        match parse_from(args.iter().copied()) {
            ParseOutcome::Run(cli) => *cli,
            other => panic!("expected Run, got {other:?}"),
        }
    }

    fn err(args: &[&str]) -> String {
        match parse_from(args.iter().copied()) {
            ParseOutcome::Error { message, .. } => message,
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[test]
    fn no_arguments_defaults_to_chat() {
        assert_eq!(
            ok(&[]).command,
            Command::Chat {
                script: vec![],
                no_repl: false
            }
        );
    }

    #[test]
    fn help_and_version_short_circuit() {
        assert!(matches!(parse_from(["--help"]), ParseOutcome::Help(_)));
        assert!(matches!(parse_from(["-h"]), ParseOutcome::Help(_)));
        match parse_from(["-V"]) {
            ParseOutcome::Version(v) => assert!(v.contains(BIN)),
            other => panic!("{other:?}"),
        }
        // Help wins over other commands, like clap.
        assert!(matches!(
            parse_from(["doctor", "--help"]),
            ParseOutcome::Help(_)
        ));
    }

    #[test]
    fn run_takes_the_rest_of_the_arguments_as_the_prompt() {
        let cli = ok(&["run", "explain", "this", "repo"]);
        assert_eq!(
            cli.command,
            Command::Run {
                prompt: Some("explain this repo".into()),
                stdin: false
            }
        );
    }

    #[test]
    fn run_requires_a_prompt_unless_stdin() {
        assert!(err(&["run"]).contains("needs a prompt"));
        let cli = ok(&["run", "--stdin"]);
        assert_eq!(
            cli.command,
            Command::Run {
                prompt: None,
                stdin: true
            }
        );
    }

    #[test]
    fn global_flags_work_before_and_after_the_command() {
        let a = ok(&["--model", "m1", "run", "hi"]);
        let b = ok(&["run", "--model", "m1", "hi"]);
        assert_eq!(a.global.overrides.model.as_deref(), Some("m1"));
        assert_eq!(b.global.overrides.model.as_deref(), Some("m1"));
        assert_eq!(
            b.command,
            Command::Run {
                prompt: Some("hi".into()),
                stdin: false
            }
        );
    }

    #[test]
    fn equals_form_and_short_clusters() {
        let cli = ok(&[
            "--model=qwen3.5:8b",
            "--base-url=http://box:11434",
            "doctor",
        ]);
        assert_eq!(cli.global.overrides.model.as_deref(), Some("qwen3.5:8b"));
        assert_eq!(
            cli.global.overrides.base_url.as_deref(),
            Some("http://box:11434")
        );

        let cli = ok(&["models", "-j"]);
        assert_eq!(cli.command, Command::Models { json: true });
    }

    #[test]
    fn double_dash_stops_flag_parsing() {
        let cli = ok(&["run", "--", "--not-a-flag"]);
        assert_eq!(
            cli.command,
            Command::Run {
                prompt: Some("--not-a-flag".into()),
                stdin: false
            }
        );
    }

    #[test]
    fn temperature_is_validated_at_parse_time_and_at_config_time() {
        let cli = ok(&["--temperature", "0.2", "run", "x"]);
        assert_eq!(cli.global.overrides.temperature, Some(0.2));
        assert!(err(&["--temperature", "hot", "run", "x"]).contains("expects a number"));
    }

    #[test]
    fn color_mode_is_validated() {
        assert_eq!(ok(&["--color", "never"]).global.color, "never");
        assert!(err(&["--color", "rainbow"]).contains("auto|always|never"));
    }

    #[test]
    fn config_init_and_check_are_mutually_exclusive() {
        assert!(err(&["config", "--check", "--init"]).contains("cannot be combined"));
        let cli = ok(&["config", "--init"]);
        assert_eq!(
            cli.command,
            Command::Config {
                toml: false,
                init: Some(PathBuf::from("qwen-agent.toml")),
                check: false
            }
        );
        let cli = ok(&["config", "--init", "custom.toml"]);
        match cli.command {
            Command::Config { init, .. } => assert_eq!(init, Some(PathBuf::from("custom.toml"))),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn missing_flag_values_are_reported_with_the_flag_name() {
        assert!(err(&["--model"]).contains("`--model` needs a value"));
        assert!(err(&["--config"]).contains("needs a value"));
    }

    #[test]
    fn unknown_options_and_commands_are_rejected() {
        assert!(err(&["--nope"]).contains("unknown option `--nope`"));
        assert!(err(&["-Z"]).contains("unknown option `-Z`"));
        assert!(err(&["frobnicate"]).contains("unknown command `frobnicate`"));
        assert!(err(&["chat", "extra"]).contains("unexpected argument"));
    }

    #[test]
    fn help_text_documents_every_command_and_flag() {
        let help = help_text();
        for needle in [
            "chat",
            "run",
            "models",
            "config",
            "doctor",
            "--base-url",
            "--model",
            "--no-stream",
            "--log-level",
            "--color",
            "QWEN_AGENT_MODEL",
            "OLLAMA_HOST",
            "NO_COLOR",
        ] {
            assert!(help.contains(needle), "help is missing {needle}");
        }
    }

    #[test]
    fn script_flags_are_collected_in_order() {
        let cli = ok(&["chat", "--script", "/status", "--script", "/exit"]);
        assert_eq!(
            cli.command,
            Command::Chat {
                script: vec!["/status".into(), "/exit".into()],
                no_repl: false
            }
        );
    }

    #[test]
    fn doctor_verbose_short_flag() {
        assert_eq!(
            ok(&["doctor", "-v"]).command,
            Command::Doctor { verbose: true }
        );
        assert_eq!(ok(&["doctor"]).command, Command::Doctor { verbose: false });
    }
}
