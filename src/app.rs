//! Composition root: turns a parsed [`Cli`] into work.
//!
//! This is the only module that knows about *all* subsystems. Everything else
//! receives what it needs through its constructor, which is what keeps the agent
//! core testable and provider-agnostic.
//!
//! Commands are dispatched here rather than in `main.rs` so that the whole
//! application can be exercised by tests with an in-memory stdin/stdout.

use crate::agent::{Context, Orchestrator};
use crate::cli::args::{Cli, Command};
use crate::cli::doctor;
use crate::cli::repl::{self, Labels, ReaderInput, ScriptedInput, Session};
use crate::cli::ui::{self, ColorMode, Style};
use crate::config::{Config, Overrides};
use crate::error::AppError;
use crate::llm::provider::LlmProvider as _;
use crate::llm::{OllamaConfig, OllamaProvider};
use crate::logging;
use std::io::{BufReader, Write};

/// Exit codes, matching shell conventions.
pub mod exit {
    pub const OK: i32 = 0;
    pub const FAILURE: i32 = 1;
    pub const USAGE: i32 = 2;
}

/// Runs a parsed CLI. Returns the process exit code.
pub fn run(cli: Cli) -> i32 {
    // Colour must be resolved before anything prints.
    let color = ColorMode::parse(&cli.global.color).unwrap_or(ColorMode::Auto);
    let style = Style::new(ui::use_color(color));

    let config = match Config::load(&cli.global.overrides) {
        Ok(cfg) => cfg,
        // `config --check` and `config --init` must still run when the file is
        // broken: reporting the problem is their whole job.
        // (the load is re-run, and reported, by `run_config`)
        Err(_) if matches!(cli.command, Command::Config { .. }) => Config::default(),
        Err(err) => {
            eprintln!("{} {err}", style.red("error:"));
            if let AppError::Config(_) = AppError::Config(err.clone()) {
                if let Some(hint) = AppError::Config(err).hint() {
                    eprintln!("       {}", style.dim(hint));
                }
            }
            return exit::FAILURE;
        }
    };
    logging::init(config.logging.level);
    for warning in &config.warnings {
        logging::warn("config", warning);
    }

    let result = match &cli.command {
        Command::Chat { script, no_repl } => run_chat(&config, &style, script, *no_repl),
        Command::Run { prompt, stdin } => run_once(&config, &style, prompt.clone(), *stdin),
        Command::Models { json } => run_models(&config, &style, *json),
        Command::Config { toml, init, check } => run_config(
            &config,
            &style,
            &mut std::io::stdout(),
            *toml,
            init.clone(),
            *check,
            &cli.global.overrides,
        ),
        Command::Doctor { verbose } => run_doctor(&config, &style, *verbose),
    };

    match result {
        Ok(code) => code,
        Err(err) => {
            eprintln!("{} {err}", style.red("error:"));
            if let Some(hint) = err.hint() {
                eprintln!("       {}", style.dim(hint));
            }
            exit::FAILURE
        }
    }
}

fn build_provider(config: &Config) -> Result<OllamaProvider, AppError> {
    let provider = OllamaProvider::new(&OllamaConfig {
        base_url: config.llm.base_url.clone(),
        model: config.llm.model.clone(),
    })
    .map_err(AppError::Llm)?;
    Ok(provider)
}

/// Builds the orchestrator shared by `chat` and `run`.
fn build_orchestrator(config: &Config) -> Result<Orchestrator<OllamaProvider>, AppError> {
    let provider = build_provider(config)?;
    let context = repl::context_for(config);
    Ok(
        Orchestrator::new(provider, context, config.agent.keep_alive.clone())
            .with_options(config.options()),
    )
}

fn run_chat(
    config: &Config,
    style: &Style,
    script: &[String],
    no_repl: bool,
) -> Result<i32, AppError> {
    let orchestrator = build_orchestrator(config)?;

    // Preflight: fail early with advice instead of mid-conversation.
    let mut banner = repl::banner_lines(config);
    match orchestrator.preflight() {
        Ok(()) => banner.push(format!("Status   : {}", style.green("Connected"))),
        Err(err) => {
            banner.push(format!("Status   : {}", style.red("Unavailable")));
            let mut session = Session::new(orchestrator, std::io::stdout(), *style)
                .with_banner(banner)
                .with_warnings(config.warnings.clone());
            session.print_banner();
            eprintln!("{} {err}", style.red("error:"));
            eprintln!("       {}", style.dim(err.hint()));
            eprintln!(
                "       {}",
                style.dim(
                    "Run `qwen-agent doctor` for a full check, or `ollama serve` to start Ollama."
                )
            );
            return Ok(exit::FAILURE);
        }
    }

    let mut session = Session::new(orchestrator, std::io::stdout(), *style)
        .with_banner(banner)
        .with_labels(Labels::default())
        .with_warnings(config.warnings.clone());

    if script.is_empty() {
        let mut input = ReaderInput::new(std::io::stdin().lock());
        session.run(&mut input);
    } else {
        let mut lines: Vec<String> = script.to_vec();
        if !no_repl {
            // After the script, continue with piped stdin when there is any.
            use std::io::Read;
            let mut rest = String::new();
            let _ = std::io::stdin().read_to_string(&mut rest);
            lines.extend(rest.lines().map(str::to_string));
        }
        let mut input = ScriptedInput::new(lines);
        session.run(&mut input);
    }
    Ok(exit::OK)
}

fn run_once(
    config: &Config,
    style: &Style,
    prompt: Option<String>,
    read_stdin: bool,
) -> Result<i32, AppError> {
    // `IsTerminal` is sealed in std, so the check happens here and the pure
    // function takes the result: that keeps `assemble_prompt` testable.
    let prompt = assemble_prompt(
        prompt,
        read_stdin,
        std::io::IsTerminal::is_terminal(&std::io::stdin()),
        &mut std::io::stdin(),
    )?;

    let orchestrator = build_orchestrator(config)?;
    // Non-interactive: no banner, prompts or stats — this is script output.
    let mut session = Session::new(orchestrator, std::io::stdout(), *style).with_labels(Labels {
        user: "You".into(),
        assistant: "Qwen".into(),
        interactive: false,
    });
    for warning in &config.warnings {
        logging::warn("config", warning);
    }
    session.handle_line(&prompt);
    let _ = std::io::stdout().flush();

    let failed = session.orchestrator().last_turn().is_none();
    Ok(if failed { exit::FAILURE } else { exit::OK })
}

/// Builds the one-shot prompt from the argument and piped input.
///
/// * `--stdin` (or no prompt) with piped input → the input is the prompt;
/// * a prompt plus piped input → the input is appended as context, which is what
///   `git diff | qwen-agent run "review this change"` needs;
/// * stdin is only read when it is **not** a terminal, so an interactive
///   invocation never hangs waiting for EOF.
fn assemble_prompt<R: std::io::Read>(
    prompt: Option<String>,
    read_stdin: bool,
    stdin_is_terminal: bool,
    stdin: &mut R,
) -> Result<String, AppError> {
    let piped = read_stdin || !stdin_is_terminal;
    let piped_text = if piped {
        let mut buf = String::new();
        stdin.read_to_string(&mut buf).map_err(AppError::Io)?;
        buf.trim_end().to_string()
    } else {
        String::new()
    };

    let prompt = match prompt {
        Some(p) if !p.trim().is_empty() => match piped_text.is_empty() {
            true => p,
            // Cap the injected context so a huge diff cannot crowd out the
            // instructions; the model's context budget is finite.
            false => {
                const MAX_CONTEXT_CHARS: usize = 12_000;
                let mut context = piped_text;
                if context.chars().count() > MAX_CONTEXT_CHARS {
                    let truncated: String = context.chars().take(MAX_CONTEXT_CHARS).collect();
                    context = format!(
                        "{truncated}\n\n[...truncated: {} characters omitted]",
                        context.chars().count() - MAX_CONTEXT_CHARS
                    );
                }
                format!("{p}\n\nInput:\n{context}")
            }
        },
        _ => piped_text,
    };

    if prompt.trim().is_empty() {
        return Err(AppError::Usage(
            "`run` needs a prompt, e.g. `qwen-agent run \"summarise src/\"`, or piped input".into(),
        ));
    }
    Ok(prompt)
}

fn run_models(config: &Config, style: &Style, json: bool) -> Result<i32, AppError> {
    let provider = build_provider(config)?;
    let models = provider.list_models().map_err(AppError::Llm)?;
    if json {
        let value = crate::json::Value::Array(
            models
                .iter()
                .map(|m| {
                    crate::json::Value::object([
                        ("name", crate::json::Value::str(&m.name)),
                        ("size", crate::json::Value::from(m.size_bytes)),
                        (
                            "parameters",
                            m.parameter_size
                                .clone()
                                .map(crate::json::Value::str)
                                .unwrap_or(crate::json::Value::Null),
                        ),
                        (
                            "quantization",
                            m.quantization
                                .clone()
                                .map(crate::json::Value::str)
                                .unwrap_or(crate::json::Value::Null),
                        ),
                        (
                            "active",
                            crate::json::Value::Bool(crate::llm::matches_model(
                                &m.name,
                                &config.llm.model,
                            )),
                        ),
                    ])
                })
                .collect(),
        );
        println!("{}", crate::json::to_string(&value));
        return Ok(exit::OK);
    }

    if models.is_empty() {
        println!("{}", style.yellow("no models installed"));
        println!(
            "{}",
            style.dim(&format!("pull one with: ollama pull {}", config.llm.model))
        );
        return Ok(exit::OK);
    }
    for model in &models {
        let active = crate::llm::matches_model(&model.name, &config.llm.model);
        let marker = if active { style.green("*") } else { " ".into() };
        let mut detail = vec![model.human_size()];
        if let Some(p) = &model.parameter_size {
            detail.push(p.clone());
        }
        if let Some(q) = &model.quantization {
            detail.push(q.clone());
        }
        println!(
            "{marker} {:<28} {}",
            model.name,
            style.dim(&detail.join(" · "))
        );
    }
    Ok(exit::OK)
}

fn run_config<W: Write>(
    config: &Config,
    style: &Style,
    out: &mut W,
    toml: bool,
    init: Option<std::path::PathBuf>,
    check: bool,
    overrides: &Overrides,
) -> Result<i32, AppError> {
    if let Some(path) = init {
        if path.exists() {
            return Err(AppError::Other(format!(
                "{} already exists; refusing to overwrite",
                path.display()
            )));
        }
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(AppError::Io)?;
            }
        }
        std::fs::write(&path, Config::default_toml()).map_err(AppError::Io)?;
        let _ = writeln!(
            out,
            "{}",
            style.green(&format!("ok: wrote {}", path.display()))
        );
        return Ok(exit::OK);
    }

    if check {
        match Config::load(overrides) {
            Ok(cfg) => {
                let _ = writeln!(out, "{}", style.green("ok: configuration is valid"));
                for warning in &cfg.warnings {
                    let _ = writeln!(out, "{} {warning}", style.yellow("warning:"));
                }
                Ok(exit::OK)
            }
            Err(err) => {
                // Report on both streams so `--check` fails visibly in a pipe
                // and in a script.
                let _ = writeln!(out, "{} {err}", style.red("invalid:"));
                eprintln!("{} {err}", style.red("invalid:"));
                Ok(exit::FAILURE)
            }
        }
    } else if toml {
        let _ = write!(out, "{}", render_config_toml(config));
        Ok(exit::OK)
    } else {
        render_config_summary(config, style, out);
        Ok(exit::OK)
    }
}

/// Renders the effective configuration as TOML (with provenance comments).
pub fn render_config_toml(config: &Config) -> String {
    let mut out = String::new();
    out.push_str("# effective configuration for qwen-agent\n");
    out.push_str(&format!(
        "# source: {}\n\n",
        config
            .source
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "<built-in defaults>".into())
    ));
    out.push_str("[agent]\n");
    out.push_str(&format!("name = \"{}\"\n", escape(&config.agent.name)));
    out.push_str(&format!(
        "max_iterations = {}\n",
        config.agent.max_iterations
    ));
    out.push_str(&format!(
        "keep_alive = \"{}\"\n",
        escape(&config.agent.keep_alive)
    ));
    out.push_str(&format!(
        "system_prompt = \"{}\"\n\n",
        escape(&config.agent.system_prompt)
    ));
    out.push_str("[llm]\n");
    out.push_str(&format!(
        "provider = \"{}\"\n",
        escape(&config.llm.provider)
    ));
    out.push_str(&format!(
        "base_url = \"{}\"\n",
        escape(&config.llm.base_url)
    ));
    out.push_str(&format!("model = \"{}\"\n", escape(&config.llm.model)));
    out.push_str(&format!("temperature = {}\n", config.llm.temperature));
    out.push_str(&format!("stream = {}\n", config.llm.stream));
    out.push_str(&format!("timeout_secs = {}\n", config.llm.timeout_secs));
    out.push_str(&format!(
        "num_ctx = {}\n",
        config
            .llm
            .num_ctx
            .map(|n| n.to_string())
            .unwrap_or_else(|| "0".into())
    ));
    out.push_str(&format!(
        "num_predict = {}\n\n",
        config
            .llm
            .num_predict
            .map(|n| n.to_string())
            .unwrap_or_else(|| "-1".into())
    ));
    out.push_str("[logging]\n");
    out.push_str(&format!(
        "level = \"{}\"\n\n",
        config.logging.level.as_str()
    ));
    out.push_str("[workspace]\n");
    out.push_str(&format!(
        "root = \"{}\"\n",
        escape(&config.workspace.root.display().to_string())
    ));
    out
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn render_config_summary<W: Write>(config: &Config, style: &Style, out: &mut W) {
    // A macro instead of a closure: the closure would hold a unique borrow of
    // `out` for its whole lifetime.
    macro_rules! label {
        ($k:expr, $v:expr) => {
            let _ = writeln!(out, "  {:<14} {}", style.dim($k), $v);
        };
    }
    let _ = writeln!(out, "{}", style.bold("Configuration"));
    let source = config
        .source
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "(built-in defaults)".to_string());
    label!("source", source);
    label!("agent", config.agent.name.clone());
    label!("provider", config.llm.provider.clone());
    label!("base_url", config.llm.base_url.clone());
    label!("model", style.bold(&config.llm.model));
    label!("temperature", config.llm.temperature.to_string());
    label!("stream", config.llm.stream.to_string());
    label!("keep_alive", config.agent.keep_alive.clone());
    label!("max_iters", config.agent.max_iterations.to_string());
    label!("log level", config.logging.level.as_str().to_string());
    label!("workspace", config.workspace.root.display().to_string());
    if config.llm.num_ctx.is_some() || config.llm.num_predict.is_some() {
        label!(
            "limits",
            format!(
                "num_ctx={:?} num_predict={:?}",
                config.llm.num_ctx, config.llm.num_predict
            )
        );
    }
    if !config.warnings.is_empty() {
        let _ = writeln!(out, "\n{}", style.yellow("warnings"));
        for warning in &config.warnings {
            let _ = writeln!(out, "  {warning}");
        }
    }
    let _ = writeln!(
        out,
        "\n{}",
        style.dim("Precedence: CLI flags > QWEN_AGENT_* env > config file > built-in defaults.")
    );
}

fn run_doctor(config: &Config, style: &Style, verbose: bool) -> Result<i32, AppError> {
    let provider = build_provider(config).ok();
    let checks = doctor::run(config, provider.as_ref(), verbose);
    print!("{}", doctor::render(&checks, style));
    Ok(if doctor::is_healthy(&checks) {
        exit::OK
    } else {
        exit::FAILURE
    })
}

/// Convenience for `Context` construction in tests and tools.
pub fn default_context(config: &Config) -> Context {
    repl::context_for(config)
}

/// Reads a whole stream into a string, used by `run --stdin`.
pub fn read_all<R: std::io::Read>(mut r: R) -> std::io::Result<String> {
    let mut buf = String::new();
    r.read_to_string(&mut buf)?;
    Ok(buf)
}

/// Buffers stdin for scripted sessions.
pub fn buffered_stdin() -> BufReader<std::io::Stdin> {
    BufReader::new(std::io::stdin())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_config_renders_parseable_toml() {
        let cfg = Config::default();
        let text = render_config_toml(&cfg);
        let reparsed = Config::from_toml_str(&text).expect("round trip must parse");
        assert_eq!(reparsed.llm.model, cfg.llm.model);
        assert_eq!(reparsed.llm.base_url, cfg.llm.base_url);
        assert_eq!(reparsed.agent.max_iterations, cfg.agent.max_iterations);
        assert_eq!(reparsed.logging.level, cfg.logging.level);
    }

    #[test]
    fn config_escaping_handles_quotes_and_newlines() {
        let mut cfg = Config::default();
        cfg.agent.system_prompt = "say \"hi\"\nthen stop".into();
        let text = render_config_toml(&cfg);
        let reparsed = Config::from_toml_str(&text).unwrap();
        assert_eq!(reparsed.agent.system_prompt, "say \"hi\"\nthen stop");
    }

    #[test]
    fn exit_codes_follow_shell_conventions() {
        assert_eq!(exit::OK, 0);
        assert_eq!(exit::FAILURE, 1);
        assert_eq!(exit::USAGE, 2);
    }

    /// Stands in for `stdin` with a known "piped" payload.
    struct Pipe(String);

    impl std::io::Read for Pipe {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let bytes = self.0.as_bytes();
            let n = bytes.len().min(buf.len());
            buf[..n].copy_from_slice(&bytes[..n]);
            self.0.drain(..n);
            Ok(n)
        }
    }

    #[test]
    fn prompt_only_needs_no_stdin() {
        let prompt =
            assemble_prompt(Some("hello".into()), false, false, &mut Pipe(String::new())).unwrap();
        assert_eq!(prompt, "hello");
    }

    #[test]
    fn piped_input_becomes_the_prompt_when_none_is_given() {
        let prompt =
            assemble_prompt(None, false, false, &mut Pipe("from the pipe\n".into())).unwrap();
        assert_eq!(prompt, "from the pipe");
    }

    #[test]
    fn piped_input_is_appended_as_context_to_an_explicit_prompt() {
        let prompt = assemble_prompt(
            Some("review this".into()),
            true,
            false,
            &mut Pipe("diff --git\n".into()),
        )
        .unwrap();
        assert!(prompt.starts_with("review this"), "{prompt}");
        assert!(prompt.contains("Input:"), "{prompt}");
        assert!(prompt.contains("diff --git"), "{prompt}");
    }

    #[test]
    fn huge_piped_context_is_truncated() {
        let big = "x".repeat(20_000);
        let prompt =
            assemble_prompt(Some("summarise".into()), true, false, &mut Pipe(big)).unwrap();
        assert!(prompt.contains("truncated"), "{}", &prompt[..80]);
        assert!(
            prompt.chars().count() < 13_000,
            "{}",
            prompt.chars().count()
        );
    }

    #[test]
    fn empty_input_is_a_usage_error() {
        let err = assemble_prompt(None, false, false, &mut Pipe(String::new())).unwrap_err();
        assert!(matches!(err, AppError::Usage(_)));
        assert!(err.to_string().contains("piped input"), "{err}");

        let err = assemble_prompt(Some("   ".into()), false, false, &mut Pipe(String::new()))
            .unwrap_err();
        assert!(matches!(err, AppError::Usage(_)));
    }

    #[test]
    fn run_once_without_any_input_is_a_usage_error() {
        let cfg = Config::default();
        // In the test harness stdin is not a terminal and is empty.
        let err = run_once(&cfg, &Style::plain(), None, false).unwrap_err();
        assert!(matches!(err, AppError::Usage(_)));
    }

    #[test]
    fn invalid_base_url_surfaces_as_a_provider_error() {
        let mut cfg = Config::default();
        cfg.llm.base_url = "localhost:11434".into();
        assert!(build_provider(&cfg).is_err());
    }
}
