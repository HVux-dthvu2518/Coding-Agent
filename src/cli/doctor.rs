//! `qwen-agent doctor` — a checklist of everything that has to be true for the
//! agent to work, in the order it usually breaks.
//!
//! Checks are **independent**: a failure in one still allows the rest to run, so
//! a user sees all problems at once instead of one per attempt. The function
//! returns structured [`Check`]s; rendering lives in [`render`].

use crate::config::Config;
use crate::llm::OllamaProvider;
use crate::llm::provider::LlmProvider;

/// Severity of a check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Pass,
    Warn,
    Fail,
    Skip,
}

impl Status {
    pub fn marker(self) -> &'static str {
        match self {
            Status::Pass => "PASS",
            Status::Warn => "WARN",
            Status::Fail => "FAIL",
            Status::Skip => "SKIP",
        }
    }
}

/// One environment check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Check {
    pub name: String,
    pub status: Status,
    pub detail: String,
    pub hint: Option<String>,
}

impl Check {
    fn pass(name: impl Into<String>, detail: impl Into<String>) -> Check {
        Check {
            name: name.into(),
            status: Status::Pass,
            detail: detail.into(),
            hint: None,
        }
    }
    fn warn(name: impl Into<String>, detail: impl Into<String>, hint: impl Into<String>) -> Check {
        Check {
            name: name.into(),
            status: Status::Warn,
            detail: detail.into(),
            hint: Some(hint.into()),
        }
    }
    fn fail(name: impl Into<String>, detail: impl Into<String>, hint: impl Into<String>) -> Check {
        Check {
            name: name.into(),
            status: Status::Fail,
            detail: detail.into(),
            hint: Some(hint.into()),
        }
    }
    fn skip(name: impl Into<String>, detail: impl Into<String>) -> Check {
        Check {
            name: name.into(),
            status: Status::Skip,
            detail: detail.into(),
            hint: None,
        }
    }
}

/// Runs all checks. `provider` is `None` when the endpoint URL itself is
/// invalid, in which case network checks are skipped with an explanatory line.
pub fn run(cfg: &Config, provider: Option<&OllamaProvider>, verbose: bool) -> Vec<Check> {
    let mut checks = Vec::new();

    checks.push(match &cfg.source {
        Some(path) => Check::pass("configuration", format!("loaded {}", path.display())),
        None => Check::warn(
            "configuration",
            "using built-in defaults (no config file found)",
            "run `qwen-agent config --init` to create one",
        ),
    });

    if cfg.warnings.is_empty() {
        checks.push(Check::pass("config warnings", "none"));
    } else {
        for warning in &cfg.warnings {
            checks.push(Check::warn(
                "config warnings",
                warning.clone(),
                "fix or remove the key",
            ));
        }
    }

    checks.push(Check::pass(
        "settings",
        format!(
            "provider={} model={} temperature={} stream={}",
            cfg.llm.provider, cfg.llm.model, cfg.llm.temperature, cfg.llm.stream
        ),
    ));

    let workspace = &cfg.workspace.root;
    checks.push(if workspace.is_dir() {
        match workspace.canonicalize() {
            Ok(abs) => Check::pass("workspace", abs.display().to_string()),
            Err(_) => Check::pass("workspace", workspace.display().to_string()),
        }
    } else {
        Check::fail(
            "workspace",
            format!("{} is not a directory", workspace.display()),
            "create it or fix `workspace.root`",
        )
    });

    let Some(provider) = provider else {
        checks.push(Check::skip(
            "ollama",
            format!("endpoint `{}` is not a valid http:// URL", cfg.llm.base_url),
        ));
        checks.push(Check::skip("model", "not checked"));
        return checks;
    };

    match provider.check_connection() {
        Err(err) => {
            checks.push(Check::fail(
                "ollama",
                format!("{} ({err})", cfg.llm.base_url),
                "start it with `ollama serve`",
            ));
            checks.push(Check::skip("model", "server unreachable"));
            return checks;
        }
        Ok(version) => checks.push(Check::pass(
            "ollama",
            format!("{} · version {version}", cfg.llm.base_url),
        )),
    }

    match provider.list_models_verbose() {
        Err(err) => {
            checks.push(Check::fail(
                "model",
                err.to_string(),
                "check the Ollama logs",
            ));
            return checks;
        }
        Ok(models) => {
            if models.is_empty() {
                checks.push(Check::fail(
                    "model",
                    "no models installed",
                    format!("run `ollama pull {}`", cfg.llm.model),
                ));
                return checks;
            }
            let found = models
                .iter()
                .find(|m| crate::llm::matches_model(&m.name, &cfg.llm.model));
            match found {
                None => {
                    let available: Vec<&str> =
                        models.iter().take(3).map(|m| m.name.as_str()).collect();
                    checks.push(Check::fail(
                        "model",
                        format!(
                            "`{}` is not installed (have: {}{})",
                            cfg.llm.model,
                            available.join(", "),
                            if models.len() > 3 { ", …" } else { "" }
                        ),
                        format!("run `ollama pull {}`", cfg.llm.model),
                    ));
                }
                Some(model) => {
                    let mut detail = format!("{} present · {}", model.name, model.human_size());
                    if let Some(size) = &model.parameter_size {
                        detail.push_str(&format!(" · {size}"));
                    }
                    if let Some(quant) = &model.quantization {
                        detail.push_str(&format!(" · {quant}"));
                    }
                    checks.push(Check::pass("model", detail));
                }
            }
            checks.push(Check::pass(
                "models installed",
                format!("{} total", models.len()),
            ));
        }
    }

    if verbose {
        match provider.show(&cfg.llm.model) {
            Err(err) => checks.push(Check::warn(
                "capabilities",
                format!("could not read model metadata: {err}"),
                "this is informational; /api/show may be disabled",
            )),
            Ok(spec) => match spec.supports_tools() {
                Some(true) => checks.push(Check::pass(
                    "capabilities",
                    format!("tools supported ({})", spec.capabilities.join(", ")),
                )),
                Some(false) => checks.push(Check::warn(
                    "capabilities",
                    format!(
                        "no tool support advertised ({})",
                        spec.capabilities.join(", ")
                    ),
                    "M1 tool calling will not work with this model; keep M0 conversational",
                )),
                None => checks.push(Check::warn(
                    "capabilities",
                    "server did not report capabilities",
                    "upgrade Ollama to a version that reports `/api/show` capabilities",
                )),
            },
        }
    }

    checks
}

/// True when nothing failed (warnings are acceptable).
pub fn is_healthy(checks: &[Check]) -> bool {
    !checks.iter().any(|c| c.status == Status::Fail)
}

/// Renders checks for the terminal.
pub fn render(checks: &[Check], style: &crate::cli::ui::Style) -> String {
    let mut out = String::new();
    for check in checks {
        let marker = match check.status {
            Status::Pass => style.green("✔"),
            Status::Warn => style.yellow("!"),
            Status::Fail => style.red("✘"),
            Status::Skip => style.dim("–"),
        };
        out.push_str(&format!("{marker} {:<18} {}\n", check.name, check.detail));
        if let Some(hint) = &check.hint {
            out.push_str(&format!(
                "  {:<18} {}\n",
                "",
                style.dim(&format!("→ {hint}"))
            ));
        }
    }
    let failures = checks.iter().filter(|c| c.status == Status::Fail).count();
    let warnings = checks.iter().filter(|c| c.status == Status::Warn).count();
    out.push('\n');
    if failures == 0 {
        out.push_str(&format!(
            "{}\n",
            style.green(&format!(
                "ready — {} check(s) passed, {warnings} warning(s)",
                checks.len() - failures - warnings
            ))
        ));
    } else {
        out.push_str(&format!(
            "{}\n",
            style.red(&format!(
                "{failures} problem(s) must be fixed before chatting"
            ))
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::ui::Style;

    #[test]
    fn status_markers_are_distinct() {
        assert_eq!(Status::Pass.marker(), "PASS");
        assert_ne!(Status::Pass.marker(), Status::Fail.marker());
    }

    #[test]
    fn health_ignores_warnings_but_not_failures() {
        let warn = Check::warn("a", "b", "c");
        let fail = Check::fail("x", "y", "z");
        assert!(is_healthy(std::slice::from_ref(&warn)));
        assert!(!is_healthy(&[warn, fail]));
    }

    #[test]
    fn render_marks_each_status_and_summarises() {
        let checks = vec![
            Check::pass("ollama", "http://localhost:11434 · version 0.9"),
            Check::warn("capabilities", "not reported", "upgrade"),
            Check::fail("model", "missing", "ollama pull qwen3.5:4b"),
            Check::skip("x", "skipped"),
        ];
        let text = render(&checks, &Style::plain());
        assert!(text.contains("ollama"));
        assert!(text.contains("missing"));
        assert!(text.contains("→ ollama pull qwen3.5:4b"));
        assert!(text.contains("1 problem(s) must be fixed"));

        let ok = render(&[Check::pass("a", "b")], &Style::plain());
        assert!(ok.contains("ready"));
    }

    #[test]
    fn unreachable_server_reports_the_fix() {
        // Point at a port nothing listens on; connect is refused immediately.
        let cfg = Config::default();
        let provider = OllamaProvider::new(&crate::llm::OllamaConfig {
            base_url: "http://127.0.0.1:1".into(),
            model: cfg.llm.model.clone(),
        })
        .unwrap();
        let checks = run(&cfg, Some(&provider), false);
        let ollama = checks.iter().find(|c| c.name == "ollama").unwrap();
        assert_eq!(ollama.status, Status::Fail);
        assert!(ollama.hint.as_deref().unwrap().contains("ollama serve"));
        assert!(!is_healthy(&checks));
        // The model check is skipped, not failed, when the server is down.
        let model = checks.iter().find(|c| c.name == "model").unwrap();
        assert_eq!(model.status, Status::Skip);
    }

    #[test]
    fn invalid_endpoint_skips_network_checks() {
        let cfg = Config::default();
        let checks = run(&cfg, None, false);
        let ollama = checks.iter().find(|c| c.name == "ollama").unwrap();
        assert_eq!(ollama.status, Status::Skip);
        assert!(ollama.detail.contains("not a valid"));
    }

    #[test]
    fn missing_workspace_fails_before_network_checks() {
        let mut cfg = Config::default();
        cfg.workspace.root = std::path::PathBuf::from("/definitely/not/here");
        let checks = run(&cfg, None, false);
        let workspace = checks.iter().find(|c| c.name == "workspace").unwrap();
        assert_eq!(workspace.status, Status::Fail);
    }

    #[test]
    fn config_warnings_become_doctor_warnings() {
        let mut cfg = Config::default();
        cfg.warnings
            .push("unknown key `llm.modle` (ignored)".into());
        let checks = run(&cfg, None, false);
        assert!(
            checks
                .iter()
                .any(|c| c.name == "config warnings" && c.status == Status::Warn)
        );
    }

    #[test]
    fn no_config_file_is_a_warning_not_a_failure() {
        let cfg = Config::default(); // source: None
        let checks = run(&cfg, None, false);
        let config = checks.iter().find(|c| c.name == "configuration").unwrap();
        assert_eq!(config.status, Status::Warn);
        assert!(config.hint.as_deref().unwrap().contains("--init"));
    }
}
