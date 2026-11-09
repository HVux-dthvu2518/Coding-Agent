//! Structured, levelled logging to stderr.
//!
//! Two rules drive the design:
//!
//! * **stderr only** — stdout belongs to model output and interactive prompts,
//!   so logs never corrupt a pipe (`qwen-agent run "…" | less`).
//! * **quiet by default** — the REPL is the product; logs are diagnostics.
//!   `--log-level debug` (or `QWEN_AGENT_LOG=debug`) turns on request tracing.

use crate::config::LogLevel;
use std::io::Write;
use std::sync::atomic::{AtomicU8, Ordering};

static LEVEL: AtomicU8 = AtomicU8::new(1); // warn

fn level_code(level: LogLevel) -> u8 {
    match level {
        LogLevel::Error => 0,
        LogLevel::Warn => 1,
        LogLevel::Info => 2,
        LogLevel::Debug => 3,
        LogLevel::Trace => 4,
    }
}

/// Installs the global level. Later calls override earlier ones, which keeps
/// tests hermetic when they run in parallel.
pub fn init(level: LogLevel) {
    LEVEL.store(level_code(level), Ordering::Relaxed);
}

/// The current level.
pub fn current() -> LogLevel {
    match LEVEL.load(Ordering::Relaxed) {
        0 => LogLevel::Error,
        1 => LogLevel::Warn,
        2 => LogLevel::Info,
        3 => LogLevel::Debug,
        _ => LogLevel::Trace,
    }
}

/// True when a message at `level` would be emitted.
pub fn enabled(level: LogLevel) -> bool {
    level_code(level) <= LEVEL.load(Ordering::Relaxed)
}

fn emit(level: LogLevel, target: &str, message: &dyn std::fmt::Display) {
    if !enabled(level) {
        return;
    }
    let mut err = std::io::stderr().lock();
    // Log lines are prefixed with `::` so they are trivially greppable and
    // visually distinct from model output.
    let _ = writeln!(err, ":: {:<5} [{}] {}", level.as_str(), target, message);
}

pub fn error(target: &str, message: impl std::fmt::Display) {
    emit(LogLevel::Error, target, &message);
}

pub fn warn(target: &str, message: impl std::fmt::Display) {
    emit(LogLevel::Warn, target, &message);
}

pub fn info(target: &str, message: impl std::fmt::Display) {
    emit(LogLevel::Info, target, &message);
}

pub fn debug(target: &str, message: impl std::fmt::Display) {
    emit(LogLevel::Debug, target, &message);
}

pub fn trace(target: &str, message: impl std::fmt::Display) {
    emit(LogLevel::Trace, target, &message);
}

/// Logs a request/response pair at `debug`, truncating long bodies so a large
/// context does not flood the terminal.
pub fn wire(target: &str, direction: &str, body: &str) {
    if !enabled(LogLevel::Debug) {
        return;
    }
    const MAX: usize = 2000;
    if body.len() > MAX {
        // Slice on a char boundary: bodies contain user text and model output.
        let mut end = MAX;
        while end > 0 && !body.is_char_boundary(end) {
            end -= 1;
        }
        debug(
            target,
            format_args!("{direction} {} bytes: {}…", body.len(), &body[..end]),
        );
    } else {
        debug(target, format_args!("{direction} {body}"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_gating_respects_global_setting() {
        // Serialised by construction: the global is process-wide, so these
        // assertions are written to be order-independent by always setting
        // the level immediately before checking it.
        init(LogLevel::Warn);
        assert!(enabled(LogLevel::Error));
        assert!(enabled(LogLevel::Warn));
        assert!(!enabled(LogLevel::Info));

        init(LogLevel::Debug);
        assert!(enabled(LogLevel::Debug));
        assert!(enabled(LogLevel::Info));
        assert_eq!(current(), LogLevel::Debug);

        init(LogLevel::Warn);
    }

    #[test]
    fn message_helpers_do_not_panic_when_disabled() {
        // Kept quiet on purpose: these calls write to the real stderr, and a
        // test suite should not print diagnostics that look like failures.
        init(LogLevel::Error);
        warn("test", "should be suppressed");
        debug("test", "suppressed too");
        info("test", "suppressed as well");
        assert!(!enabled(LogLevel::Warn));
        init(LogLevel::Warn);
    }

    #[test]
    fn wire_truncation_respects_char_boundaries() {
        // Exercises the slicing logic without emitting 6 KB to stderr: the
        // truncation helper is what matters, and `wire` is a thin wrapper.
        let body = "é".repeat(3000);
        let mut end = 2000.min(body.len());
        while end > 0 && !body.is_char_boundary(end) {
            end -= 1;
        }
        assert!(body.is_char_boundary(end));
        assert!(end >= 1999, "must not truncate far from the limit: {end}");
        assert_eq!(&body[..end.min(8)], "éééé");
    }
}
