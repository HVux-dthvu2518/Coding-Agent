//! Terminal styling and markdown-lite rendering.
//!
//! Colour is opt-out aware in three ways, because this binary runs in terminals
//! the author will never see:
//!
//! * `--color=never` / `always` / `auto` (clap-style, also `NO_COLOR`, `TERM=dumb`);
//! * output is not a TTY → no colour and no cursor control;
//! * width is capped and detected from `COLUMNS`, so long answers wrap sanely in
//!   narrow panes instead of being mangled by the terminal.
//!
//! The markdown renderer is intentionally tiny: headings, fenced code,
//! inline code, bold and bullets. It makes streamed 4B output readable without
//! pulling in a parser that would dominate the dependency budget.

use std::fmt::Write as _;
use std::io::IsTerminal;

/// Whether styling is enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMode {
    Auto,
    Always,
    Never,
}

impl ColorMode {
    pub fn parse(s: &str) -> Option<ColorMode> {
        match s {
            "auto" => Some(ColorMode::Auto),
            "always" | "yes" | "true" => Some(ColorMode::Always),
            "never" | "no" | "false" => Some(ColorMode::Never),
            _ => None,
        }
    }
}

/// Resolves the mode against the environment.
pub fn use_color(mode: ColorMode) -> bool {
    match mode {
        ColorMode::Always => true,
        ColorMode::Never => false,
        ColorMode::Auto => {
            if std::env::var_os("NO_COLOR").is_some() {
                return false;
            }
            if let Ok(term) = std::env::var("TERM") {
                if term == "dumb" {
                    return false;
                }
            }
            std::io::stdout().is_terminal()
        }
    }
}

/// ANSI styling helpers. All are no-ops when `enabled` is false.
#[derive(Debug, Clone, Copy)]
pub struct Style {
    pub enabled: bool,
}

impl Style {
    pub fn new(enabled: bool) -> Style {
        Style { enabled }
    }

    /// Style chosen for a pipe: no escape sequences.
    pub fn plain() -> Style {
        Style { enabled: false }
    }

    pub fn paint(&self, code: &str, text: &str) -> String {
        if self.enabled && !text.is_empty() {
            format!("\x1b[{code}m{text}\x1b[0m")
        } else {
            text.to_string()
        }
    }

    pub fn bold(&self, t: &str) -> String {
        self.paint("1", t)
    }
    pub fn dim(&self, t: &str) -> String {
        self.paint("2", t)
    }
    pub fn italic(&self, t: &str) -> String {
        self.paint("3", t)
    }
    pub fn cyan(&self, t: &str) -> String {
        self.paint("36", t)
    }
    pub fn green(&self, t: &str) -> String {
        self.paint("32", t)
    }
    pub fn yellow(&self, t: &str) -> String {
        self.paint("33", t)
    }
    pub fn red(&self, t: &str) -> String {
        self.paint("31", t)
    }
    pub fn magenta(&self, t: &str) -> String {
        self.paint("35", t)
    }
    pub fn blue(&self, t: &str) -> String {
        self.paint("34", t)
    }

    /// Inverse video, used for the status line.
    pub fn inverse(&self, t: &str) -> String {
        self.paint("7", t)
    }
}

/// Terminal width in columns, clamped to something readable.
pub fn term_width() -> usize {
    let from_env = std::env::var("COLUMNS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok());
    let width = from_env.unwrap_or(80);
    width.clamp(40, 120)
}

/// Visible width of a string, ignoring ANSI escape sequences.
pub fn visible_len(s: &str) -> usize {
    let mut len = 0;
    let mut in_escape = false;
    for c in s.chars() {
        if in_escape {
            if c.is_ascii_alphabetic() {
                in_escape = false;
            }
        } else if c == '\x1b' {
            in_escape = true;
        } else {
            len += 1;
        }
    }
    len
}

/// A box drawing helper for banners and notices.
pub fn boxed(title: &str, lines: &[String], style: &Style, width: usize) -> String {
    let inner = width.saturating_sub(4).max(20);
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{}",
        style.dim(&format!("╭{}╮", "─".repeat(width.saturating_sub(2))))
    );
    let pad = (inner.saturating_sub(title.chars().count())) / 2;
    let _ = writeln!(
        out,
        "{} {}{}{} {}",
        style.dim("│"),
        " ".repeat(pad),
        style.bold(title),
        " ".repeat(inner.saturating_sub(pad + title.chars().count())),
        style.dim("│")
    );
    for line in lines {
        let visible = visible_len(line);
        let padding = inner.saturating_sub(visible);
        let _ = writeln!(
            out,
            "{} {}{} {}",
            style.dim("│"),
            line,
            " ".repeat(padding),
            style.dim("│")
        );
    }
    let _ = writeln!(
        out,
        "{}",
        style.dim(&format!("╰{}╯", "─".repeat(width.saturating_sub(2))))
    );
    out
}

/// Wraps `text` to `width`, breaking on spaces and preserving existing newlines.
pub fn wrap(text: &str, width: usize) -> Vec<String> {
    let width = width.max(10);
    let mut lines = Vec::new();
    for raw in text.split('\n') {
        if raw.chars().count() <= width {
            lines.push(raw.to_string());
            continue;
        }
        let mut current = String::new();
        for word in raw.split(' ') {
            let word_len = word.chars().count();
            if current.is_empty() {
                if word_len > width {
                    // Hard-break a very long token (URLs, base64 blobs).
                    let mut chunk = String::new();
                    for c in word.chars() {
                        if chunk.chars().count() == width {
                            lines.push(std::mem::take(&mut chunk));
                        }
                        chunk.push(c);
                    }
                    current = chunk;
                } else {
                    current.push_str(word);
                }
            } else if current.chars().count() + 1 + word_len <= width {
                current.push(' ');
                current.push_str(word);
            } else {
                lines.push(std::mem::take(&mut current));
                current.push_str(word);
            }
        }
        if !current.is_empty() {
            lines.push(current);
        }
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

/// Renders markdown-lite with ANSI styling. Returns the text unchanged when
/// styling is disabled, so piped output stays machine-friendly.
pub fn render_markdown(text: &str, style: &Style) -> String {
    if !style.enabled {
        return text.to_string();
    }
    let mut out = String::new();
    let mut in_code = false;
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            in_code = !in_code;
            out.push_str(&style.dim(line.trim_end()));
            out.push('\n');
            continue;
        }
        if in_code {
            out.push_str(&style.green(line.trim_end()));
            out.push('\n');
            continue;
        }
        let (prefix, rest) = split_heading(line);
        if let Some((level, body)) = prefix {
            let title = inline_styles(body.trim_end(), style);
            out.push_str(&style.bold(&match level {
                1 => style.cyan(&title),
                2 => style.cyan(&title),
                _ => title,
            }));
            out.push('\n');
            continue;
        }
        out.push_str(&inline_styles(rest, style));
    }
    out
}

fn split_heading(line: &str) -> (Option<(usize, &str)>, &str) {
    let trimmed = line.trim_start_matches(' ');
    let hashes = trimmed.chars().take_while(|c| *c == '#').count();
    if (1..=6).contains(&hashes) {
        let rest = &trimmed[hashes..];
        if rest.starts_with(' ') {
            return (Some((hashes, rest)), line);
        }
    }
    (None, line)
}

/// Applies bold, italic, inline-code and bullet styling to one line.
fn inline_styles(line: &str, style: &Style) -> String {
    let mut out = String::new();
    let mut buf = String::new();

    let indent: String = line.chars().take_while(|c| *c == ' ').collect();
    let is_bullet =
        line[indent.len()..].starts_with("- ") || line[indent.len()..].starts_with("* ");
    let skip = if is_bullet {
        out.push_str(&indent);
        out.push_str(&style.cyan("• "));
        indent.chars().count() + 2
    } else {
        0
    };

    let mut chars = line.chars().skip(skip).peekable();
    while let Some(c) = chars.next() {
        match c {
            '`' => {
                // Inline code: style up to the closing backtick.
                let mut code = String::new();
                while let Some(&n) = chars.peek() {
                    if n == '`' {
                        chars.next();
                        break;
                    }
                    code.push(n);
                    chars.next();
                }
                out.push_str(&style.green(&code));
            }
            '*' | '_' if chars.peek() == Some(&c) => {
                chars.next();
                let mut bold = String::new();
                let mut closed = false;
                while let Some(n) = chars.next() {
                    if n == c && chars.peek() == Some(&c) {
                        chars.next();
                        closed = true;
                        break;
                    }
                    bold.push(n);
                }
                if closed {
                    out.push_str(&style.bold(&bold));
                } else {
                    // Unterminated marker: leave the text as typed.
                    buf.push(c);
                    buf.push(c);
                    buf.push_str(&bold);
                }
            }
            _ => buf.push(c),
        }
    }
    out.push_str(&buf);
    out
}

/// Prints the "You >" style prompt without a trailing newline, flushing stderr so
/// it appears before input blocking.
pub fn prompt(style: &Style, label: &str) -> String {
    format!("{} ", style.bold(&style.cyan(&format!("{label} >"))))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on() -> Style {
        Style::new(true)
    }

    #[test]
    fn color_mode_parsing() {
        assert_eq!(ColorMode::parse("auto"), Some(ColorMode::Auto));
        assert_eq!(ColorMode::parse("always"), Some(ColorMode::Always));
        assert_eq!(ColorMode::parse("never"), Some(ColorMode::Never));
        assert_eq!(ColorMode::parse("maybe"), None);
    }

    #[test]
    fn plain_style_emits_no_escapes() {
        let s = Style::plain();
        assert_eq!(s.bold("x"), "x");
        assert_eq!(s.red("problem"), "problem");
        assert!(on().bold("x").contains("\x1b[1m"));
    }

    #[test]
    fn visible_len_ignores_ansi() {
        let styled = on().bold("hello");
        assert_eq!(visible_len(&styled), 5);
        assert_eq!(visible_len("plain"), 5);
    }

    #[test]
    fn wrap_breaks_on_spaces_and_respects_width() {
        let lines = wrap("the quick brown fox jumps over the lazy dog", 20);
        assert!(lines.iter().all(|l| l.chars().count() <= 20), "{lines:?}");
        assert!(lines.len() > 1);
        assert_eq!(
            lines.join(" "),
            "the quick brown fox jumps over the lazy dog"
        );
    }

    #[test]
    fn wrap_preserves_explicit_newlines_and_blank_lines() {
        let lines = wrap("a\n\nb", 40);
        assert_eq!(lines, vec!["a", "", "b"]);
    }

    #[test]
    fn wrap_hard_breaks_long_tokens() {
        let long = "x".repeat(50);
        let lines = wrap(&long, 20);
        assert!(lines.iter().all(|l| l.chars().count() <= 20));
        assert_eq!(lines.concat(), long);
    }

    #[test]
    fn markdown_headings_and_code_are_styled() {
        let rendered = render_markdown("# Title\nbody\n```\ncode\n```\n", &on());
        assert!(rendered.contains("\x1b[1m"));
        assert!(rendered.contains("\x1b[36m"), "headings are cyan");
        assert!(rendered.contains("\x1b[32m"), "code is green");
        // Text content survives styling.
        let stripped: String = rendered.chars().filter(|c| *c != '\x1b').collect();
        assert!(stripped.contains("Title"));
        assert!(stripped.contains("body"));
    }

    #[test]
    fn markdown_is_a_noop_without_color() {
        let src = "# Title\n`code` **bold**\n";
        assert_eq!(render_markdown(src, &Style::plain()), src);
    }

    #[test]
    fn inline_code_and_bold_are_replaced_by_styled_text() {
        let out = render_markdown("use `Vec<T>` and **careful** editing", &on());
        assert!(out.contains("Vec<T>"));
        assert!(!out.contains('`'), "backticks must be consumed");
        assert!(!out.contains("**"));
        assert!(out.contains("\x1b[32m"));
    }

    #[test]
    fn unterminated_markers_do_not_eat_the_text() {
        let out = render_markdown("a *b c", &on());
        assert!(out.contains('*'));
        assert!(out.contains("b c"));
        let out = render_markdown("a `unclosed", &on());
        assert!(out.contains("unclosed"));
    }

    #[test]
    fn bullets_are_styled() {
        let out = render_markdown("- one\n  - nested\n", &on());
        assert!(out.contains('•'));
        assert!(out.contains("nested"));
    }

    #[test]
    fn boxed_draws_a_frame() {
        let text = boxed(
            "Qwen Agent",
            &["Model: qwen3.5:4b".to_string()],
            &Style::plain(),
            40,
        );
        assert!(text.starts_with('╭'));
        assert!(text.trim_end().ends_with('╯'));
        assert!(text.contains("Qwen Agent"));
        assert!(text.contains("Model: qwen3.5:4b"));
        // Lines are padded to the same visible width.
        let widths: Vec<usize> = text.lines().map(visible_len).collect();
        assert!(widths.iter().all(|w| *w == widths[0]), "{widths:?}");
    }

    #[test]
    fn term_width_is_clamped() {
        // Cannot safely mutate env in parallel tests; assert the clamp bounds.
        let w = term_width();
        assert!((40..=120).contains(&w));
    }
}
