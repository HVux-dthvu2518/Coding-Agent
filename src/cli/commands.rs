//! REPL slash-commands.
//!
//! Parsing is separated from execution for two reasons: a 4B agent's UX hinges
//! on typos being *cheap* (unknown command → suggestion, not a crashed session),
//! and command handling needs unit tests that do not require a terminal.

/// A parsed slash command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Help,
    Clear,
    Exit,
    Model,
    Models,
    Status,
    Stats,
    History,
    System(Option<String>),
    Config,
    Save(Option<String>),
    Retry,
    Unknown {
        name: String,
        suggestion: Option<String>,
    },
}

/// Result of parsing a line of input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Input {
    /// Normal conversation text.
    Message(String),
    Command(Command),
    /// Whitespace only; do not send anything to the model.
    Empty,
}

/// The canonical command table, used for `/help` and typo suggestions.
pub const COMMANDS: &[(&str, &str)] = &[
    ("/help", "show this list"),
    ("/exit", "leave (also /quit, Ctrl-D)"),
    ("/clear", "forget the conversation, keep the system prompt"),
    ("/model", "show the active model"),
    ("/models", "list models installed in Ollama"),
    ("/status", "check the server and model"),
    ("/stats", "tokens, speed and context use of the last turn"),
    ("/history", "print the conversation so far"),
    (
        "/system",
        "show the system prompt; `/system <text>` replaces it",
    ),
    ("/config", "print the effective configuration"),
    ("/save", "write the conversation to a markdown file"),
    ("/retry", "re-send the previous message"),
];

/// Parses one line of REPL input.
pub fn parse(line: &str) -> Input {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Input::Empty;
    }
    if !trimmed.starts_with('/') {
        return Input::Message(trimmed.to_string());
    }

    let (verb, rest) = match trimmed.split_once(char::is_whitespace) {
        Some((v, r)) => (v, r.trim()),
        None => (trimmed, ""),
    };
    let name = verb.to_ascii_lowercase();

    let command = match name.as_str() {
        "/help" | "/h" | "/?" => Command::Help,
        "/exit" | "/quit" | "/q" => Command::Exit,
        "/clear" | "/reset" => Command::Clear,
        "/model" => Command::Model,
        "/models" => Command::Models,
        "/status" => Command::Status,
        "/stats" => Command::Stats,
        "/history" | "/log" => Command::History,
        "/config" => Command::Config,
        "/retry" => Command::Retry,
        "/save" => Command::Save(if rest.is_empty() {
            None
        } else {
            Some(rest.to_string())
        }),
        "/system" => Command::System(if rest.is_empty() {
            None
        } else {
            Some(rest.to_string())
        }),
        _ => Command::Unknown {
            suggestion: suggest(&name),
            name: verb.to_string(),
        },
    };
    Input::Command(command)
}

/// True when the input begins with `/` (so the REPL can decide about
/// readline-style editing hints).
pub fn is_command_name(line: &str) -> bool {
    line.trim_start().starts_with('/')
}

/// Levenshtein distance, used for "did you mean" suggestions.
pub fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

fn suggest(name: &str) -> Option<String> {
    let mut best: Option<(usize, &str)> = None;
    for (candidate, _) in COMMANDS {
        let d = edit_distance(name, candidate);
        if best.map(|(bd, _)| d < bd).unwrap_or(true) {
            best = Some((d, candidate));
        }
    }
    // Only suggest when the typo is close; otherwise stay quiet.
    match best {
        Some((d, candidate)) if d <= 2 => Some(candidate.to_string()),
        _ => None,
    }
}

/// The `/help` body, pre-formatted for the given width.
pub fn help_text() -> String {
    let mut out = String::from("Commands\n");
    for (name, desc) in COMMANDS {
        out.push_str(&format!("  {name:<10} {desc}\n"));
    }
    out.push_str("\nAnything else is sent to the model.");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_is_a_message() {
        assert_eq!(parse("hello there"), Input::Message("hello there".into()));
        assert_eq!(parse("  spaced  "), Input::Message("spaced".into()));
        // A lone '/' is not a command name we know, but it is command-shaped.
        assert!(matches!(
            parse("/"),
            Input::Command(Command::Unknown { .. })
        ));
        // Text containing a slash mid-sentence is still a message.
        assert_eq!(
            parse("path is src/main.rs"),
            Input::Message("path is src/main.rs".into())
        );
    }

    #[test]
    fn empty_input_is_not_sent() {
        assert_eq!(parse(""), Input::Empty);
        assert_eq!(parse("   \n\t "), Input::Empty);
    }

    #[test]
    fn commands_parse_case_insensitively_with_aliases() {
        assert_eq!(parse("/HELP"), Input::Command(Command::Help));
        assert_eq!(parse("/h"), Input::Command(Command::Help));
        assert_eq!(parse("/quit"), Input::Command(Command::Exit));
        assert_eq!(parse("/reset"), Input::Command(Command::Clear));
        assert_eq!(parse("/log"), Input::Command(Command::History));
    }

    #[test]
    fn commands_with_arguments_keep_the_rest() {
        assert_eq!(
            parse("/system be terse"),
            Input::Command(Command::System(Some("be terse".into())))
        );
        assert_eq!(parse("/system"), Input::Command(Command::System(None)));
        assert_eq!(
            parse("/save notes.md"),
            Input::Command(Command::Save(Some("notes.md".into())))
        );
    }

    #[test]
    fn typos_get_a_suggestion() {
        match parse("/hlep") {
            Input::Command(Command::Unknown { suggestion, name }) => {
                assert_eq!(name, "/hlep");
                assert_eq!(suggestion.as_deref(), Some("/help"));
            }
            other => panic!("expected unknown command, got {other:?}"),
        }
    }

    #[test]
    fn wild_input_gets_no_suggestion() {
        match parse("/xyzzyplugh") {
            Input::Command(Command::Unknown { suggestion, .. }) => assert!(suggestion.is_none()),
            other => panic!("expected unknown command, got {other:?}"),
        }
    }

    #[test]
    fn edit_distance_is_symmetric_and_correct() {
        assert_eq!(edit_distance("kitten", "sitting"), 3);
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("abc", ""), 3);
        assert_eq!(edit_distance("same", "same"), 0);
        assert_eq!(edit_distance("ab", "ba"), 2);
    }

    #[test]
    fn help_lists_every_command() {
        let help = help_text();
        for (name, _) in COMMANDS {
            assert!(help.contains(name), "help is missing {name}");
        }
    }

    #[test]
    fn command_detection_helper() {
        assert!(is_command_name("/help"));
        assert!(is_command_name("   /x"));
        assert!(!is_command_name("help"));
    }
}
