//! A minimal TOML reader — enough for `config/default.toml`, nothing more.
//!
//! Supported syntax:
//!
//! ```toml
//! # comment
//! top_level_key = "value"
//!
//! [section]
//! string  = "text"      # basic strings with \" \\ \n \t escapes
//! literal = 'text'
//! integer = 10
//! float   = 0.7
//! boolean = true
//! ```
//!
//! Deliberately **not** supported: arrays, inline tables, multi-line strings,
//! dates, dotted keys. Encountering them produces an error naming the line,
//! which is friendlier than silently ignoring a setting the user typed.

use crate::error::TomlError;

/// A scalar TOML value.
#[derive(Debug, Clone, PartialEq)]
pub enum TomlValue {
    Str(String),
    Integer(i64),
    Float(f64),
    Bool(bool),
}

impl TomlValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            TomlValue::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            TomlValue::Integer(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            TomlValue::Float(f) => Some(*f),
            TomlValue::Integer(i) => Some(*i as f64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            TomlValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            TomlValue::Str(_) => "string",
            TomlValue::Integer(_) => "integer",
            TomlValue::Float(_) => "float",
            TomlValue::Bool(_) => "boolean",
        }
    }
}

/// A parsed document: sections (plus the implicit root section `""`) in file
/// order, each holding key/value pairs in file order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Table {
    sections: Vec<(String, Vec<(String, TomlValue)>)>,
}

impl Table {
    /// Looks up `key` inside `section`.
    pub fn get(&self, section: &str, key: &str) -> Option<&TomlValue> {
        self.sections
            .iter()
            .find(|(name, _)| name == section)
            .and_then(|(_, entries)| entries.iter().find(|(k, _)| k == key))
            .map(|(_, v)| v)
    }

    /// Section names in file order, excluding the implicit root.
    pub fn section_names(&self) -> impl Iterator<Item = &str> {
        self.sections
            .iter()
            .filter(|(name, _)| !name.is_empty())
            .map(|(name, _)| name.as_str())
    }

    /// All key/value pairs for a section (empty slice when absent).
    pub fn entries(&self, section: &str) -> &[(String, TomlValue)] {
        self.sections
            .iter()
            .find(|(name, _)| name == section)
            .map(|(_, entries)| entries.as_slice())
            .unwrap_or(&[])
    }

    /// True when the document contains no keys at all.
    pub fn is_empty(&self) -> bool {
        self.sections.iter().all(|(_, entries)| entries.is_empty())
    }

    fn section_mut(&mut self, name: &str) -> &mut Vec<(String, TomlValue)> {
        if let Some(idx) = self.sections.iter().position(|(n, _)| n == name) {
            &mut self.sections[idx].1
        } else {
            self.sections.push((name.to_string(), Vec::new()));
            &mut self.sections.last_mut().expect("just pushed").1
        }
    }
}

/// Parses a TOML document.
pub fn parse(input: &str) -> Result<Table, TomlError> {
    let mut table = Table::default();
    let mut current = String::new();

    for (idx, raw_line) in input.lines().enumerate() {
        let line_no = idx + 1;
        let line = strip_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }

        if line.starts_with('[') {
            if !line.ends_with(']') || line.len() < 2 {
                return Err(err(line_no, "section header is missing a closing `]`"));
            }
            let name = line[1..line.len() - 1].trim();
            if name.is_empty() {
                return Err(err(line_no, "empty section header"));
            }
            if name.contains('[') {
                return Err(err(line_no, "array-of-tables `[[...]]` is not supported"));
            }
            current = validate_section_name(line_no, name)?;
            table.section_mut(&current);
            continue;
        }

        let (key, value_src) = match line.split_once('=') {
            Some(parts) => parts,
            None => {
                return Err(err(
                    line_no,
                    format!("expected `key = value`, found `{line}`"),
                ));
            }
        };
        let key = key.trim();
        if key.is_empty() {
            return Err(err(line_no, "missing key before `=`"));
        }
        if key.contains('.') {
            return Err(err(
                line_no,
                "dotted keys are not supported; use `[section]` headers",
            ));
        }
        let value = parse_value(line_no, value_src.trim())?;
        if table.get(&current, key).is_some() {
            return Err(err(line_no, format!("duplicate key `{key}`")));
        }
        table.section_mut(&current).push((key.to_string(), value));
    }

    Ok(table)
}

fn err(line: usize, message: impl Into<String>) -> TomlError {
    TomlError {
        line,
        message: message.into(),
    }
}

fn validate_section_name(line: usize, name: &str) -> Result<String, TomlError> {
    let ok = name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.');
    if !ok {
        return Err(err(line, format!("invalid section name `{name}`")));
    }
    Ok(name.to_string())
}

/// Removes a trailing `# comment`, respecting quoted strings.
fn strip_comment(line: &str) -> &str {
    let bytes = line.as_bytes();
    let mut in_string: Option<u8> = None;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate() {
        if let Some(quote) = in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' && quote == b'"' {
                escaped = true;
            } else if b == quote {
                in_string = None;
            }
            continue;
        }
        match b {
            b'"' | b'\'' => in_string = Some(b),
            b'#' => return &line[..i],
            _ => {}
        }
    }
    line
}

fn parse_value(line: usize, src: &str) -> Result<TomlValue, TomlError> {
    if src.is_empty() {
        return Err(err(line, "missing value after `=`"));
    }
    match src.as_bytes()[0] {
        b'"' => {
            let s = parse_basic_string(line, src)?;
            Ok(TomlValue::Str(s))
        }
        b'\'' => {
            if src.len() < 2 || !src.ends_with('\'') {
                return Err(err(line, "unterminated literal string"));
            }
            Ok(TomlValue::Str(src[1..src.len() - 1].to_string()))
        }
        b'[' => Err(err(line, "arrays are not supported in this configuration")),
        b'{' => Err(err(
            line,
            "inline tables are not supported in this configuration",
        )),
        _ => {
            let lowered = src.to_ascii_lowercase();
            match lowered.as_str() {
                "true" => return Ok(TomlValue::Bool(true)),
                "false" => return Ok(TomlValue::Bool(false)),
                _ => {}
            }
            if let Ok(i) = src.parse::<i64>() {
                return Ok(TomlValue::Integer(i));
            }
            if let Ok(f) = src.parse::<f64>() {
                if f.is_finite() {
                    return Ok(TomlValue::Float(f));
                }
                return Err(err(line, format!("non-finite number `{src}`")));
            }
            Err(err(
                line,
                format!("unrecognised value `{src}` (quote strings with \"...\")"),
            ))
        }
    }
}

fn parse_basic_string(line: usize, src: &str) -> Result<String, TomlError> {
    let bytes = src.as_bytes();
    let mut out = String::new();
    let mut i = 1;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => {
                if i + 1 == bytes.len() {
                    return Ok(out);
                }
                return Err(err(line, "trailing characters after closing quote"));
            }
            b'\\' => {
                i += 1;
                let esc = *bytes
                    .get(i)
                    .ok_or_else(|| err(line, "unterminated escape in string"))?;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'n' => out.push('\n'),
                    b't' => out.push('\t'),
                    b'r' => out.push('\r'),
                    other => {
                        return Err(err(
                            line,
                            format!("unsupported escape `\\{}`", other as char),
                        ));
                    }
                }
                i += 1;
            }
            _ => {
                // Copy one whole UTF-8 character.
                let start = i;
                i += 1;
                while i < bytes.len() && (bytes[i] & 0b1100_0000) == 0b1000_0000 {
                    i += 1;
                }
                out.push_str(
                    std::str::from_utf8(&bytes[start..i])
                        .map_err(|_| err(line, "invalid UTF-8 in string"))?,
                );
            }
        }
    }
    Err(err(line, "unterminated string"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sections_and_scalars() {
        let src = r#"
# leading comment
name = "root"

[agent]
name = "Qwen Agent"   # trailing comment
max_iterations = 10

[llm]
temperature = 0.7
stream = true
path = 'C:\literal'
"#;
        let t = parse(src).unwrap();
        assert_eq!(t.get("", "name").unwrap().as_str(), Some("root"));
        assert_eq!(t.get("agent", "name").unwrap().as_str(), Some("Qwen Agent"));
        assert_eq!(t.get("agent", "max_iterations").unwrap().as_i64(), Some(10));
        assert_eq!(t.get("llm", "temperature").unwrap().as_f64(), Some(0.7));
        assert_eq!(t.get("llm", "stream").unwrap().as_bool(), Some(true));
        assert_eq!(t.get("llm", "path").unwrap().as_str(), Some("C:\\literal"));
        assert!(t.get("llm", "nope").is_none());
        assert_eq!(t.section_names().count(), 2);
    }

    #[test]
    fn decodes_escapes() {
        let t = parse(r#"s = "a\nb\t\"c\"""#).unwrap();
        assert_eq!(t.get("", "s").unwrap().as_str().unwrap(), "a\nb\t\"c\"");
    }

    #[test]
    fn hash_inside_string_is_not_a_comment() {
        let t = parse(r#"s = "a # b" # real comment"#).unwrap();
        assert_eq!(t.get("", "s").unwrap().as_str(), Some("a # b"));
    }

    #[test]
    fn reports_line_numbers() {
        let e = parse("[a]\nx = \n").unwrap_err();
        assert_eq!(e.line, 2);
        assert!(e.message.contains("missing value"), "{e}");

        let e = parse("x = 1\nx = 2\n").unwrap_err();
        assert!(e.message.contains("duplicate"), "{e}");

        let e = parse("x = [1,2]\n").unwrap_err();
        assert!(e.message.contains("arrays"), "{e}");

        let e = parse("[broken\n").unwrap_err();
        assert!(e.message.contains("closing"), "{e}");
    }

    #[test]
    fn unquoted_words_are_errors() {
        let e = parse("model = qwen3\n").unwrap_err();
        assert!(e.message.contains("unrecognised value"), "{e}");
    }
}
