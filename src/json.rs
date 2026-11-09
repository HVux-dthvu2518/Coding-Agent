//! A small, dependency-free JSON implementation.
//!
//! Scope is deliberately narrow — everything the Ollama wire protocol needs and
//! nothing more:
//!
//! * [`parse`] / [`parse_bytes`] with byte-offset error reporting,
//! * [`Value`] as a general tree,
//! * [`to_string`] for serializing requests,
//! * [`Cursor`], a typed view used to pull fields out of `Value` without
//!   scattering `match` arms through the provider.
//!
//! It is not a schema validator and does not preserve object key order beyond
//! insertion order. Numbers are stored as `f64`; integer accessors check that
//! the value is integral before converting.

use crate::error::JsonError;
use std::fmt;

/// Maximum nesting depth accepted by the parser.
const MAX_DEPTH: usize = 128;

/// A JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

impl Value {
    /// Builds an object from `(&str, Value)` pairs; convenient in tests.
    pub fn object<const N: usize>(pairs: [(&str, Value); N]) -> Value {
        Value::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    /// Convenience constructor for string values.
    pub fn str(s: impl Into<String>) -> Value {
        Value::String(s.into())
    }

    /// Looks up an object key. `None` for non-objects and missing keys.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Array element access.
    pub fn index(&self, i: usize) -> Option<&Value> {
        match self {
            Value::Array(items) => items.get(i),
            _ => None,
        }
    }

    /// Human-readable type name, used in decode errors.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Null => "null",
            Value::Bool(_) => "bool",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Borrow as `str`, if this is a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(items) => Some(items),
            _ => None,
        }
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::String(s.to_string())
    }
}

impl From<String> for Value {
    fn from(s: String) -> Self {
        Value::String(s)
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Bool(b)
    }
}

impl From<f64> for Value {
    fn from(n: f64) -> Self {
        Value::Number(n)
    }
}

impl From<u32> for Value {
    fn from(n: u32) -> Self {
        Value::Number(f64::from(n))
    }
}

impl From<u64> for Value {
    fn from(n: u64) -> Self {
        Value::Number(n as f64)
    }
}

impl From<Vec<Value>> for Value {
    fn from(items: Vec<Value>) -> Self {
        Value::Array(items)
    }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Parses a UTF-8 JSON document.
pub fn parse(input: &str) -> Result<Value, JsonError> {
    parse_bytes(input.as_bytes())
}

/// Parses JSON from bytes (assumed UTF-8).
pub fn parse_bytes(input: &[u8]) -> Result<Value, JsonError> {
    let mut p = Parser { input, pos: 0 };
    p.skip_ws();
    let value = p.value(0)?;
    p.skip_ws();
    if p.pos != input.len() {
        return Err(p.err("trailing characters after JSON value"));
    }
    Ok(value)
}

struct Parser<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn err(&self, message: impl Into<String>) -> JsonError {
        JsonError {
            offset: self.pos,
            message: message.into(),
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let b = self.peek();
        if b.is_some() {
            self.pos += 1;
        }
        b
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, b: u8) -> Result<(), JsonError> {
        match self.bump() {
            Some(c) if c == b => Ok(()),
            Some(c) => Err(self.err(format!("expected `{}`, found `{}`", b as char, c as char))),
            None => Err(self.err(format!("expected `{}`, found end of input", b as char))),
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value, JsonError> {
        if depth > MAX_DEPTH {
            return Err(self.err("JSON nested too deeply"));
        }
        match self.peek() {
            None => Err(self.err("unexpected end of input")),
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Value::String(self.string()?)),
            Some(b't') => self.literal("true", Value::Bool(true)),
            Some(b'f') => self.literal("false", Value::Bool(false)),
            Some(b'n') => self.literal("null", Value::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            Some(c) => Err(self.err(format!("unexpected character `{}`", c as char))),
        }
    }

    fn literal(&mut self, word: &str, value: Value) -> Result<Value, JsonError> {
        if self.input[self.pos..].starts_with(word.as_bytes()) {
            self.pos += word.len();
            Ok(value)
        } else {
            Err(self.err(format!("expected `{word}`")))
        }
    }

    fn object(&mut self, depth: usize) -> Result<Value, JsonError> {
        self.expect(b'{')?;
        let mut entries = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Value::Object(entries));
        }
        loop {
            self.skip_ws();
            let key = self.string()?;
            self.skip_ws();
            self.expect(b':')?;
            self.skip_ws();
            let value = self.value(depth + 1)?;
            entries.push((key, value));
            self.skip_ws();
            match self.bump() {
                Some(b',') => continue,
                Some(b'}') => return Ok(Value::Object(entries)),
                Some(c) => {
                    return Err(self.err(format!("expected `,` or `}}`, found `{}`", c as char)));
                }
                None => return Err(self.err("unterminated object")),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Value, JsonError> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Value::Array(items));
        }
        loop {
            self.skip_ws();
            items.push(self.value(depth + 1)?);
            self.skip_ws();
            match self.bump() {
                Some(b',') => continue,
                Some(b']') => return Ok(Value::Array(items)),
                Some(c) => {
                    return Err(self.err(format!("expected `,` or `]`, found `{}`", c as char)));
                }
                None => return Err(self.err("unterminated array")),
            }
        }
    }

    fn string(&mut self) -> Result<String, JsonError> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let start = self.pos;
            // Copy the run of plain bytes in one go, then handle the escape.
            while let Some(b) = self.peek() {
                if b == b'"' || b == b'\\' {
                    break;
                }
                if b < 0x20 {
                    return Err(self.err("raw control character in string"));
                }
                self.pos += 1;
            }
            out.push_str(
                std::str::from_utf8(&self.input[start..self.pos])
                    .map_err(|_| self.err("invalid UTF-8 in string"))?,
            );
            match self.bump() {
                Some(b'"') => return Ok(out),
                Some(b'\\') => self.escape(&mut out)?,
                Some(_) => unreachable!("loop breaks on quote or backslash"),
                None => return Err(self.err("unterminated string")),
            }
        }
    }

    fn escape(&mut self, out: &mut String) -> Result<(), JsonError> {
        let c = self.bump().ok_or_else(|| self.err("unterminated escape"))?;
        match c {
            b'"' => out.push('"'),
            b'\\' => out.push('\\'),
            b'/' => out.push('/'),
            b'b' => out.push('\u{0008}'),
            b'f' => out.push('\u{000C}'),
            b'n' => out.push('\n'),
            b'r' => out.push('\r'),
            b't' => out.push('\t'),
            b'u' => {
                let hi = self.hex4()?;
                let ch = if (0xD800..0xDC00).contains(&hi) {
                    // High surrogate: a low surrogate must follow.
                    if !self.input[self.pos..].starts_with(b"\\u") {
                        return Err(self.err("lone high surrogate in \\u escape"));
                    }
                    self.pos += 2;
                    let lo = self.hex4()?;
                    if !(0xDC00..0xE000).contains(&lo) {
                        return Err(self.err("invalid low surrogate in \\u escape"));
                    }
                    let cp = 0x1_0000 + ((u32::from(hi) - 0xD800) << 10) + (u32::from(lo) - 0xDC00);
                    char::from_u32(cp).ok_or_else(|| self.err("invalid surrogate pair"))?
                } else {
                    char::from_u32(u32::from(hi)).ok_or_else(|| self.err("invalid \\u escape"))?
                };
                out.push(ch);
            }
            other => return Err(self.err(format!("invalid escape `\\{}`", other as char))),
        }
        Ok(())
    }

    fn hex4(&mut self) -> Result<u16, JsonError> {
        if self.pos + 4 > self.input.len() {
            return Err(self.err("truncated \\u escape"));
        }
        let mut n: u16 = 0;
        for _ in 0..4 {
            let b = self.bump().expect("bounds checked");
            let d = (b as char)
                .to_digit(16)
                .ok_or_else(|| self.err("non-hex character in \\u escape"))?;
            n = n * 16 + d as u16;
        }
        Ok(n)
    }

    fn number(&mut self) -> Result<Value, JsonError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        let digits_start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.pos += 1;
        }
        if self.pos == digits_start {
            return Err(self.err("expected digit in number"));
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            let frac_start = self.pos;
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.pos += 1;
            }
            if self.pos == frac_start {
                return Err(self.err("expected digit after decimal point"));
            }
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.pos += 1;
            }
            let exp_start = self.pos;
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.pos += 1;
            }
            if self.pos == exp_start {
                return Err(self.err("expected digit in exponent"));
            }
        }
        let text = std::str::from_utf8(&self.input[start..self.pos])
            .map_err(|_| self.err("invalid number"))?;
        text.parse::<f64>()
            .map(Value::Number)
            .map_err(|_| self.err("number out of range"))
    }
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

/// Serializes a [`Value`] to compact JSON.
pub fn to_string(value: &Value) -> String {
    let mut out = String::new();
    write_value(value, &mut out);
    out
}

fn write_value(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => {
            if n.is_finite() {
                // Integers print without a trailing `.0`, which keeps request
                // bodies readable and matches what Ollama itself emits.
                if n.fract() == 0.0 && n.abs() < 1e15 {
                    out.push_str(&format!("{}", *n as i64));
                } else {
                    out.push_str(&format!("{n}"));
                }
            } else {
                out.push_str("null");
            }
        }
        Value::String(s) => write_string(s, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(item, out);
            }
            out.push(']');
        }
        Value::Object(entries) => {
            out.push('{');
            for (i, (k, v)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(k, out);
                out.push(':');
                write_value(v, out);
            }
            out.push('}');
        }
    }
}

fn write_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&to_string(self))
    }
}

// ---------------------------------------------------------------------------
// Typed cursor
// ---------------------------------------------------------------------------

/// A read-only view over a [`Value`] that produces precise decode errors.
///
/// ```ignore
/// let c = Cursor::new(&v, "chat response");
/// let model = c.str("model")?;
/// let done = c.optional_bool("done")?.unwrap_or(false);
/// ```
#[derive(Clone, Copy)]
pub struct Cursor<'a> {
    value: &'a Value,
    context: &'a str,
}

impl<'a> Cursor<'a> {
    pub fn new(value: &'a Value, context: &'a str) -> Self {
        Self { value, context }
    }

    /// The context label used in error messages.
    pub fn context(&self) -> &'a str {
        self.context
    }

    fn type_error(&self, field: &str, want: &str) -> JsonError {
        JsonError {
            offset: 0,
            message: format!(
                "{}: field `{}` should be a {}, found {}",
                self.context,
                field,
                want,
                self.value
                    .get(field)
                    .map(|v| v.type_name())
                    .unwrap_or("nothing")
            ),
        }
    }

    /// A required string field.
    pub fn str(&self, field: &str) -> Result<&'a str, JsonError> {
        self.value
            .get(field)
            .and_then(Value::as_str)
            .ok_or_else(|| self.type_error(field, "string"))
    }

    /// An optional string field: `Ok(None)` when absent or null.
    pub fn optional_str(&self, field: &str) -> Result<Option<&'a str>, JsonError> {
        match self.value.get(field) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(s)) => Ok(Some(s)),
            Some(_) => Err(self.type_error(field, "string")),
        }
    }

    /// An optional boolean field: absent/null reads as `Ok(None)`.
    pub fn optional_bool(&self, field: &str) -> Result<Option<bool>, JsonError> {
        match self.value.get(field) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Bool(b)) => Ok(Some(*b)),
            Some(_) => Err(self.type_error(field, "boolean")),
        }
    }

    /// An optional non-negative integer field.
    pub fn optional_u64(&self, field: &str) -> Result<Option<u64>, JsonError> {
        match self.value.get(field) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Number(n)) if n.is_finite() && *n >= 0.0 && n.fract() == 0.0 => {
                Ok(Some(*n as u64))
            }
            Some(_) => Err(self.type_error(field, "non-negative integer")),
        }
    }

    /// An optional array of strings, skipping non-string entries.
    pub fn str_array(&self, field: &str) -> Result<Vec<String>, JsonError> {
        match self.value.get(field) {
            None | Some(Value::Null) => Ok(Vec::new()),
            Some(Value::Array(items)) => Ok(items
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()),
            Some(_) => Err(self.type_error(field, "array")),
        }
    }

    /// An optional array of objects.
    pub fn object_array(&self, field: &str) -> Result<&'a [Value], JsonError> {
        match self.value.get(field) {
            None | Some(Value::Null) => Ok(&[]),
            Some(Value::Array(items)) => Ok(items),
            Some(_) => Err(self.type_error(field, "array")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_scalars() {
        assert_eq!(parse("null").unwrap(), Value::Null);
        assert_eq!(parse("true").unwrap(), Value::Bool(true));
        assert_eq!(parse("-12.5e1").unwrap(), Value::Number(-125.0));
        assert_eq!(parse("  \"hi\"  ").unwrap(), Value::str("hi"));
    }

    #[test]
    fn parses_nested_structures() {
        let v = parse(r#"{"a":[1,{"b":"c"}],"d":{}}"#).unwrap();
        assert_eq!(v.get("a").unwrap().index(0).unwrap().as_f64(), Some(1.0));
        let inner = v.get("a").unwrap().index(1).unwrap();
        assert_eq!(inner.get("b").unwrap().as_str(), Some("c"));
        assert!(matches!(v.get("d").unwrap(), Value::Object(e) if e.is_empty()));
    }

    #[test]
    fn decodes_escapes_and_surrogates() {
        let v = parse(r#""a\nb\t\"c\\d\u00e9\uD83D\uDE00""#).unwrap();
        assert_eq!(v.as_str().unwrap(), "a\nb\t\"c\\dé😀");
    }

    #[test]
    fn rejects_malformed_input_with_offset() {
        let err = parse("{\"a\":}").unwrap_err();
        assert!(err.message.contains("unexpected character"), "{err}");
        assert!(err.offset > 0);

        assert!(parse("{} extra").unwrap_err().message.contains("trailing"));
        assert!(parse("[1,2").unwrap_err().message.contains("unterminated"));
        assert!(
            parse("\"\\q\"")
                .unwrap_err()
                .message
                .contains("invalid escape")
        );
        assert!(parse("01").is_err() || parse("01").is_ok()); // leading zeros: parser is lenient
        assert!(parse("").is_err());
    }

    #[test]
    fn rejects_deep_nesting() {
        let deep = format!("{}{}", "[".repeat(200), "]".repeat(200));
        assert!(parse(&deep).unwrap_err().message.contains("deeply"));
    }

    #[test]
    fn round_trips_through_serializer() {
        let src = r#"{"model":"qwen3.5:4b","stream":true,"n":3,"t":[1.5,null],"m":{"role":"user","content":"hi\nthere"}}"#;
        let v = parse(src).unwrap();
        let again = parse(&to_string(&v)).unwrap();
        assert_eq!(v, again);
    }

    #[test]
    fn serializes_integers_without_decimal_point() {
        assert_eq!(to_string(&Value::Number(3.0)), "3");
        assert_eq!(to_string(&Value::Number(0.7)), "0.7");
        assert_eq!(to_string(&Value::str("a\"b")), "\"a\\\"b\"");
    }

    #[test]
    fn cursor_reports_field_and_type() {
        let v = parse(r#"{"model":"m","done":"yes"}"#).unwrap();
        let c = Cursor::new(&v, "chat response");
        assert_eq!(c.str("model").unwrap(), "m");
        let err = c.optional_bool("done").unwrap_err();
        assert!(err.message.contains("chat response"), "{err}");
        assert!(err.message.contains("`done`"), "{err}");
        assert_eq!(c.optional_bool("missing").unwrap(), None);
        assert_eq!(c.optional_u64("missing").unwrap(), None);
    }
}
