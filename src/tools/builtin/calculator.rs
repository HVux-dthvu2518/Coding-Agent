//! `calculator.evaluate` — arithmetic on a parsed expression.
//!
//! This is the tool that proves the pipeline without touching the filesystem, so
//! it is also the one that must be *obviously* safe. It never shells out, never
//! `eval`s and never grows a general-purpose scripting language:
//!
//! * a hand-written lexer and recursive-descent parser;
//! * `+ - * / % ^`, unary sign, parentheses, `pi`/`e` constants;
//! * integer arithmetic stays integer (`2/4` is `0.5`, but `6/3` prints `2`);
//! * denominators are checked for zero *before* dividing;
//! * the expression is length-capped and depth-capped, so no input can make the
//!   parser or the evaluator expensive.

use crate::json::Value;
use crate::tools::Tool;
use crate::tools::context::ToolContext;
use crate::tools::definition::{Param, Permission, Schema, ToolDefinition};
use crate::tools::result::{ToolError, ToolResult};
use crate::tools::validation::ValidatedArgs;

/// Longest accepted expression.
pub const MAX_EXPRESSION_LEN: usize = 500;
/// Maximum parser nesting (parentheses / unary chains).
const MAX_DEPTH: usize = 24;

/// The calculator tool.
#[derive(Debug, Default, Clone, Copy)]
pub struct Calculator;

impl Tool for Calculator {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            "calculator.evaluate",
            "Evaluate an arithmetic expression and return the exact result. \
             Use this instead of doing arithmetic yourself. Supports + - * / % ^, \
             parentheses, unary minus, and the constants pi and e.",
            vec![Param::required(
                "expression",
                Schema::string_of_len(
                    "Arithmetic expression, for example `25 * (10 + 5)`. \
                     Use ^ for powers and * for multiplication.",
                    1,
                    MAX_EXPRESSION_LEN,
                ),
            )],
            Permission::ReadOnly,
        )
    }

    fn execute(
        &self,
        args: &ValidatedArgs,
        _context: &ToolContext,
    ) -> Result<ToolResult, ToolError> {
        // Validation guarantees presence and type; absence here would be a
        // definition bug, and is reported rather than panicked.
        let expression = args.str("expression").ok_or_else(|| ToolError::Rejected {
            tool: "calculator.evaluate".into(),
            reason: "missing `expression`".into(),
        })?;

        let value = evaluate(expression).map_err(|reason| ToolError::Rejected {
            tool: "calculator.evaluate".into(),
            reason,
        })?;

        let rendered = render_number(value);
        Ok(ToolResult::success(
            "calculator.evaluate",
            Value::object([
                ("expression", Value::str(expression.trim())),
                ("result", Value::Number(value)),
                ("formatted", Value::str(&rendered)),
            ]),
            format!("{expression} = {rendered}"),
        ))
    }
}

/// Evaluates an arithmetic expression.
pub fn evaluate(input: &str) -> Result<f64, String> {
    if input.trim().is_empty() {
        return Err("expression is empty".into());
    }
    if input.chars().count() > MAX_EXPRESSION_LEN {
        return Err(format!(
            "expression is longer than {MAX_EXPRESSION_LEN} characters"
        ));
    }
    let tokens = lex(input)?;
    // Named constants only make sense as whole operands; `2 apples` must not
    // silently degrade to a syntax error about `apples` being unknown.
    ensure_operands_are_valid(&tokens)?;
    let mut parser = Parser {
        tokens: &tokens,
        pos: 0,
    };
    let value = parser.expression(0)?;
    if let Some(token) = parser.peek() {
        return Err(format!(
            "unexpected `{}` after the expression",
            token.describe()
        ));
    }
    if !value.is_finite() {
        return Err("result is not a finite number".into());
    }
    Ok(value)
}

/// Rejects a name that is not a known constant before parsing, so the message
/// names the actual problem.
fn ensure_operands_are_valid(tokens: &[Token]) -> Result<(), String> {
    for token in tokens {
        if let Token::Identifier(name) = token {
            if name != "pi" && name != "e" {
                return Err(format!("unknown name `{name}` (supported: pi, e)"));
            }
        }
    }
    Ok(())
}

/// Formats a result the way a person expects: no trailing `.0` on integers.
pub fn render_number(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e15 {
        format!("{}", value as i64)
    } else {
        // Trim floating-point noise to a readable precision.
        let mut text = format!("{value:.10}");
        while text.contains('.') && text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
        text
    }
}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Caret,
    LParen,
    RParen,
    Identifier(String),
}

impl Token {
    fn describe(&self) -> String {
        match self {
            Token::Number(n) => render_number(*n),
            Token::Plus => "+".into(),
            Token::Minus => "-".into(),
            Token::Star => "*".into(),
            Token::Slash => "/".into(),
            Token::Percent => "%".into(),
            Token::Caret => "^".into(),
            Token::LParen => "(".into(),
            Token::RParen => ")".into(),
            Token::Identifier(name) => name.clone(),
        }
    }
}

fn lex(input: &str) -> Result<Vec<Token>, String> {
    let chars: Vec<char> = input.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            ' ' | '\t' | '\n' | '\r' => i += 1,
            '+' => {
                tokens.push(Token::Plus);
                i += 1;
            }
            '-' | '\u{2212}' => {
                tokens.push(Token::Minus);
                i += 1;
            }
            '*' | '\u{00d7}' => {
                tokens.push(Token::Star);
                i += 1;
            }
            '/' | '\u{00f7}' => {
                tokens.push(Token::Slash);
                i += 1;
            }
            '%' => {
                tokens.push(Token::Percent);
                i += 1;
            }
            '^' => {
                tokens.push(Token::Caret);
                i += 1;
            }
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            _ if c.is_ascii_digit() || c == '.' => {
                let start = i;
                let mut seen_dot = false;
                let mut seen_digit = false;
                while i < chars.len() {
                    let d = chars[i];
                    if d.is_ascii_digit() {
                        seen_digit = true;
                        i += 1;
                    } else if d == '.' && !seen_dot {
                        seen_dot = true;
                        i += 1;
                    } else {
                        break;
                    }
                }
                if !seen_digit {
                    return Err("malformed number".into());
                }
                // Exponent form: 1e3, 2.5E-2.
                if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                    let mut j = i + 1;
                    if j < chars.len() && (chars[j] == '+' || chars[j] == '-') {
                        j += 1;
                    }
                    if j < chars.len() && chars[j].is_ascii_digit() {
                        while j < chars.len() && chars[j].is_ascii_digit() {
                            j += 1;
                        }
                        i = j;
                    }
                }
                let text: String = chars[start..i].iter().collect();
                let value: f64 = text
                    .parse()
                    .map_err(|_| format!("`{text}` is not a number"))?;
                tokens.push(Token::Number(value));
            }
            _ if c.is_ascii_alphabetic() => {
                let start = i;
                while i < chars.len() && chars[i].is_ascii_alphabetic() {
                    i += 1;
                }
                let name: String = chars[start..i].iter().collect();
                tokens.push(Token::Identifier(name.to_ascii_lowercase()));
            }
            other => return Err(format!("unsupported character `{other}`")),
        }
    }
    if tokens.is_empty() {
        return Err("expression is empty".into());
    }
    Ok(tokens)
}

// ---------------------------------------------------------------------------
// Parser / evaluator
// ---------------------------------------------------------------------------

struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn bump(&mut self) -> Option<&Token> {
        let token = self.tokens.get(self.pos);
        if token.is_some() {
            self.pos += 1;
        }
        token
    }

    /// Precedence climbing: `+ -` < `* / %` < `^` (right-associative).
    fn expression(&mut self, depth: usize) -> Result<f64, String> {
        if depth > MAX_DEPTH {
            return Err("expression is nested too deeply".into());
        }
        let mut left = self.term(depth)?;
        while let Some(token) = self.peek() {
            let op = match token {
                Token::Plus => '+',
                Token::Minus => '-',
                _ => break,
            };
            self.pos += 1;
            let right = self.term(depth)?;
            left = match op {
                '+' => left + right,
                _ => left - right,
            };
            if !left.is_finite() {
                return Err("arithmetic overflow".into());
            }
        }
        Ok(left)
    }

    fn term(&mut self, depth: usize) -> Result<f64, String> {
        let mut left = self.power(depth)?;
        while let Some(token) = self.peek() {
            let op = match token {
                Token::Star => '*',
                Token::Slash => '/',
                Token::Percent => '%',
                _ => break,
            };
            self.pos += 1;
            let right = self.power(depth)?;
            left = match op {
                '*' => left * right,
                '/' => {
                    if right == 0.0 {
                        return Err("division by zero".into());
                    }
                    left / right
                }
                _ => {
                    if right == 0.0 {
                        return Err("modulo by zero".into());
                    }
                    left % right
                }
            };
            if !left.is_finite() {
                return Err("arithmetic overflow".into());
            }
        }
        Ok(left)
    }

    fn power(&mut self, depth: usize) -> Result<f64, String> {
        let base = self.unary(depth)?;
        if matches!(self.peek(), Some(Token::Caret)) {
            self.pos += 1;
            // Right-associative: 2^3^2 == 2^(3^2).
            let exponent = self.power(depth + 1)?;
            let value = base.powf(exponent);
            if !value.is_finite() {
                return Err("arithmetic overflow".into());
            }
            return Ok(value);
        }
        Ok(base)
    }

    fn unary(&mut self, depth: usize) -> Result<f64, String> {
        if depth > MAX_DEPTH {
            return Err("expression is nested too deeply".into());
        }
        match self.bump() {
            Some(Token::Minus) => Ok(-self.unary(depth + 1)?),
            Some(Token::Plus) => self.unary(depth + 1),
            Some(Token::LParen) => {
                let value = self.expression(depth + 1)?;
                match self.bump() {
                    Some(Token::RParen) => Ok(value),
                    _ => Err("missing closing parenthesis".into()),
                }
            }
            Some(Token::Number(n)) => Ok(*n),
            Some(Token::Identifier(name)) => match name.as_str() {
                "pi" => Ok(std::f64::consts::PI),
                "e" => Ok(std::f64::consts::E),
                other => Err(format!("unknown name `{other}` (try pi or e)")),
            },
            Some(token) => Err(format!("unexpected `{}`", token.describe())),
            None => Err("incomplete expression".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json;

    fn run(expression: &str) -> Value {
        let args = crate::tools::validation::validate(
            &Calculator.definition(),
            &json::parse(&format!(
                r#"{{"expression":{}}}"#,
                json::to_string(&Value::str(expression))
            ))
            .unwrap(),
        )
        .unwrap();
        Calculator
            .execute(
                &args,
                &ToolContext::new(".", crate::agent::planner::WorkerRole::Coder),
            )
            .unwrap()
            .data
            .unwrap()
    }

    fn evaluate_ok(expression: &str) -> f64 {
        evaluate(expression).unwrap_or_else(|e| panic!("`{expression}` failed: {e}"))
    }

    #[test]
    fn the_documented_example_works() {
        let data = run("25 * (10 + 5)");
        assert_eq!(data.get("result").unwrap().as_f64(), Some(375.0));
        assert_eq!(data.get("formatted").unwrap().as_str(), Some("375"));
    }

    #[test]
    fn precedence_and_parentheses_are_correct() {
        assert_eq!(evaluate_ok("2 + 3 * 4"), 14.0);
        assert_eq!(evaluate_ok("(2 + 3) * 4"), 20.0);
        assert_eq!(evaluate_ok("2 * 3 + 4 * 5"), 26.0);
        assert_eq!(evaluate_ok("10 - 2 - 3"), 5.0, "left associative");
    }

    #[test]
    fn powers_are_right_associative() {
        assert_eq!(evaluate_ok("2^3"), 8.0);
        assert_eq!(evaluate_ok("2^3^2"), 512.0, "2^(3^2), not (2^3)^2");
        assert_eq!(evaluate_ok("2^-1"), 0.5);
    }

    #[test]
    fn division_and_modulo_work() {
        assert_eq!(evaluate_ok("7 / 2"), 3.5);
        assert_eq!(evaluate_ok("6 / 3"), 2.0);
        assert_eq!(evaluate_ok("7 % 4"), 3.0);
    }

    #[test]
    fn unary_signs_and_double_negation() {
        assert_eq!(evaluate_ok("-5 + 3"), -2.0);
        assert_eq!(evaluate_ok("--5"), 5.0);
        assert_eq!(evaluate_ok("-(2+3)"), -5.0);
        assert_eq!(evaluate_ok("3 * -2"), -6.0);
    }

    #[test]
    fn constants_are_recognised_case_insensitively() {
        assert!((evaluate_ok("pi") - std::f64::consts::PI).abs() < 1e-12);
        assert!((evaluate_ok("E") - std::f64::consts::E).abs() < 1e-12);
        assert!((evaluate_ok("2*pi") - 2.0 * std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn decimals_and_exponent_notation() {
        assert_eq!(evaluate_ok("0.5 + 0.25"), 0.75);
        assert_eq!(evaluate_ok("1e3"), 1000.0);
        assert_eq!(evaluate_ok("2.5e-2"), 0.025);
    }

    #[test]
    fn division_by_zero_is_refused_not_infinite() {
        assert_eq!(evaluate("1/0").unwrap_err(), "division by zero");
        assert_eq!(evaluate("1 % 0").unwrap_err(), "modulo by zero");
        assert_eq!(evaluate("5/(3-3)").unwrap_err(), "division by zero");
    }

    #[test]
    fn malformed_input_is_refused_with_a_reason() {
        assert!(evaluate("").unwrap_err().contains("empty"));
        assert!(evaluate("   ").unwrap_err().contains("empty"));
        assert!(evaluate("2 +").unwrap_err().contains("incomplete"));
        assert!(evaluate("(2 + 3").unwrap_err().contains("closing"));
        assert!(evaluate("2 + 3)").unwrap_err().contains("unexpected"));
        assert!(
            evaluate("2 $ 3")
                .unwrap_err()
                .contains("unsupported character")
        );
        assert!(
            evaluate("2 apples")
                .unwrap_err()
                .contains("unknown name `apples`")
        );
        assert!(
            evaluate("2 apples")
                .unwrap_err()
                .contains("supported: pi, e")
        );
        assert!(evaluate("..").unwrap_err().contains("malformed"));
    }

    #[test]
    fn deeply_nested_input_is_refused_before_it_can_hurt() {
        let nested = format!("{}1{}", "(".repeat(200), ")".repeat(200));
        assert!(evaluate(&nested).unwrap_err().contains("nested too deeply"));
    }

    #[test]
    fn overlong_input_is_refused() {
        let long = "1+".repeat(400) + "1";
        assert!(evaluate(&long).unwrap_err().contains("longer than"));
    }

    #[test]
    fn integer_results_render_without_a_decimal_point() {
        assert_eq!(render_number(375.0), "375");
        assert_eq!(render_number(-2.0), "-2");
        assert_eq!(render_number(0.5), "0.5");
        assert_eq!(render_number(1.0 / 3.0), "0.3333333333");
        assert_eq!(render_number(2.0 * std::f64::consts::PI), "6.2831853072");
    }

    #[test]
    fn result_payload_is_structured_for_the_model() {
        let data = run("6/3");
        assert_eq!(data.get("expression").unwrap().as_str(), Some("6/3"));
        assert_eq!(data.get("result").unwrap().as_f64(), Some(2.0));
        assert_eq!(data.get("formatted").unwrap().as_str(), Some("2"));
    }

    #[test]
    fn the_summary_is_human_readable() {
        let args = crate::tools::validation::validate(
            &Calculator.definition(),
            &json::parse(r#"{"expression":"2+2"}"#).unwrap(),
        )
        .unwrap();
        let result = Calculator
            .execute(
                &args,
                &ToolContext::new(".", crate::agent::planner::WorkerRole::Coder),
            )
            .unwrap();
        assert_eq!(result.summary, "2+2 = 4");
    }

    #[test]
    fn the_definition_declares_a_required_expression() {
        let definition = Calculator.definition();
        assert_eq!(definition.name, "calculator.evaluate");
        assert_eq!(definition.permission, Permission::ReadOnly);
        assert_eq!(definition.required_param_names(), vec!["expression"]);
        assert!(definition.description.contains("arithmetic"));
    }
}
