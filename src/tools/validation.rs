//! Argument validation against a [`ToolDefinition`].
//!
//! The M1 brief warns against an incomplete JSON Schema engine, and it is right:
//! a validator that silently accepts what it does not understand is worse than
//! no validator, because it launders malformed model output into trusted calls.
//!
//! So this module implements a **closed subset** — string (with length and enum),
//! integer (with bounds), number (with bounds), boolean — and refuses everything
//! else at definition time. What it accepts, it validates strictly:
//!
//! * unknown arguments are rejected (matches `additionalProperties: false`);
//! * missing required arguments are rejected;
//! * types are checked without coercion (`"5"` is not an integer);
//! * integers must be integral — `5.5` and `1e30` are refused;
//! * every rejection produces a message the model can act on.

use crate::json::Value;
use crate::tools::definition::{Schema, ToolDefinition};

/// Validates `arguments` against `definition`.
///
/// Returns the parsed values on success. Validation is separate from execution
/// so the registry can advertise exactly what it enforces.
pub fn validate(definition: &ToolDefinition, arguments: &Value) -> Result<ValidatedArgs, String> {
    let object = match arguments {
        Value::Object(entries) => entries,
        Value::Null => {
            // Some models send no arguments; treat as an empty object only if
            // nothing is required.
            if let Some(missing) = first_missing(definition, &[]) {
                return Err(format!("missing required argument `{missing}`"));
            }
            return Ok(ValidatedArgs { values: Vec::new() });
        }
        other => {
            return Err(format!(
                "arguments must be a JSON object, got {}",
                other.type_name()
            ));
        }
    };

    let mut values: Vec<(String, Value)> = Vec::new();

    for (name, value) in object {
        let param = definition.param(name).ok_or_else(|| {
            let known: Vec<&str> = definition.params.iter().map(|p| p.name.as_str()).collect();
            if known.is_empty() {
                format!(
                    "`{name}` is not an argument of `{}` (it takes none)",
                    definition.name
                )
            } else {
                format!(
                    "`{name}` is not an argument of `{}` (expected: {})",
                    definition.name,
                    known.join(", ")
                )
            }
        })?;
        check_schema(&param.schema, value)
            .map_err(|reason| format!("argument `{name}` {reason}"))?;
        values.push((name.clone(), value.clone()));
    }

    for param in &definition.params {
        if !param.required {
            continue;
        }
        if !values.iter().any(|(name, _)| name == &param.name) {
            return Err(format!(
                "missing required argument `{}` ({})",
                param.name,
                param.schema.description()
            ));
        }
    }

    Ok(ValidatedArgs { values })
}

fn first_missing(definition: &ToolDefinition, present: &[&str]) -> Option<String> {
    definition
        .params
        .iter()
        .filter(|p| p.required)
        .find(|p| !present.contains(&p.name.as_str()))
        .map(|p| p.name.clone())
}

/// Type- and range-checks one value against one schema.
fn check_schema(schema: &Schema, value: &Value) -> Result<(), String> {
    match (schema, value) {
        (
            Schema::String {
                min_len,
                max_len,
                allowed,
                ..
            },
            Value::String(text),
        ) => {
            let len = text.chars().count();
            if let Some(min) = min_len {
                if len < *min {
                    return Err(format!("must be at least {min} character(s) long"));
                }
            }
            if let Some(max) = max_len {
                if len > *max {
                    return Err(format!("must be at most {max} character(s) long"));
                }
            }
            if let Some(allowed) = allowed {
                if !allowed.iter().any(|a| a == text) {
                    return Err(format!("must be one of: {}", allowed.join(", ")));
                }
            }
            Ok(())
        }
        (Schema::Integer { min, max, .. }, Value::Number(n)) => {
            if !n.is_finite() {
                return Err("must be a finite integer".into());
            }
            if n.fract() != 0.0 {
                return Err(format!("must be an integer, got {n}"));
            }
            // Guard the cast: 1e30 is integral by `fract` but not representable.
            if n.abs() > 9.007_199_254_740_992e15 {
                return Err("is too large to be an integer".into());
            }
            let v = *n as i64;
            if let Some(min) = min {
                if v < *min {
                    return Err(format!("must be >= {min}, got {v}"));
                }
            }
            if let Some(max) = max {
                if v > *max {
                    return Err(format!("must be <= {max}, got {v}"));
                }
            }
            Ok(())
        }
        (Schema::Number { min, max, .. }, Value::Number(n)) => {
            if !n.is_finite() {
                return Err("must be a finite number".into());
            }
            if let Some(min) = min {
                if n < min {
                    return Err(format!("must be >= {min}, got {n}"));
                }
            }
            if let Some(max) = max {
                if n > max {
                    return Err(format!("must be <= {max}, got {n}"));
                }
            }
            Ok(())
        }
        (Schema::Boolean { .. }, Value::Bool(_)) => Ok(()),
        (schema, value) => Err(format!(
            "must be {}, got {}",
            schema.type_name(),
            value.type_name()
        )),
    }
}

/// Arguments that passed validation, with typed accessors.
///
/// Accessors cannot fail: validation already proved the type, so a lookup miss
/// means "not supplied" and an accessor mismatch would be a bug in a tool's own
/// definition. Tools therefore contain no argument parsing noise.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ValidatedArgs {
    values: Vec<(String, Value)>,
}

impl ValidatedArgs {
    pub fn new(values: Vec<(String, Value)>) -> Self {
        ValidatedArgs { values }
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.values.iter().find(|(k, _)| k == name).map(|(_, v)| v)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    pub fn str(&self, name: &str) -> Option<&str> {
        self.get(name).and_then(Value::as_str)
    }

    pub fn integer(&self, name: &str) -> Option<i64> {
        self.get(name)
            .and_then(Value::as_f64)
            .filter(|n| n.is_finite())
            .map(|n| n as i64)
    }

    pub fn boolean(&self, name: &str) -> Option<bool> {
        self.get(name).and_then(Value::as_bool)
    }

    /// Names of the arguments supplied, for logging.
    pub fn names(&self) -> Vec<&str> {
        self.values.iter().map(|(k, _)| k.as_str()).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::definition::{Param, Permission, Schema, ToolDefinition};

    fn def() -> ToolDefinition {
        ToolDefinition::new(
            "workspace.list_files",
            "List files.",
            vec![
                Param::optional("path", Schema::string_of_len("Directory.", 1, 200)),
                Param::optional(
                    "depth",
                    Schema::Integer {
                        description: "Levels.".to_string(),
                        min: Some(1),
                        max: Some(3),
                    },
                ),
                Param::optional("all", Schema::boolean("Include hidden.")),
            ],
            Permission::WorkspaceRead,
        )
    }

    fn parse(text: &str) -> Value {
        crate::json::parse(text).unwrap()
    }

    #[test]
    fn empty_arguments_are_valid_when_nothing_is_required() {
        let validated = validate(&def(), &parse("{}")).unwrap();
        assert!(validated.is_empty());
        assert!(!validated.contains("path"));
    }

    #[test]
    fn null_arguments_are_treated_as_empty() {
        assert!(validate(&def(), &Value::Null).is_ok());
    }

    #[test]
    fn valid_arguments_are_exposed_with_types() {
        let validated = validate(&def(), &parse(r#"{"path":"src","depth":2,"all":true}"#)).unwrap();
        assert_eq!(validated.str("path"), Some("src"));
        assert_eq!(validated.integer("depth"), Some(2));
        assert_eq!(validated.boolean("all"), Some(true));
        assert_eq!(validated.names(), vec!["path", "depth", "all"]);
    }

    #[test]
    fn missing_required_arguments_are_rejected_by_name() {
        let required = ToolDefinition::new(
            "calculator.evaluate",
            "Calc.",
            vec![Param::required("expression", Schema::string("Expr."))],
            Permission::ReadOnly,
        );
        let err = validate(&required, &parse("{}")).unwrap_err();
        assert!(
            err.contains("missing required argument `expression`"),
            "{err}"
        );
        // The description is included so the model knows what to supply.
        assert!(err.contains("Expr."), "{err}");

        // Also for a null payload.
        let err = validate(&required, &Value::Null).unwrap_err();
        assert!(err.contains("expression"), "{err}");
    }

    #[test]
    fn unknown_arguments_are_rejected_with_the_expected_list() {
        let err = validate(&def(), &parse(r#"{"pth":"src"}"#)).unwrap_err();
        assert!(err.contains("`pth` is not an argument"), "{err}");
        assert!(err.contains("path"), "{err}");

        let required = ToolDefinition::new(
            "calculator.evaluate",
            "Calc.",
            vec![Param::required("expression", Schema::string("Expr."))],
            Permission::ReadOnly,
        );
        let err = validate(&required, &parse(r#"{"expr":"1+1"}"#)).unwrap_err();
        assert!(err.contains("expected: expression"), "{err}");
    }

    #[test]
    fn zero_argument_tools_reject_any_argument() {
        let time = ToolDefinition::new("system.get_time", "Clock.", vec![], Permission::ReadOnly);
        assert!(validate(&time, &parse("{}")).is_ok());
        let err = validate(&time, &parse(r#"{"tz":"UTC"}"#)).unwrap_err();
        assert!(err.contains("takes none"), "{err}");
    }

    #[test]
    fn wrong_types_are_rejected_without_coercion() {
        let err = validate(&def(), &parse(r#"{"depth":"2"}"#)).unwrap_err();
        assert!(err.contains("must be integer, got string"), "{err}");

        let err = validate(&def(), &parse(r#"{"depth":true}"#)).unwrap_err();
        assert!(err.contains("must be integer, got bool"), "{err}");

        let err = validate(&def(), &parse(r#"{"path":42}"#)).unwrap_err();
        assert!(err.contains("must be string, got number"), "{err}");

        let err = validate(&def(), &parse(r#"{"all":"yes"}"#)).unwrap_err();
        assert!(err.contains("must be boolean, got string"), "{err}");
    }

    #[test]
    fn non_integral_numbers_are_not_integers() {
        let err = validate(&def(), &parse(r#"{"depth":2.5}"#)).unwrap_err();
        assert!(err.contains("must be an integer"), "{err}");

        // 2.0 is integral and acceptable.
        let validated = validate(&def(), &parse(r#"{"depth":2.0}"#)).unwrap();
        assert_eq!(validated.integer("depth"), Some(2));
    }

    #[test]
    fn integer_bounds_are_enforced() {
        let err = validate(&def(), &parse(r#"{"depth":0}"#)).unwrap_err();
        assert!(err.contains("must be >= 1"), "{err}");
        let err = validate(&def(), &parse(r#"{"depth":9}"#)).unwrap_err();
        assert!(err.contains("must be <= 3"), "{err}");
    }

    #[test]
    fn huge_numbers_are_refused_rather_than_wrapping() {
        let err = validate(&def(), &parse(r#"{"depth":1e30}"#)).unwrap_err();
        assert!(
            err.contains("too large") || err.contains("must be <="),
            "{err}"
        );
    }

    #[test]
    fn string_length_bounds_are_enforced() {
        let err = validate(&def(), &parse(r#"{"path":""}"#)).unwrap_err();
        assert!(err.contains("at least 1 character"), "{err}");
        let long = "a".repeat(201);
        let err = validate(&def(), &parse(&format!(r#"{{"path":"{long}"}}"#))).unwrap_err();
        assert!(err.contains("at most 200"), "{err}");
    }

    #[test]
    fn enum_values_are_enforced() {
        let sorted = ToolDefinition::new(
            "workspace.list_files",
            "List.",
            vec![Param::optional(
                "sort",
                Schema::enum_of("Order.", &["name", "size"]),
            )],
            Permission::WorkspaceRead,
        );
        assert!(validate(&sorted, &parse(r#"{"sort":"size"}"#)).is_ok());
        let err = validate(&sorted, &parse(r#"{"sort":"date"}"#)).unwrap_err();
        assert!(err.contains("must be one of: name, size"), "{err}");
    }

    #[test]
    fn array_arguments_are_not_supported_and_say_so() {
        let err = validate(&def(), &parse(r#"{"path":["a"]}"#)).unwrap_err();
        assert!(err.contains("must be string, got array"), "{err}");
    }

    #[test]
    fn top_level_must_be_an_object() {
        let err = validate(&def(), &parse("[1,2]")).unwrap_err();
        assert!(err.contains("must be a JSON object, got array"), "{err}");
        let err = validate(&def(), &parse(r#""text""#)).unwrap_err();
        assert!(err.contains("got string"), "{err}");
    }
}
