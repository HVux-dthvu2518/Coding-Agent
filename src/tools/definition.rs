//! Tool metadata: the single source of truth for both what the model is told
//! and what the executor enforces.
//!
//! The registry serializes [`ToolDefinition`] into the `tools` field of a chat
//! request and validates arguments against the *same* struct. That is the
//! anti-drift rule from the M1 brief: an advertised schema that differs from the
//! validated schema is a security hole, not a cosmetic bug.

use crate::json::Value;

/// How dangerous a tool is, used for policy and for what the user is told.
///
/// Ordering is meaningful: `ReadOnly < Mutating`. A worker that may not mutate
/// is refused anything above `ReadOnly`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Permission {
    /// No side effects outside the process (calculation, clock, listing).
    ReadOnly,
    /// Reads workspace content; can leak data but changes nothing.
    WorkspaceRead,
    /// Writes to the workspace.
    WorkspaceWrite,
    /// Runs arbitrary commands.
    Exec,
}

impl Permission {
    pub fn as_str(self) -> &'static str {
        match self {
            Permission::ReadOnly => "read-only",
            Permission::WorkspaceRead => "workspace-read",
            Permission::WorkspaceWrite => "workspace-write",
            Permission::Exec => "exec",
        }
    }

    /// True when executing this is safe without asking the user.
    pub fn is_silent(self) -> bool {
        matches!(self, Permission::ReadOnly | Permission::WorkspaceRead)
    }
}

/// JSON-schema subset used by tool parameters.
///
/// Deliberately closed: every variant here is fully implemented by
/// [`crate::tools::validation`]. Anything richer (combinators, `$ref`,
/// `additionalProperties` objects, `patternProperties`) is *rejected at
/// construction time* rather than silently ignored — an unvalidated argument is
/// worse than no argument.
#[derive(Debug, Clone, PartialEq)]
pub enum Schema {
    String {
        description: String,
        min_len: Option<usize>,
        max_len: Option<usize>,
        /// Closed set of accepted values.
        allowed: Option<Vec<String>>,
    },
    Integer {
        description: String,
        min: Option<i64>,
        max: Option<i64>,
    },
    Number {
        description: String,
        min: Option<f64>,
        max: Option<f64>,
    },
    Boolean {
        description: String,
    },
}

impl Schema {
    /// Convenience constructor: a plain string.
    pub fn string(description: impl Into<String>) -> Schema {
        Schema::String {
            description: description.into(),
            min_len: None,
            max_len: None,
            allowed: None,
        }
    }

    /// A string restricted to a fixed set of values.
    pub fn enum_of(description: impl Into<String>, allowed: &[&str]) -> Schema {
        Schema::String {
            description: description.into(),
            min_len: None,
            max_len: None,
            allowed: Some(allowed.iter().map(|s| (*s).to_string()).collect()),
        }
    }

    /// A string with length bounds (used to cap expressions and paths).
    pub fn string_of_len(description: impl Into<String>, min_len: usize, max_len: usize) -> Schema {
        Schema::String {
            description: description.into(),
            min_len: Some(min_len),
            max_len: Some(max_len),
            allowed: None,
        }
    }

    pub fn integer(description: impl Into<String>) -> Schema {
        Schema::Integer {
            description: description.into(),
            min: None,
            max: None,
        }
    }

    pub fn number(description: impl Into<String>) -> Schema {
        Schema::Number {
            description: description.into(),
            min: None,
            max: None,
        }
    }

    pub fn boolean(description: impl Into<String>) -> Schema {
        Schema::Boolean {
            description: description.into(),
        }
    }

    pub fn description(&self) -> &str {
        match self {
            Schema::String { description, .. }
            | Schema::Integer { description, .. }
            | Schema::Number { description, .. }
            | Schema::Boolean { description } => description,
        }
    }

    /// The JSON Schema `type` keyword for this variant.
    pub fn type_name(&self) -> &'static str {
        match self {
            Schema::String { .. } => "string",
            Schema::Integer { .. } => "integer",
            Schema::Number { .. } => "number",
            Schema::Boolean { .. } => "boolean",
        }
    }

    /// Serializes to the JSON Schema fragment Ollama's `tools` field expects.
    pub fn to_json(&self) -> Value {
        let mut entries = vec![("type".to_string(), Value::str(self.type_name()))];
        match self {
            Schema::String {
                description,
                min_len,
                max_len,
                allowed,
            } => {
                entries.push(("description".to_string(), Value::str(description)));
                if let Some(n) = min_len {
                    entries.push(("minLength".to_string(), Value::Number(*n as f64)));
                }
                if let Some(n) = max_len {
                    entries.push(("maxLength".to_string(), Value::Number(*n as f64)));
                }
                if let Some(values) = allowed {
                    entries.push((
                        "enum".to_string(),
                        Value::Array(values.iter().map(Value::str).collect()),
                    ));
                }
            }
            Schema::Integer {
                description,
                min,
                max,
            } => {
                entries.push(("description".to_string(), Value::str(description)));
                if let Some(n) = min {
                    entries.push(("minimum".to_string(), Value::Number(*n as f64)));
                }
                if let Some(n) = max {
                    entries.push(("maximum".to_string(), Value::Number(*n as f64)));
                }
            }
            Schema::Number {
                description,
                min,
                max,
            } => {
                entries.push(("description".to_string(), Value::str(description)));
                if let Some(n) = min {
                    entries.push(("minimum".to_string(), Value::Number(*n)));
                }
                if let Some(n) = max {
                    entries.push(("maximum".to_string(), Value::Number(*n)));
                }
            }
            Schema::Boolean { description } => {
                entries.push(("description".to_string(), Value::str(description)));
            }
        }
        Value::Object(entries)
    }
}

/// A named, typed tool parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub schema: Schema,
    pub required: bool,
}

impl Param {
    pub fn required(name: impl Into<String>, schema: Schema) -> Param {
        Param {
            name: name.into(),
            schema,
            required: true,
        }
    }

    pub fn optional(name: impl Into<String>, schema: Schema) -> Param {
        Param {
            name: name.into(),
            schema,
            required: false,
        }
    }
}

/// Everything the model is told about a tool, and everything the validator uses.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDefinition {
    /// Namespaced, lowercase: `calculator.evaluate`.
    pub name: String,
    /// Shown to the model — this is prompt engineering, treat it as such.
    pub description: String,
    pub params: Vec<Param>,
    pub permission: Permission,
}

impl ToolDefinition {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        params: Vec<Param>,
        permission: Permission,
    ) -> ToolDefinition {
        ToolDefinition {
            name: name.into(),
            description: description.into(),
            params,
            permission,
        }
    }

    /// Looks up a declared parameter.
    pub fn param(&self, name: &str) -> Option<&Param> {
        self.params.iter().find(|p| p.name == name)
    }

    pub fn required_param_names(&self) -> Vec<&str> {
        self.params
            .iter()
            .filter(|p| p.required)
            .map(|p| p.name.as_str())
            .collect()
    }

    /// The `tools[]` entry for an Ollama chat request.
    ///
    /// Shape: `{"type":"function","function":{"name","description","parameters"}}`
    pub fn to_ollama_json(&self) -> Value {
        let mut properties = Vec::new();
        let mut required = Vec::new();
        for param in &self.params {
            properties.push((param.name.clone(), param.schema.to_json()));
            if param.required {
                required.push(Value::str(&param.name));
            }
        }
        let mut parameters = vec![
            ("type".to_string(), Value::str("object")),
            ("properties".to_string(), Value::Object(properties)),
        ];
        // Omit `required` when empty: some templates render `[]` poorly.
        if !required.is_empty() {
            parameters.push(("required".to_string(), Value::Array(required)));
        }
        parameters.push(("additionalProperties".to_string(), Value::Bool(false)));

        let function = Value::Object(vec![
            ("name".to_string(), Value::str(&self.name)),
            ("description".to_string(), Value::str(&self.description)),
            ("parameters".to_string(), Value::Object(parameters)),
        ]);
        Value::Object(vec![
            ("type".to_string(), Value::str("function")),
            ("function".to_string(), function),
        ])
    }

    /// Validates the definition itself. Cheap invariant checks that would
    /// otherwise surface as confusing runtime behaviour, run once at registry
    /// build time.
    pub fn validate(&self) -> Result<(), String> {
        if self.name.trim().is_empty() {
            return Err("tool name must not be empty".into());
        }
        if !self
            .name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '.' || c == '_')
        {
            return Err(format!(
                "tool name `{}` must be lowercase and may contain only [a-z0-9._]",
                self.name
            ));
        }
        if self.description.trim().is_empty() {
            return Err(format!("tool `{}` needs a description", self.name));
        }
        let mut seen = Vec::new();
        for param in &self.params {
            if param.name.trim().is_empty() {
                return Err(format!(
                    "tool `{}` has a parameter without a name",
                    self.name
                ));
            }
            if seen.contains(&param.name) {
                return Err(format!(
                    "tool `{}` declares parameter `{}` twice",
                    self.name, param.name
                ));
            }
            seen.push(param.name.clone());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ToolDefinition {
        ToolDefinition::new(
            "calculator.evaluate",
            "Evaluate an arithmetic expression.",
            vec![Param::required(
                "expression",
                Schema::string_of_len("The expression to evaluate.", 1, 500),
            )],
            Permission::ReadOnly,
        )
    }

    #[test]
    fn ollama_json_has_the_documented_shape() {
        let json = sample().to_ollama_json();
        assert_eq!(json.get("type").unwrap().as_str(), Some("function"));
        let function = json.get("function").unwrap();
        assert_eq!(
            function.get("name").unwrap().as_str(),
            Some("calculator.evaluate")
        );
        assert!(
            function
                .get("description")
                .unwrap()
                .as_str()
                .unwrap()
                .contains("arithmetic")
        );

        let params = function.get("parameters").unwrap();
        assert_eq!(params.get("type").unwrap().as_str(), Some("object"));
        assert_eq!(
            params.get("additionalProperties").unwrap().as_bool(),
            Some(false),
            "unknown arguments must be refused by the schema we advertise"
        );

        let expression = params.get("properties").unwrap().get("expression").unwrap();
        assert_eq!(expression.get("type").unwrap().as_str(), Some("string"));
        assert_eq!(expression.get("minLength").unwrap().as_f64(), Some(1.0));
        assert_eq!(expression.get("maxLength").unwrap().as_f64(), Some(500.0));

        let required = params.get("required").unwrap().as_array().unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0].as_str(), Some("expression"));
    }

    #[test]
    fn required_is_omitted_when_there_are_no_parameters() {
        let def = ToolDefinition::new("system.get_time", "Clock.", vec![], Permission::ReadOnly);
        let params = def.to_ollama_json();
        let params = params
            .get("function")
            .unwrap()
            .get("parameters")
            .unwrap()
            .clone();
        assert!(
            params.get("required").is_none(),
            "empty required must be omitted"
        );
        assert!(
            params.get("properties").unwrap().get("x").is_none(),
            "a zero-argument tool advertises no properties"
        );
    }

    #[test]
    fn optional_parameters_are_not_required() {
        let def = ToolDefinition::new(
            "workspace.list_files",
            "List files.",
            vec![
                Param::optional("path", Schema::string("Directory.")),
                Param::optional("depth", Schema::integer("Levels.")),
            ],
            Permission::WorkspaceRead,
        );
        assert!(def.required_param_names().is_empty());
        assert_eq!(def.params.len(), 2);
        assert_eq!(def.param("depth").unwrap().schema.type_name(), "integer");
        assert!(def.param("nope").is_none());
    }

    #[test]
    fn enum_schema_serializes_allowed_values() {
        let schema = Schema::enum_of("Ordering.", &["name", "size"]);
        let json = schema.to_json();
        let allowed = json.get("enum").unwrap().as_array().unwrap();
        assert_eq!(allowed.len(), 2);
        assert_eq!(allowed[1].as_str(), Some("size"));
    }

    #[test]
    fn numeric_bounds_serialize() {
        let schema = Schema::Integer {
            description: "levels".into(),
            min: Some(1),
            max: Some(5),
        };
        let json = schema.to_json();
        assert_eq!(json.get("minimum").unwrap().as_f64(), Some(1.0));
        assert_eq!(json.get("maximum").unwrap().as_f64(), Some(5.0));

        let schema = Schema::Number {
            description: "score".into(),
            min: Some(0.5),
            max: None,
        };
        let json = schema.to_json();
        assert_eq!(json.get("minimum").unwrap().as_f64(), Some(0.5));
        assert!(json.get("maximum").is_none());
    }

    #[test]
    fn permission_ordering_and_silence() {
        assert!(Permission::ReadOnly < Permission::WorkspaceRead);
        assert!(Permission::WorkspaceRead < Permission::WorkspaceWrite);
        assert!(Permission::WorkspaceWrite < Permission::Exec);
        assert!(Permission::ReadOnly.is_silent());
        assert!(Permission::WorkspaceRead.is_silent());
        assert!(!Permission::WorkspaceWrite.is_silent());
        assert!(!Permission::Exec.is_silent());
        assert_eq!(Permission::WorkspaceWrite.as_str(), "workspace-write");
    }

    #[test]
    fn definitions_validate_their_own_invariants() {
        assert!(sample().validate().is_ok());

        let empty = ToolDefinition::new("", "d", vec![], Permission::ReadOnly);
        assert!(empty.validate().unwrap_err().contains("must not be empty"));

        let upper = ToolDefinition::new("Calculator.Evaluate", "d", vec![], Permission::ReadOnly);
        assert!(upper.validate().unwrap_err().contains("lowercase"));

        let no_desc = ToolDefinition::new("a.b", "  ", vec![], Permission::ReadOnly);
        assert!(no_desc.validate().unwrap_err().contains("description"));

        let dup = ToolDefinition::new(
            "a.b",
            "d",
            vec![
                Param::required("x", Schema::string("x")),
                Param::optional("x", Schema::string("x")),
            ],
            Permission::ReadOnly,
        );
        assert!(dup.validate().unwrap_err().contains("twice"));
    }
}
