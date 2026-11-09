//! Provider-independent chat request types.
//!
//! Everything the orchestrator hands to an [`crate::llm::provider::LlmProvider`]
//! is expressed here, so swapping Ollama for `llama.cpp`, LM Studio or a remote
//! OpenAI-compatible endpoint in M5 only requires a new `provider` impl — no
//! changes in the agent core.
//!
//! Tool-call fields are part of the model from day one even though M0 never
//! populates them: M1 (tool calling) then adds behaviour, not structure.

use crate::json::{self, Value};

/// Who produced a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    /// The result of a tool call, sent back to the model.
    Tool,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }

    pub fn parse(s: &str) -> Option<Role> {
        match s {
            "system" => Some(Role::System),
            "user" => Some(Role::User),
            "assistant" => Some(Role::Assistant),
            "tool" => Some(Role::Tool),
            _ => None,
        }
    }
}

/// A tool invocation proposed by the model (parsed in M1, carried in M0).
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    /// Provider-assigned id, when the provider supplies one.
    pub id: Option<String>,
    pub name: String,
    /// Arguments as a JSON object. Providers that return a JSON *string*
    /// are normalized to an object here.
    pub arguments: Value,
}

impl ToolCall {
    pub fn new(name: impl Into<String>, arguments: Value) -> Self {
        Self {
            id: None,
            name: name.into(),
            arguments,
        }
    }

    fn to_value(&self) -> Value {
        // Ollama expects `arguments` as an object, so a JSON *string* form is
        // normalized before it leaves the crate.
        let arguments = match &self.arguments {
            Value::String(raw) => json::parse(raw).unwrap_or_else(|_| self.arguments.clone()),
            other => other.clone(),
        };
        let function = Value::Object(vec![
            ("name".to_string(), Value::str(&self.name)),
            ("arguments".to_string(), arguments),
        ]);
        let mut obj = Vec::new();
        if let Some(id) = &self.id {
            obj.push(("id".to_string(), Value::str(id)));
        }
        obj.push(("function".to_string(), function));
        Value::Object(obj)
    }

    /// Parses one entry of a provider's `tool_calls` array.
    pub fn from_value(v: &Value) -> Option<ToolCall> {
        let function = v.get("function")?;
        let name = function.get("name")?.as_str()?.to_string();
        let arguments = match function.get("arguments") {
            // Some builds send the arguments as an encoded JSON string.
            Some(Value::String(s)) => json::parse(s).unwrap_or_else(|_| Value::str(s)),
            Some(other) => other.clone(),
            None => Value::Object(Vec::new()),
        };
        Some(ToolCall {
            id: v.get("id").and_then(|i| i.as_str()).map(str::to_string),
            name,
            arguments,
        })
    }
}

/// A single chat message.
#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    /// Base64 images for multimodal models (unused in M0).
    pub images: Vec<String>,
    /// Name of the tool this message answers (role = tool only).
    pub tool_name: Option<String>,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            images: Vec::new(),
            tool_name: None,
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Message::new(Role::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Message::new(Role::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Message::new(Role::Assistant, content)
    }

    pub fn tool(tool_name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_name: Some(tool_name.into()),
            ..Message::new(Role::Tool, content)
        }
    }

    pub fn is_empty_assistant(&self) -> bool {
        self.role == Role::Assistant && self.content.trim().is_empty() && self.tool_calls.is_empty()
    }

    pub fn to_value(&self) -> Value {
        let mut entries = vec![
            ("role".to_string(), Value::str(self.role.as_str())),
            ("content".to_string(), Value::str(&self.content)),
        ];
        if !self.images.is_empty() {
            entries.push((
                "images".to_string(),
                Value::Array(self.images.iter().map(Value::str).collect()),
            ));
        }
        if !self.tool_calls.is_empty() {
            entries.push((
                "tool_calls".to_string(),
                Value::Array(self.tool_calls.iter().map(ToolCall::to_value).collect()),
            ));
        }
        if let Some(name) = &self.tool_name {
            entries.push(("tool_name".to_string(), Value::str(name)));
        }
        Value::Object(entries)
    }

    pub fn from_value(v: &Value) -> Message {
        let role = v
            .get("role")
            .and_then(Value::as_str)
            .and_then(Role::parse)
            .unwrap_or(Role::Assistant);
        let content = v
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let tool_calls = v
            .get("tool_calls")
            .and_then(Value::as_array)
            .map(|items| items.iter().filter_map(ToolCall::from_value).collect())
            .unwrap_or_default();
        let images = v
            .get("images")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|i| i.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        Message {
            role,
            content,
            tool_calls,
            images,
            tool_name: v
                .get("tool_name")
                .and_then(Value::as_str)
                .map(str::to_string),
        }
    }
}

/// Sampling options forwarded to the provider.
#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    pub temperature: f64,
    pub num_ctx: Option<u64>,
    pub num_predict: Option<i64>,
    pub seed: Option<u64>,
    pub stop: Vec<String>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            num_ctx: None,
            num_predict: None,
            seed: None,
            stop: Vec::new(),
        }
    }
}

impl Options {
    fn to_value(&self) -> Value {
        let mut entries = vec![("temperature".to_string(), Value::Number(self.temperature))];
        if let Some(n) = self.num_ctx {
            entries.push(("num_ctx".to_string(), Value::Number(n as f64)));
        }
        if let Some(n) = self.num_predict {
            entries.push(("num_predict".to_string(), Value::Number(n as f64)));
        }
        if let Some(s) = self.seed {
            entries.push(("seed".to_string(), Value::Number(s as f64)));
        }
        if !self.stop.is_empty() {
            entries.push((
                "stop".to_string(),
                Value::Array(self.stop.iter().map(Value::str).collect()),
            ));
        }
        Value::Object(entries)
    }
}

/// A complete chat request.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub stream: bool,
    pub options: Options,
    /// JSON schema for the system prompt (M1).
    pub tools: Vec<Value>,
    /// Ollama-only: how long to keep the model resident, e.g. `"10m"`.
    pub keep_alive: Option<String>,
    /// Ollama-only: request the model's reasoning trace when supported.
    pub think: Option<bool>,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<Message>, stream: bool) -> Self {
        Self {
            model: model.into(),
            messages,
            stream,
            options: Options::default(),
            tools: Vec::new(),
            keep_alive: None,
            think: None,
        }
    }

    pub fn with_options(mut self, options: Options) -> Self {
        self.options = options;
        self
    }

    /// Serializes to the Ollama `/api/chat` request body.
    pub fn to_json(&self) -> String {
        let mut entries = vec![
            ("model".to_string(), Value::str(&self.model)),
            (
                "messages".to_string(),
                Value::Array(self.messages.iter().map(Message::to_value).collect()),
            ),
            ("stream".to_string(), Value::Bool(self.stream)),
            ("options".to_string(), self.options.to_value()),
        ];
        if !self.tools.is_empty() {
            entries.push(("tools".to_string(), Value::Array(self.tools.clone())));
        }
        if let Some(keep_alive) = &self.keep_alive {
            entries.push(("keep_alive".to_string(), Value::str(keep_alive)));
        }
        if let Some(think) = self.think {
            entries.push(("think".to_string(), Value::Bool(think)));
        }
        json::to_string(&Value::Object(entries))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_json_has_expected_shape() {
        let mut req = ChatRequest::new("qwen3.5:4b", vec![Message::user("hi")], true);
        req.keep_alive = Some("5m".into());
        let v = json::parse(&req.to_json()).unwrap();
        assert_eq!(v.get("model").unwrap().as_str(), Some("qwen3.5:4b"));
        assert_eq!(v.get("stream").unwrap().as_bool(), Some(true));
        let msg = v.get("messages").unwrap().index(0).unwrap();
        assert_eq!(msg.get("role").unwrap().as_str(), Some("user"));
        assert_eq!(msg.get("content").unwrap().as_str(), Some("hi"));
        assert_eq!(
            v.get("options")
                .unwrap()
                .get("temperature")
                .unwrap()
                .as_f64(),
            Some(0.7)
        );
        assert_eq!(v.get("keep_alive").unwrap().as_str(), Some("5m"));
        assert!(
            v.get("tools").is_none(),
            "tools key must be omitted when empty"
        );
    }

    #[test]
    fn tool_calls_survive_a_round_trip() {
        let mut asst = Message::assistant("");
        asst.tool_calls.push(ToolCall::new(
            "read_file",
            json::parse(r#"{"path":"a.txt"}"#).unwrap(),
        ));
        let back = Message::from_value(&asst.to_value());
        assert_eq!(back.tool_calls.len(), 1);
        assert_eq!(back.tool_calls[0].name, "read_file");
        assert_eq!(
            back.tool_calls[0].arguments.get("path").unwrap().as_str(),
            Some("a.txt")
        );
        assert_eq!(back.role, Role::Assistant);
    }

    #[test]
    fn tool_arguments_sent_as_string_are_normalized() {
        let raw = json::parse(r#"{"function":{"name":"calc","arguments":"{\"a\":1}"}}"#).unwrap();
        let call = ToolCall::from_value(&raw).unwrap();
        assert_eq!(call.arguments.get("a").unwrap().as_f64(), Some(1.0));
    }

    #[test]
    fn malformed_tool_call_is_skipped() {
        let raw = json::parse(r#"{"no_function":true}"#).unwrap();
        assert!(ToolCall::from_value(&raw).is_none());
    }

    #[test]
    fn role_parse_rejects_unknown() {
        assert_eq!(Role::parse("user"), Some(Role::User));
        assert_eq!(Role::parse("wizard"), None);
    }
}
