//! The tool registry: registration, discovery, and dispatch.
//!
//! One registry instance owns the definitions that are *both* advertised to the
//! model and used to validate its calls, which is what keeps the two in sync.
//!
//! `execute` never returns `Result`: a failed lookup, a denied permission or a
//! crashed tool are all values ([`ToolResult`]) because the model must always
//! receive an observation. Only a programming error (a tool whose definition
//! changed between registration and dispatch) can produce an inconsistency, and
//! that is caught at registration time instead.

use crate::llm::request::ToolCall;
use crate::tools::Tool;
use crate::tools::context::ToolContext;
use crate::tools::definition::ToolDefinition;
use crate::tools::result::{ToolError, ToolResult};
use crate::tools::validation;
use std::collections::BTreeMap;

/// A name → tool map with strict invariants.
#[derive(Default)]
pub struct ToolRegistry {
    /// `BTreeMap` keeps `list()` deterministic, which matters for prompt
    /// stability across runs.
    tools: BTreeMap<String, Box<dyn Tool>>,
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// Why a registration was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistryError {
    pub tool: String,
    pub reason: String,
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cannot register `{}`: {}", self.tool, self.reason)
    }
}

impl std::error::Error for RegistryError {}

impl ToolRegistry {
    pub fn new() -> ToolRegistry {
        ToolRegistry::default()
    }

    /// Registers a tool, refusing duplicates and malformed definitions.
    ///
    /// Duplicate rejection is load-bearing: silently overwriting a tool would
    /// mean the advertised schema and the executed behaviour can diverge.
    pub fn register<T: Tool + 'static>(&mut self, tool: T) -> Result<(), RegistryError> {
        let definition = tool.definition();
        if let Err(reason) = definition.validate() {
            return Err(RegistryError {
                tool: definition.name.clone(),
                reason,
            });
        }
        if self.tools.contains_key(&definition.name) {
            return Err(RegistryError {
                tool: definition.name.clone(),
                reason: "a tool with this name is already registered".into(),
            });
        }
        self.tools.insert(definition.name.clone(), Box::new(tool));
        Ok(())
    }

    /// Registers several tools; stops at the first failure.
    pub fn register_all<I, T>(&mut self, tools: I) -> Result<(), RegistryError>
    where
        I: IntoIterator<Item = T>,
        T: Tool + 'static,
    {
        for tool in tools {
            self.register(tool)?;
        }
        Ok(())
    }

    /// Removes a tool, returning whether it existed.
    pub fn unregister(&mut self, name: &str) -> bool {
        self.tools.remove(name).is_some()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(AsRef::as_ref)
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Every registered definition, sorted by name.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|t| t.definition()).collect()
    }

    /// Definitions the given role may be offered, sorted by name.
    ///
    /// Advertising only permitted tools is better than advertising everything
    /// and denying later: a 4B model that cannot see `shell.exec` cannot spend a
    /// turn asking for it.
    pub fn definitions_for(&self, role: crate::agent::planner::WorkerRole) -> Vec<ToolDefinition> {
        self.definitions()
            .into_iter()
            .filter(|d| crate::agent::workers::may_use_tool(role, &d.name))
            .collect()
    }

    /// Tool names, sorted.
    pub fn names(&self) -> Vec<&str> {
        self.tools.keys().map(String::as_str).collect()
    }

    /// Validates arguments without executing. Used by tests and by `doctor`.
    pub fn validate(&self, name: &str, arguments: &crate::json::Value) -> Result<(), ToolError> {
        let tool = self.get(name).ok_or_else(|| ToolError::UnknownTool {
            name: name.to_string(),
        })?;
        validation::validate(&tool.definition(), arguments).map_err(|reason| {
            ToolError::InvalidArguments {
                tool: name.to_string(),
                reason,
            }
        })?;
        Ok(())
    }

    /// Runs one model-requested call through lookup → permission → validation →
    /// execution, always producing a result.
    pub fn execute(&self, call: &ToolCall, context: &ToolContext) -> ToolResult {
        let result = self.execute_inner(call, context);
        result.with_call_id(call.id.clone())
    }

    fn execute_inner(&self, call: &ToolCall, context: &ToolContext) -> ToolResult {
        // 1. Lookup. Unknown tools are a model mistake, not a crash.
        let Some(tool) = self.get(&call.name) else {
            return ToolResult::failure(ToolError::UnknownTool {
                name: call.name.clone(),
            });
        };
        let definition = tool.definition();

        // 2. Permission. Refusal is never treated as execution.
        if let Err(reason) = context.check_permission(definition.permission, &definition.name) {
            return ToolResult::denied(ToolError::PermissionDenied {
                tool: definition.name,
                reason,
            });
        }

        // 3. Validation, against the same schema the model was shown.
        let args = match validation::validate(&definition, &call.arguments) {
            Ok(args) => args,
            Err(reason) => {
                return ToolResult::denied(ToolError::InvalidArguments {
                    tool: definition.name,
                    reason,
                });
            }
        };

        // 4. Execution, timed.
        let started = std::time::Instant::now();
        let outcome = tool.execute(&args, context);
        let elapsed = started.elapsed();

        let result = match outcome {
            Ok(mut result) => {
                result.duration = elapsed;
                result
            }
            Err(error) => ToolResult::failure(error).with_duration(elapsed),
        };

        // 5. Budget check: a tool that overran is reported even if it succeeded,
        // because a slow tool in a loop is how an agent hangs.
        if elapsed > context.timeout {
            let mut timed_out = ToolResult::failure(ToolError::Timeout {
                tool: definition.name,
                limit_ms: context.timeout.as_millis() as u64,
            })
            .with_duration(elapsed);
            timed_out.denied = result.denied;
            timed_out
        } else {
            result
        }
    }

    /// Convenience for `doctor`: a human-readable list of what is available to a
    /// role.
    pub fn describe(&self, role: crate::agent::planner::WorkerRole) -> Vec<String> {
        self.definitions_for(role)
            .into_iter()
            .map(|d| format!("{} [{}] — {}", d.name, d.permission.as_str(), d.description))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::planner::WorkerRole;
    use crate::json::{self, Value};
    use crate::tools::definition::{Param, Permission, Schema};
    use crate::tools::validation::ValidatedArgs;

    /// A tool that echoes its argument; lets tests exercise dispatch without
    /// depending on the built-ins.
    struct Echo {
        name: &'static str,
        permission: Permission,
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Echo {
        fn new(name: &'static str, permission: Permission) -> Echo {
            Echo {
                name,
                permission,
                calls: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            }
        }
    }

    impl Tool for Echo {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new(
                self.name,
                "Echoes its input.",
                vec![Param::optional("text", Schema::string("Text."))],
                self.permission,
            )
        }

        fn execute(
            &self,
            args: &ValidatedArgs,
            _context: &ToolContext,
        ) -> Result<ToolResult, ToolError> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let text = args.str("text").unwrap_or("nothing");
            Ok(ToolResult::success(
                self.name,
                Value::object([("echo", Value::str(text))]),
                format!("echoed {text}"),
            ))
        }
    }

    struct Boom;

    impl Tool for Boom {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new(
                "workspace.read_file",
                "Always fails.",
                vec![],
                Permission::ReadOnly,
            )
        }

        fn execute(
            &self,
            _args: &ValidatedArgs,
            _context: &ToolContext,
        ) -> Result<ToolResult, ToolError> {
            Err(ToolError::ExecutionFailed {
                tool: "workspace.read_file".into(),
                reason: "kaboom".into(),
            })
        }
    }

    fn ctx(role: WorkerRole) -> ToolContext {
        ToolContext::new(".", role)
    }

    fn call(name: &str, args: &str) -> ToolCall {
        ToolCall {
            id: Some(format!("call-{name}")),
            name: name.to_string(),
            arguments: json::parse(args).unwrap(),
        }
    }

    #[test]
    fn registration_rejects_duplicates() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Echo::new("calculator.evaluate", Permission::ReadOnly))
            .unwrap();
        let err = registry
            .register(Echo::new("calculator.evaluate", Permission::ReadOnly))
            .unwrap_err();
        assert!(err.reason.contains("already registered"), "{err}");
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn registration_rejects_malformed_definitions() {
        struct Bad;
        impl Tool for Bad {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition::new("Bad Name", "", vec![], Permission::ReadOnly)
            }
            fn execute(
                &self,
                _a: &ValidatedArgs,
                _c: &ToolContext,
            ) -> Result<ToolResult, ToolError> {
                unreachable!()
            }
        }
        let mut registry = ToolRegistry::new();
        let err = registry.register(Bad).unwrap_err();
        assert!(
            err.reason.contains("lowercase") || err.reason.contains("description"),
            "{err}"
        );
        assert!(registry.is_empty());
    }

    #[test]
    fn lookup_and_listing_are_deterministic() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Echo::new("workspace.read_file", Permission::ReadOnly))
            .unwrap();
        registry
            .register(Echo::new("workspace.list_files", Permission::ReadOnly))
            .unwrap();
        assert_eq!(
            registry.names(),
            vec!["workspace.list_files", "workspace.read_file"]
        );
        assert!(registry.contains("workspace.read_file"));
        assert!(registry.get("workspace.read_file").is_some());
        assert!(registry.get("nope").is_none());
        assert_eq!(
            registry.definitions().first().unwrap().name,
            "workspace.list_files"
        );
    }

    #[test]
    fn unregister_removes_and_reports() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Echo::new("calculator.evaluate", Permission::ReadOnly))
            .unwrap();
        assert!(registry.unregister("calculator.evaluate"));
        assert!(!registry.unregister("calculator.evaluate"));
        assert!(registry.is_empty());
    }

    #[test]
    fn execute_dispatches_and_stamps_the_call_id() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Echo::new("calculator.evaluate", Permission::ReadOnly))
            .unwrap();
        let result = registry.execute(
            &call("calculator.evaluate", r#"{"text":"hi"}"#),
            &ctx(WorkerRole::Coder),
        );
        assert!(result.is_ok(), "{result:?}");
        assert_eq!(result.call_id.as_deref(), Some("call-calculator.evaluate"));
        assert_eq!(
            result.data.unwrap().get("echo").unwrap().as_str(),
            Some("hi")
        );
        assert_eq!(result.summary, "echoed hi");
    }

    #[test]
    fn unknown_tool_is_a_controlled_error() {
        let registry = ToolRegistry::new();
        let result = registry.execute(&call("does.not.exist", "{}"), &ctx(WorkerRole::Coder));
        assert_eq!(result.error_kind(), Some("unknown_tool"));
        assert!(result.is_err());
        assert!(
            !result.denied,
            "unknown is a model mistake, not a policy refusal"
        );
        assert!(result.preview().contains("does.not.exist"));
    }

    #[test]
    fn permission_is_checked_before_execution() {
        let mut registry = ToolRegistry::new();
        let tool = Echo::new("calculator.evaluate", Permission::WorkspaceWrite);
        let counter = tool.calls.clone();
        registry.register(tool).unwrap();

        // The researcher may not use this tool at all.
        let result = registry.execute(
            &call("calculator.evaluate", "{}"),
            &ctx(WorkerRole::Researcher),
        );
        assert_eq!(result.error_kind(), Some("permission_denied"));
        assert!(result.denied);
        assert_eq!(
            counter.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "a denied tool must never run"
        );

        // The coder may.
        let result = registry.execute(&call("calculator.evaluate", "{}"), &ctx(WorkerRole::Coder));
        assert!(result.is_ok(), "{result:?}");
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn invalid_arguments_never_reach_the_tool() {
        let mut registry = ToolRegistry::new();
        let tool = Echo::new("calculator.evaluate", Permission::ReadOnly);
        let counter = tool.calls.clone();
        registry.register(tool).unwrap();

        let result = registry.execute(
            &call("calculator.evaluate", r#"{"text":42}"#),
            &ctx(WorkerRole::Coder),
        );
        assert_eq!(result.error_kind(), Some("invalid_arguments"));
        assert!(result.denied);
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 0);

        let result = registry.execute(
            &call("calculator.evaluate", r#"{"bogus":1}"#),
            &ctx(WorkerRole::Coder),
        );
        assert_eq!(result.error_kind(), Some("invalid_arguments"));
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[test]
    fn tool_failures_become_results_not_panics() {
        let mut registry = ToolRegistry::new();
        registry.register(Boom).unwrap();
        let result = registry.execute(&call("workspace.read_file", "{}"), &ctx(WorkerRole::Coder));
        assert_eq!(result.error_kind(), Some("execution_failed"));
        assert!(result.preview().contains("kaboom"));
        // The model is told, in JSON.
        let content = json::parse(&result.to_model_content()).unwrap();
        assert_eq!(content.get("ok").unwrap().as_bool(), Some(false));
    }

    #[test]
    fn a_slow_tool_is_reported_as_a_timeout() {
        struct Slow;
        impl Tool for Slow {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition::new(
                    "workspace.list_files",
                    "Sleeps.",
                    vec![],
                    Permission::ReadOnly,
                )
            }
            fn execute(
                &self,
                _a: &ValidatedArgs,
                _c: &ToolContext,
            ) -> Result<ToolResult, ToolError> {
                std::thread::sleep(std::time::Duration::from_millis(30));
                Ok(ToolResult::success(
                    "workspace.list_files",
                    Value::Null,
                    "slept",
                ))
            }
        }
        let mut registry = ToolRegistry::new();
        registry.register(Slow).unwrap();
        let mut context = ctx(WorkerRole::Coder);
        context.timeout = std::time::Duration::from_millis(5);
        let result = registry.execute(&call("workspace.list_files", "{}"), &context);
        assert_eq!(result.error_kind(), Some("timeout"));
    }

    #[test]
    fn definitions_for_filters_by_role() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Echo::new("workspace.read_file", Permission::WorkspaceRead))
            .unwrap();
        registry
            .register(Echo::new("shell.exec", Permission::Exec))
            .unwrap();

        let researcher = registry.definitions_for(WorkerRole::Researcher);
        assert_eq!(researcher.len(), 1);
        assert_eq!(researcher[0].name, "workspace.read_file");

        let orchestrator = registry.definitions_for(WorkerRole::Orchestrator);
        assert!(
            orchestrator.is_empty(),
            "the orchestrator is offered no tools"
        );

        let coder = registry.definitions_for(WorkerRole::Coder);
        assert_eq!(coder.len(), 2);
    }

    #[test]
    fn validate_reports_without_executing() {
        let mut registry = ToolRegistry::new();
        let tool = Echo::new("calculator.evaluate", Permission::ReadOnly);
        let counter = tool.calls.clone();
        registry.register(tool).unwrap();

        assert!(
            registry
                .validate(
                    "calculator.evaluate",
                    &json::parse(r#"{"text":"x"}"#).unwrap()
                )
                .is_ok()
        );
        let err = registry
            .validate(
                "calculator.evaluate",
                &json::parse(r#"{"text":1}"#).unwrap(),
            )
            .unwrap_err();
        assert_eq!(err.kind(), "invalid_arguments");
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(
            registry.validate("nope", &Value::Null).unwrap_err().kind(),
            "unknown_tool"
        );
    }

    #[test]
    fn describe_lists_tools_with_permission_class() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Echo::new("workspace.read_file", Permission::WorkspaceRead))
            .unwrap();
        let lines = registry.describe(WorkerRole::Researcher);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("workspace.read_file"), "{lines:?}");
        assert!(lines[0].contains("workspace-read"), "{lines:?}");
    }

    #[test]
    fn registry_is_usable_as_a_dyn_tool_store() {
        // Object safety: `Box<dyn Tool>` is what makes the registry extensible.
        let mut registry = ToolRegistry::new();
        registry
            .register_all(vec![
                Echo::new("workspace.read_file", Permission::ReadOnly),
                Echo::new("calculator.evaluate", Permission::ReadOnly),
            ])
            .unwrap();
        assert_eq!(registry.len(), 2);
        let as_dyn: &dyn Tool = registry.get("calculator.evaluate").unwrap();
        assert_eq!(as_dyn.definition().name, "calculator.evaluate");
    }
}
