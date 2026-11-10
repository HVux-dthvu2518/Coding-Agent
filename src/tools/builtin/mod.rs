//! The M1 built-in tools.
//!
//! Exactly three, per the milestone brief: one that touches nothing
//! (`calculator.evaluate`), one that reads the workspace and nothing else
//! (`workspace.list_files`), and one that closes the model's knowledge gap
//! (`system.get_time`). Shell, file writes, git and browser tools are M3/M5 and
//! are deliberately absent — every tool added here widens what a 4B model can
//! do to the user's machine.

pub mod calculator;
pub mod time;
pub mod workspace;

pub use calculator::Calculator;
pub use time::GetTime;
pub use workspace::ListFiles;

use crate::tools::registry::ToolRegistry;

/// Registers the built-in tools.
///
/// The single place the tool set is defined, so `app` never assembles it by hand
/// and tests cannot accidentally run against a different set.
pub fn register_all(
    registry: &mut ToolRegistry,
) -> Result<(), crate::tools::registry::RegistryError> {
    registry.register(Calculator)?;
    registry.register(ListFiles)?;
    registry.register(GetTime)?;
    Ok(())
}

/// A registry containing exactly the built-ins.
pub fn builtin_registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    // Registration can only fail on a programming error; surfacing it here keeps
    // the API ergonomic for callers while still failing loudly in tests.
    expect_registered(&mut registry);
    registry
}

fn expect_registered(registry: &mut ToolRegistry) {
    if let Err(error) = register_all(registry) {
        panic!("built-in tool registration is broken: {error}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::planner::WorkerRole;
    use crate::json;

    #[test]
    fn exactly_the_three_m1_tools_are_registered() {
        let registry = builtin_registry();
        assert_eq!(
            registry.names(),
            vec![
                "calculator.evaluate",
                "system.get_time",
                "workspace.list_files"
            ]
        );
        assert_eq!(registry.len(), 3);
    }

    #[test]
    fn every_builtin_definition_is_self_consistent() {
        for definition in builtin_registry().definitions() {
            definition
                .validate()
                .unwrap_or_else(|e| panic!("`{}` is malformed: {e}", definition.name));
            // Advertised and enforced schemas come from the same struct; check
            // the JSON actually contains what validation requires.
            let json = definition.to_ollama_json();
            let function = json.get("function").unwrap();
            assert_eq!(
                function.get("name").unwrap().as_str(),
                Some(&*definition.name)
            );
            let parameters = function.get("parameters").unwrap();
            let required = parameters
                .get("required")
                .and_then(|v| v.as_array())
                .map(<[json::Value]>::len)
                .unwrap_or(0);
            assert_eq!(required, definition.required_param_names().len());
        }
    }

    #[test]
    fn only_the_workspace_tool_touches_the_filesystem() {
        let registry = builtin_registry();
        let permissions: Vec<&str> = registry
            .definitions()
            .iter()
            .map(|d| d.permission.as_str())
            .collect();
        assert_eq!(
            permissions,
            vec!["read-only", "read-only", "workspace-read"],
            "no M1 tool may write or execute"
        );
    }

    #[test]
    fn every_working_role_gets_all_three_and_the_orchestrator_none() {
        let registry = builtin_registry();
        for role in [
            WorkerRole::Researcher,
            WorkerRole::Coder,
            WorkerRole::Reviewer,
        ] {
            let names: Vec<String> = registry
                .definitions_for(role)
                .into_iter()
                .map(|d| d.name)
                .collect();
            assert_eq!(names.len(), 3, "{} got {names:?}", role.as_str());
        }
        assert!(
            registry
                .definitions_for(WorkerRole::Orchestrator)
                .is_empty()
        );
    }

    #[test]
    fn registering_twice_is_refused() {
        let mut registry = builtin_registry();
        let err = register_all(&mut registry).unwrap_err();
        assert!(err.reason.contains("already registered"), "{err}");
    }

    #[test]
    fn a_calculator_call_works_end_to_end_through_the_registry() {
        let registry = builtin_registry();
        let call = crate::llm::request::ToolCall {
            id: Some("call_1".into()),
            name: "calculator.evaluate".into(),
            arguments: json::parse(r#"{"expression":"25 * (10 + 5)"}"#).unwrap(),
        };
        let result = registry.execute(
            &call,
            &crate::tools::ToolContext::new(".", WorkerRole::Coder),
        );
        assert!(result.is_ok(), "{result:?}");
        assert_eq!(result.call_id.as_deref(), Some("call_1"));
        assert_eq!(
            result.data.unwrap().get("formatted").unwrap().as_str(),
            Some("375")
        );
    }

    #[test]
    fn a_time_call_works_end_to_end_through_the_registry() {
        let registry = builtin_registry();
        let call = crate::llm::request::ToolCall {
            id: None,
            name: "system.get_time".into(),
            arguments: json::parse("{}").unwrap(),
        };
        let result = registry.execute(
            &call,
            &crate::tools::ToolContext::new(".", WorkerRole::Coder),
        );
        assert!(result.is_ok(), "{result:?}");
        assert!(result.data.unwrap().get("local_time").is_some());
    }
}
