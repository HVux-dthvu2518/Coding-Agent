//! Batch execution of the tool calls a model asks for in one turn.
//!
//! The registry executes one call; the executor handles a *set* of calls with the
//! concerns that only appear in batch:
//!
//! * **correlation** — every result carries the `call_id` it answers;
//! * **completeness** — a request for *n* calls always yields *n* results, even
//!   if the first one is denied (a model that gets no answer for call 2 will
//!   invent one);
//! * **ordering** — results are returned in request order, so the transcript
//!   reads the way the model asked;
//! * **isolation** — one failing call does not abort the others;
//! * **repetition** — an identical repeated call to a *pure* tool is answered
//!   from the first result instead of re-running it. Only for
//!   [`Permission::is_silent`] tools, and only when the arguments match exactly,
//!   so a mutating tool always runs and a different query never gets a stale
//!   answer.

use crate::llm::request::ToolCall;
use crate::tools::context::ToolContext;
use crate::tools::registry::ToolRegistry;
use crate::tools::result::ToolResult;

/// Executes the calls of one assistant turn.
pub struct Executor<'a> {
    registry: &'a ToolRegistry,
    /// Collapse identical repeated calls to pure tools within one turn.
    pub dedupe: bool,
}

impl<'a> Executor<'a> {
    pub fn new(registry: &'a ToolRegistry) -> Executor<'a> {
        Executor {
            registry,
            dedupe: true,
        }
    }

    /// Runs every call, collecting results in request order.
    ///
    /// `on_result` is invoked per call so the UI can trace execution as it
    /// happens; returning `Err(())` stops before the next call (user cancel).
    pub fn execute_all<F>(
        &self,
        calls: &[ToolCall],
        context: &ToolContext,
        on_result: &mut F,
    ) -> Vec<ToolResult>
    where
        F: FnMut(&ToolResult) -> Result<(), ()>,
    {
        let mut completed: Vec<(ToolCall, ToolResult)> = Vec::with_capacity(calls.len());
        let mut results: Vec<ToolResult> = Vec::with_capacity(calls.len());

        for call in calls {
            let result = match self.reusable(&completed, call) {
                Some(previous) => ToolResult {
                    call_id: call.id.clone(),
                    summary: format!("{} (reused)", previous.summary),
                    ..previous.clone()
                },
                None => {
                    let result = self.registry.execute(call, context);
                    completed.push((call.clone(), result.clone()));
                    result
                }
            };

            let stop = on_result(&result).is_err();
            results.push(result);
            if stop {
                break;
            }
        }

        results
    }

    /// A previous result for the same tool with the same arguments, when reuse
    /// is safe.
    fn reusable<'c>(
        &self,
        completed: &'c [(ToolCall, ToolResult)],
        call: &ToolCall,
    ) -> Option<&'c ToolResult> {
        if !self.dedupe {
            return None;
        }
        // Only pure tools may be answered from cache: a future workspace write
        // must run every time it is asked for.
        let tool = self.registry.get(&call.name)?;
        if !tool.definition().permission.is_silent() {
            return None;
        }
        completed
            .iter()
            .find(|(previous_call, previous_result)| {
                previous_call.name == call.name
                    && same_arguments(previous_call, call)
                    // A failure is retried rather than cached.
                    && previous_result.is_ok()
            })
            .map(|(_, previous_result)| previous_result)
    }
}

/// Structural comparison of two tool calls' arguments.
///
/// Serialization of the same value is stable, and for the shallow objects tool
/// arguments occupy this is both sufficient and conservative: object key order
/// differences make calls *distinct*, which can only cost an extra execution —
/// never produce a stale answer.
pub fn same_arguments(a: &ToolCall, b: &ToolCall) -> bool {
    crate::json::to_string(&a.arguments) == crate::json::to_string(&b.arguments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::planner::WorkerRole;
    use crate::json::{self, Value};
    use crate::tools::Tool;
    use crate::tools::definition::{Param, Permission, Schema, ToolDefinition};
    use crate::tools::result::ToolError;
    use crate::tools::validation::ValidatedArgs;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Counter {
        runs: Arc<AtomicUsize>,
        permission: Permission,
        fail: bool,
    }

    impl Tool for Counter {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new(
                "calculator.evaluate",
                "Counts invocations.",
                vec![Param::optional("tag", Schema::string("Tag."))],
                self.permission,
            )
        }
        fn execute(&self, args: &ValidatedArgs, _c: &ToolContext) -> Result<ToolResult, ToolError> {
            self.runs.fetch_add(1, Ordering::SeqCst);
            if self.fail {
                return Err(ToolError::ExecutionFailed {
                    tool: "calculator.evaluate".into(),
                    reason: "failed on purpose".into(),
                });
            }
            let tag = args.str("tag").unwrap_or("-");
            Ok(ToolResult::success(
                "calculator.evaluate",
                Value::object([("tag", Value::str(tag))]),
                format!("run with tag {tag}"),
            ))
        }
    }

    struct Boom;

    impl Tool for Boom {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new(
                "workspace.read_file",
                "Fails.",
                vec![],
                Permission::ReadOnly,
            )
        }
        fn execute(&self, _a: &ValidatedArgs, _c: &ToolContext) -> Result<ToolResult, ToolError> {
            Err(ToolError::Rejected {
                tool: "workspace.read_file".into(),
                reason: "always".into(),
            })
        }
    }

    fn setup(permission: Permission, fail: bool) -> (ToolRegistry, Arc<AtomicUsize>) {
        let runs = Arc::new(AtomicUsize::new(0));
        let mut registry = ToolRegistry::new();
        registry
            .register(Counter {
                runs: runs.clone(),
                permission,
                fail,
            })
            .unwrap();
        (registry, runs)
    }

    fn call(id: &str, name: &str, args: &str) -> ToolCall {
        ToolCall {
            id: Some(id.to_string()),
            name: name.to_string(),
            arguments: json::parse(args).unwrap(),
        }
    }

    fn ctx() -> ToolContext {
        ToolContext::new(".", WorkerRole::Coder)
    }

    #[test]
    fn every_call_gets_a_result_in_request_order() {
        let (registry, runs) = setup(Permission::ReadOnly, false);
        let executor = Executor::new(&registry);
        let calls = vec![
            call("1", "calculator.evaluate", r#"{"tag":"a"}"#),
            call("2", "calculator.evaluate", r#"{"tag":"b"}"#),
            call("3", "calculator.evaluate", r#"{"tag":"c"}"#),
        ];
        let results = executor.execute_all(&calls, &ctx(), &mut |_| Ok(()));
        assert_eq!(results.len(), 3);
        let ids: Vec<_> = results.iter().map(|r| r.call_id.clone().unwrap()).collect();
        assert_eq!(ids, vec!["1", "2", "3"]);
        assert_eq!(
            runs.load(Ordering::SeqCst),
            3,
            "different arguments must all run"
        );
        assert!(results[0].summary.contains("tag a"));
        assert!(results[2].summary.contains("tag c"));
    }

    #[test]
    fn one_failure_does_not_abort_the_batch() {
        let mut registry = ToolRegistry::new();
        registry.register(Boom).unwrap();
        registry
            .register(Counter {
                runs: Arc::new(AtomicUsize::new(0)),
                permission: Permission::ReadOnly,
                fail: false,
            })
            .unwrap();

        let executor = Executor::new(&registry);
        let calls = vec![
            call("a", "workspace.read_file", "{}"),
            call("b", "calculator.evaluate", r#"{"tag":"ok"}"#),
            call("c", "missing.tool", "{}"),
        ];
        let results = executor.execute_all(&calls, &ctx(), &mut |_| Ok(()));
        assert_eq!(results.len(), 3, "a failure must not swallow later calls");
        assert!(results[0].is_err());
        assert!(results[1].is_ok());
        assert!(results[2].is_err());
        assert_eq!(results[0].call_id.as_deref(), Some("a"));
        assert_eq!(results[1].call_id.as_deref(), Some("b"));
        assert_eq!(results[2].call_id.as_deref(), Some("c"));
    }

    #[test]
    fn a_denied_call_still_produces_a_correlated_result() {
        let (registry, runs) = setup(Permission::ReadOnly, false);
        let executor = Executor::new(&registry);
        let calls = vec![call("x", "calculator.evaluate", "{}")];
        let results = executor.execute_all(
            &calls,
            &ToolContext::new(".", WorkerRole::Orchestrator),
            &mut |_| Ok(()),
        );
        assert_eq!(results.len(), 1);
        assert!(results[0].denied);
        assert_eq!(results[0].error_kind(), Some("permission_denied"));
        assert_eq!(results[0].call_id.as_deref(), Some("x"));
        assert_eq!(runs.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn identical_read_only_calls_are_reused() {
        let (registry, runs) = setup(Permission::ReadOnly, false);
        let executor = Executor::new(&registry);
        let calls = vec![
            call("1", "calculator.evaluate", r#"{"tag":"same"}"#),
            call("2", "calculator.evaluate", r#"{"tag":"same"}"#),
        ];
        let results = executor.execute_all(&calls, &ctx(), &mut |_| Ok(()));
        assert_eq!(results.len(), 2);
        assert_eq!(
            runs.load(Ordering::SeqCst),
            1,
            "the second call is served from the first"
        );
        assert!(results[1].summary.contains("reused"));
        assert_eq!(results[1].call_id.as_deref(), Some("2"));
    }

    #[test]
    fn different_arguments_are_never_served_from_cache() {
        // Guards the bug where de-duplication keyed only on the tool name.
        let (registry, runs) = setup(Permission::ReadOnly, false);
        let executor = Executor::new(&registry);
        let calls = vec![
            call("1", "calculator.evaluate", r#"{"tag":"first"}"#),
            call("2", "calculator.evaluate", r#"{"tag":"second"}"#),
        ];
        let results = executor.execute_all(&calls, &ctx(), &mut |_| Ok(()));
        assert_eq!(runs.load(Ordering::SeqCst), 2);
        assert!(
            results[1].summary.contains("tag second"),
            "{:?}",
            results[1]
        );
        assert!(!results[1].summary.contains("reused"));
    }

    #[test]
    fn mutating_tools_are_never_deduplicated() {
        let (registry, runs) = setup(Permission::WorkspaceWrite, false);
        let executor = Executor::new(&registry);
        let calls = vec![
            call("1", "calculator.evaluate", r#"{"tag":"same"}"#),
            call("2", "calculator.evaluate", r#"{"tag":"same"}"#),
        ];
        executor.execute_all(&calls, &ctx(), &mut |_| Ok(()));
        assert_eq!(
            runs.load(Ordering::SeqCst),
            2,
            "a mutating tool must run every time"
        );
    }

    #[test]
    fn dedupe_can_be_disabled() {
        let (registry, runs) = setup(Permission::ReadOnly, false);
        let mut executor = Executor::new(&registry);
        executor.dedupe = false;
        let calls = vec![
            call("1", "calculator.evaluate", "{}"),
            call("2", "calculator.evaluate", "{}"),
        ];
        executor.execute_all(&calls, &ctx(), &mut |_| Ok(()));
        assert_eq!(runs.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn failures_are_never_reused() {
        let (registry, runs) = setup(Permission::ReadOnly, true);
        let executor = Executor::new(&registry);
        let calls = vec![
            call("1", "calculator.evaluate", "{}"),
            call("2", "calculator.evaluate", "{}"),
        ];
        let results = executor.execute_all(&calls, &ctx(), &mut |_| Ok(()));
        assert_eq!(results.len(), 2);
        assert_eq!(
            runs.load(Ordering::SeqCst),
            2,
            "a failure must be retried, not cached"
        );
        assert!(results.iter().all(|r| r.is_err()));
    }

    #[test]
    fn cancellation_stops_before_the_next_call() {
        let (registry, runs) = setup(Permission::ReadOnly, false);
        let executor = Executor::new(&registry);
        let calls = vec![
            call("1", "calculator.evaluate", r#"{"tag":"a"}"#),
            call("2", "calculator.evaluate", r#"{"tag":"b"}"#),
        ];
        let mut seen = 0;
        let results = executor.execute_all(&calls, &ctx(), &mut |_| {
            seen += 1;
            Err(())
        });
        assert_eq!(results.len(), 1);
        assert_eq!(seen, 1);
        assert_eq!(runs.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn an_empty_batch_is_not_an_error() {
        let (registry, _) = setup(Permission::ReadOnly, false);
        let executor = Executor::new(&registry);
        assert!(
            executor
                .execute_all(&[], &ctx(), &mut |_| Ok(()))
                .is_empty()
        );
    }

    #[test]
    fn argument_comparison_is_structural_and_conservative() {
        let a = call("1", "t", r#"{"x":1,"y":[2,3]}"#);
        let same = call("2", "t", r#"{"x":1,"y":[2,3]}"#);
        let different = call("3", "t", r#"{"x":2,"y":[2,3]}"#);
        assert!(same_arguments(&a, &same));
        assert!(!same_arguments(&a, &different));
        // Key order differences are treated as distinct: conservative, which
        // can only cost an extra execution.
        let reordered = call("4", "t", r#"{"y":[2,3],"x":1}"#);
        assert!(!same_arguments(&a, &reordered));
    }
}
