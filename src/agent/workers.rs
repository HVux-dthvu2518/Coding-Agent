//! Specialized worker roles — the seam for M5, no runtime cost in M0.
//!
//! The architecture document is emphatic that the 4B model should not run as
//! three competing agents. Workers are therefore **logical roles over one model
//! instance**: each role is (a) a prompt suffix, (b) an allow-list of tools it
//! may be offered, and (c) a note about whether it can write to the workspace.
//!
//! The allow-list is enforced twice: the registry only advertises permitted
//! tools (`ToolRegistry::definitions_for`), and dispatch re-checks it
//! (`ToolContext::check_permission`). A 4B model that cannot see `shell.exec`
//! cannot spend a turn asking for it, and one that asks anyway is refused.
//!
//! `test.` is deliberately **not** an allowed prefix: tests that need a tool
//! must name it realistically, so the policy is exercised as shipped.

use crate::agent::planner::WorkerRole;

/// Static policy for a worker role.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerProfile {
    pub role: WorkerRole,
    /// Operation names the role may be offered (prefix match, e.g. `fs.`), or
    /// empty for "no tools".
    pub allowed_tools: &'static [&'static str],
    /// Whether this role may write to the workspace (M3 tool enforcement).
    pub may_mutate: bool,
    /// Prompt suffix appended when the role is delegated to.
    pub prompt_suffix: &'static str,
}

const ORCHESTRATOR: WorkerProfile = WorkerProfile {
    role: WorkerRole::Orchestrator,
    allowed_tools: &[],
    may_mutate: false,
    prompt_suffix: "You are the orchestrator: talk to the user, decompose objectives, and \
                    delegate. Do not claim to have run tools yourself.",
};

const RESEARCHER: WorkerProfile = WorkerProfile {
    role: WorkerRole::Researcher,
    allowed_tools: &[
        "fs.read",
        "fs.list",
        "fs.stat",
        "web.",
        "workspace.",
        "calculator.evaluate",
        "system.get_time",
    ],
    may_mutate: false,
    prompt_suffix: "You are the researcher: gather facts from provided material only. \
                    Quote the source path or URL for every claim. Never modify files.",
};

const CODER: WorkerProfile = WorkerProfile {
    role: WorkerRole::Coder,
    allowed_tools: &[
        "fs.read",
        "fs.write",
        "fs.list",
        "fs.stat",
        "shell.",
        "git.",
        "workspace.",
        "calculator.evaluate",
        "system.get_time",
    ],
    may_mutate: true,
    prompt_suffix: "You are the coder: make the smallest change that satisfies the objective. \
                    Read before writing. Explain non-obvious edits in one sentence.",
};

const REVIEWER: WorkerProfile = WorkerProfile {
    role: WorkerRole::Reviewer,
    allowed_tools: &[
        "fs.read",
        "fs.list",
        "fs.stat",
        "shell.",
        "git.diff",
        "git.log",
        "workspace.",
        "calculator.evaluate",
        "system.get_time",
    ],
    may_mutate: false,
    prompt_suffix: "You are the reviewer: find defects in the proposed change and in its tests. \
                    Report concrete problems with file and line references. Never fix them here.",
};

const PROFILES: [WorkerProfile; 4] = [ORCHESTRATOR, RESEARCHER, CODER, REVIEWER];

/// Looks up the profile for a role.
pub fn profile(role: WorkerRole) -> &'static WorkerProfile {
    PROFILES
        .iter()
        .find(|p| p.role == role)
        .expect("every WorkerRole has a profile")
}

/// Whether `role` may use the tool named `tool` (which may be namespaced, e.g.
/// `fs.read` or `shell.exec`).
pub fn may_use_tool(role: WorkerRole, tool: &str) -> bool {
    let allowed = profile(role).allowed_tools;
    allowed
        .iter()
        .any(|a| tool == *a || (a.ends_with('.') && tool.starts_with(a)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_role_has_a_profile() {
        for role in [
            WorkerRole::Orchestrator,
            WorkerRole::Researcher,
            WorkerRole::Coder,
            WorkerRole::Reviewer,
        ] {
            assert_eq!(profile(role).role, role);
            assert!(!profile(role).prompt_suffix.is_empty());
        }
    }

    #[test]
    fn only_the_coder_may_mutate() {
        assert!(profile(WorkerRole::Coder).may_mutate);
        assert!(!profile(WorkerRole::Researcher).may_mutate);
        assert!(!profile(WorkerRole::Reviewer).may_mutate);
        assert!(!profile(WorkerRole::Orchestrator).may_mutate);
    }

    #[test]
    fn orchestrator_is_offered_no_tools() {
        for tool in [
            "fs.read",
            "workspace.list_files",
            "calculator.evaluate",
            "system.get_time",
        ] {
            assert!(!may_use_tool(WorkerRole::Orchestrator, tool), "{tool}");
        }
    }

    #[test]
    fn the_m1_builtins_are_reachable_by_working_roles() {
        for role in [
            WorkerRole::Researcher,
            WorkerRole::Coder,
            WorkerRole::Reviewer,
        ] {
            for tool in [
                "calculator.evaluate",
                "system.get_time",
                "workspace.list_files",
            ] {
                assert!(
                    may_use_tool(role, tool),
                    "{} may not use {tool}",
                    role.as_str()
                );
            }
        }
    }

    #[test]
    fn tool_allow_lists_use_prefixes() {
        assert!(may_use_tool(WorkerRole::Researcher, "fs.read"));
        assert!(may_use_tool(WorkerRole::Researcher, "web.fetch"));
        assert!(!may_use_tool(WorkerRole::Researcher, "fs.write"));
        assert!(may_use_tool(WorkerRole::Coder, "shell.exec"));
        assert!(!may_use_tool(WorkerRole::Reviewer, "fs.write"));
        assert!(may_use_tool(WorkerRole::Reviewer, "git.diff"));
        assert!(!may_use_tool(WorkerRole::Reviewer, "git.commit"));
    }
}
