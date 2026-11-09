//! Specialized worker roles — the seam for M5, no runtime cost in M0.
//!
//! The architecture document is emphatic that the 4B model should not run as
//! three competing agents. Workers are therefore **logical roles over one model
//! instance**: each role is (a) a prompt suffix, (b) an allow-list of tools it
//! may be offered, and (c) a note about whether it can write to the workspace.
//!
//! M0 uses roles only to label the plan steps produced by `planner`. M1 will
//! use `allowed_tools` to narrow the tool schema sent with a delegated turn,
//! which is what actually keeps a 4B model inside its lanes.

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
    allowed_tools: &["fs.read", "fs.list", "fs.stat", "web."],
    may_mutate: false,
    prompt_suffix: "You are the researcher: gather facts from provided material only. \
                    Quote the source path or URL for every claim. Never modify files.",
};

const CODER: WorkerProfile = WorkerProfile {
    role: WorkerRole::Coder,
    allowed_tools: &["fs.read", "fs.write", "fs.list", "fs.stat", "shell.", "git."],
    may_mutate: true,
    prompt_suffix: "You are the coder: make the smallest change that satisfies the objective. \
                    Read before writing. Explain non-obvious edits in one sentence.",
};

const REVIEWER: WorkerProfile = WorkerProfile {
    role: WorkerRole::Reviewer,
    allowed_tools: &["fs.read", "fs.list", "fs.stat", "shell.", "git.diff", "git.log"],
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
    fn orchestrator_is_tool_free_until_m1() {
        assert!(!may_use_tool(WorkerRole::Orchestrator, "fs.read"));
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
