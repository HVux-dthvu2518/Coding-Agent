//! The restricted context handed to a tool.
//!
//! Tools never see the orchestrator, the provider or the conversation. They get
//! a [`ToolContext`]: a workspace root, a role, and time/limit policy. That is
//! the difference between "the calculator is a function" and "the calculator can
//! reach into application state".
//!
//! Path resolution lives here rather than in each tool, because workspace
//! containment is exactly the kind of check that gets forgotten in one of five
//! implementations. Any future filesystem tool calls
//! [`ToolContext::resolve_in_workspace`] and inherits the guarantee.

use crate::agent::planner::WorkerRole;
use std::path::{Component, Path, PathBuf};

/// Policy and environment for one tool execution.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolContext {
    workspace_root: PathBuf,
    role: WorkerRole,
    /// Upper bound for a single tool execution.
    pub timeout: std::time::Duration,
}

impl ToolContext {
    pub fn new(workspace_root: impl Into<PathBuf>, role: WorkerRole) -> ToolContext {
        ToolContext {
            workspace_root: workspace_root.into(),
            role,
            timeout: std::time::Duration::from_secs(5),
        }
    }

    /// The role whose permissions are being enforced.
    pub fn role(&self) -> WorkerRole {
        self.role
    }

    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }

    /// Resolves `user_path` inside the workspace, refusing anything that escapes
    /// it.
    ///
    /// Rejects absolute paths, `..` components, NUL bytes and — crucially —
    /// symlinks that point outside the workspace, which a lexical check alone
    /// would miss.
    pub fn resolve_in_workspace(&self, user_path: &str, tool: &str) -> Result<PathBuf, String> {
        if user_path.contains('\0') {
            return Err("path contains a NUL byte".into());
        }
        let candidate = Path::new(user_path);
        if candidate.is_absolute() {
            return Err(format!(
                "absolute paths are not allowed; use a path relative to the workspace root ({tool})"
            ));
        }
        // Lexical containment first: prevents `..` from being resolved at all.
        let mut depth = 0i32;
        for component in candidate.components() {
            match component {
                Component::ParentDir => {
                    depth -= 1;
                    if depth < 0 {
                        return Err("path escapes the workspace root".into());
                    }
                }
                Component::Normal(_) => depth += 1,
                Component::CurDir => {}
                Component::RootDir | Component::Prefix(_) => {
                    return Err("path must be relative to the workspace root".into());
                }
            }
        }

        let joined = self.workspace_root.join(candidate);
        // `canonicalize` follows symlinks; only meaningful when the target
        // exists, so a missing path is returned as-is and reported by the tool.
        if let (Ok(real_root), Ok(real_target)) =
            (self.workspace_root.canonicalize(), joined.canonicalize())
        {
            if !real_target.starts_with(&real_root) {
                return Err("path resolves outside the workspace root (symlink)".into());
            }
        }
        Ok(joined)
    }

    /// True when the tool may run given the role's policy.
    pub fn check_permission(
        &self,
        permission: crate::tools::definition::Permission,
        tool: &str,
    ) -> Result<(), String> {
        let profile = crate::agent::workers::profile(self.role);
        if !crate::agent::workers::may_use_tool(self.role, tool) {
            return Err(format!(
                "the {} role may not use `{tool}`",
                profile.role.as_str()
            ));
        }
        if !profile.may_mutate && permission > crate::tools::definition::Permission::WorkspaceRead {
            return Err(format!(
                "the {} role may not perform {} operations",
                profile.role.as_str(),
                permission.as_str()
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> ToolContext {
        ToolContext::new("/work/project", WorkerRole::Coder)
    }

    #[test]
    fn exposes_root_and_role() {
        let context = ctx();
        assert_eq!(context.workspace_root(), Path::new("/work/project"));
        assert_eq!(context.role(), WorkerRole::Coder);
        assert!(context.timeout.as_secs() >= 1);
    }

    #[test]
    fn relative_paths_resolve_under_the_root() {
        let resolved = ctx()
            .resolve_in_workspace("src/main.rs", "workspace.list_files")
            .unwrap();
        assert_eq!(resolved, Path::new("/work/project/src/main.rs"));

        let resolved = ctx().resolve_in_workspace("./README.md", "t").unwrap();
        assert_eq!(resolved, Path::new("/work/project/README.md"));

        let resolved = ctx().resolve_in_workspace(".", "t").unwrap();
        assert_eq!(resolved, Path::new("/work/project/."));
    }

    #[test]
    fn parent_traversal_is_refused() {
        let err = ctx()
            .resolve_in_workspace("../etc/passwd", "t")
            .unwrap_err();
        assert!(err.contains("escapes"), "{err}");

        let err = ctx()
            .resolve_in_workspace("src/../../outside", "t")
            .unwrap_err();
        assert!(err.contains("escapes"), "{err}");

        // An interior `..` that stays inside is allowed.
        assert!(ctx().resolve_in_workspace("src/../README.md", "t").is_ok());
    }

    #[test]
    fn absolute_paths_are_refused() {
        let err = ctx().resolve_in_workspace("/etc/passwd", "t").unwrap_err();
        assert!(err.contains("absolute"), "{err}");
    }

    #[test]
    fn nul_bytes_are_refused() {
        let err = ctx().resolve_in_workspace("src/a\0b", "t").unwrap_err();
        assert!(err.contains("NUL"), "{err}");
    }

    #[test]
    fn symlinks_out_of_the_workspace_are_refused() {
        let dir = std::env::temp_dir().join(format!("qwen-ctx-{}", std::process::id()));
        let outside = dir.join("outside");
        let root = dir.join("root");
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(outside.join("secret.txt"), "classified").unwrap();

        #[cfg(unix)]
        {
            let link = root.join("escape");
            let _ = std::fs::remove_file(&link);
            std::os::unix::fs::symlink(&outside, &link).unwrap();

            let context = ToolContext::new(&root, WorkerRole::Coder);
            let err = context
                .resolve_in_workspace("escape/secret.txt", "workspace.read_file")
                .unwrap_err();
            assert!(err.contains("outside the workspace"), "{err}");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_path_does_not_error_during_resolution() {
        // Containment is checked lexically; existence is the tool's business.
        let resolved = ctx()
            .resolve_in_workspace("does/not/exist.txt", "t")
            .unwrap();
        assert_eq!(resolved, Path::new("/work/project/does/not/exist.txt"));
    }

    #[test]
    fn permission_checks_follow_the_worker_profile() {
        use crate::tools::definition::Permission;
        // The orchestrator has no tools at all.
        let orchestrator = ToolContext::new(".", WorkerRole::Orchestrator);
        assert!(
            orchestrator
                .check_permission(Permission::ReadOnly, "calculator.evaluate")
                .is_err()
        );

        // The researcher may read, not write.
        let researcher = ToolContext::new(".", WorkerRole::Researcher);
        assert!(
            researcher
                .check_permission(Permission::WorkspaceRead, "workspace.list_files")
                .is_ok()
        );
        assert!(
            researcher
                .check_permission(Permission::WorkspaceWrite, "workspace.list_files")
                .is_err()
        );

        // The coder may write.
        let coder = ToolContext::new(".", WorkerRole::Coder);
        assert!(
            coder
                .check_permission(Permission::WorkspaceWrite, "workspace.list_files")
                .is_ok()
        );
    }

    #[test]
    fn permission_check_reports_the_role() {
        let context = ToolContext::new(".", WorkerRole::Researcher);
        let err = context
            .check_permission(crate::tools::definition::Permission::ReadOnly, "shell.exec")
            .unwrap_err();
        assert!(err.contains("researcher"), "{err}");
    }
}
