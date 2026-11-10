//! `workspace.list_files` — read-only directory listing inside the workspace.
//!
//! Design constraints that matter more than the feature:
//!
//! * **containment** — resolution goes through
//!   [`ToolContext::resolve_in_workspace`], so `..`, absolute paths and escaping
//!   symlinks are refused before anything touches the disk;
//! * **bounded output** — a 4B model with a few thousand tokens of context cannot
//!   read a `node_modules` tree, so entries are capped, the walk is capped, and
//!   `truncated` is reported honestly rather than silently;
//! * **no surprises** — dotfiles are included (users ask "what is in this
//!   directory" and expect `.gitignore` to be there), but the walk never follows
//!   symlinked directories, which is how listings explode or loop;
//! * **read-only** — `Permission::WorkspaceRead`, so roles that cannot mutate are
//!   allowed and the orchestrator (no tools) is not.

use crate::json::Value;
use crate::tools::Tool;
use crate::tools::context::ToolContext;
use crate::tools::definition::{Param, Permission, Schema, ToolDefinition};
use crate::tools::result::{ToolError, ToolResult};
use crate::tools::validation::ValidatedArgs;
use std::path::Path;

/// Most entries returned in one call.
pub const MAX_ENTRIES: usize = 200;
/// Deepest directory level returned (1 = the listing of `path` itself).
pub const MAX_DEPTH: i64 = 3;

/// The workspace listing tool.
#[derive(Debug, Default, Clone, Copy)]
pub struct ListFiles;

impl Tool for ListFiles {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            "workspace.list_files",
            "List files and directories in the project workspace. Use this to \
             discover what a project contains. Paths are relative to the \
             workspace root; `.` means the whole project. Read-only.",
            vec![
                Param::optional(
                    "path",
                    Schema::string_of_len(
                        "Directory to list, relative to the workspace root. Defaults to `.`.",
                        1,
                        200,
                    ),
                ),
                Param::optional(
                    "depth",
                    Schema::Integer {
                        description: format!(
                            "How many directory levels to descend (1-{MAX_DEPTH}); 1 lists only the given directory."
                        ),
                        min: Some(1),
                        max: Some(MAX_DEPTH),
                    },
                ),
            ],
            Permission::WorkspaceRead,
        )
    }

    fn execute(
        &self,
        args: &ValidatedArgs,
        context: &ToolContext,
    ) -> Result<ToolResult, ToolError> {
        let tool = "workspace.list_files";
        let requested = args.str("path").unwrap_or(".");
        let depth = args.integer("depth").unwrap_or(1);

        let root = context
            .resolve_in_workspace(requested, tool)
            .map_err(|reason| ToolError::Rejected {
                tool: tool.into(),
                reason,
            })?;

        if !root.exists() {
            return Err(ToolError::Rejected {
                tool: tool.into(),
                reason: format!("`{requested}` does not exist in the workspace"),
            });
        }
        if !root.is_dir() {
            return Err(ToolError::Rejected {
                tool: tool.into(),
                reason: format!("`{requested}` is a file, not a directory"),
            });
        }

        let mut entries: Vec<Entry> = Vec::new();
        let mut truncated = false;
        // Listing `.` shows bare names; listing `src` shows `src/main.rs`, so
        // the model can copy a returned path straight into the next call.
        let prefix = if requested == "." || requested.is_empty() {
            String::new()
        } else {
            requested.trim_end_matches('/').to_string()
        };
        collect(&root, &prefix, depth, &mut entries, &mut truncated).map_err(|reason| {
            ToolError::ExecutionFailed {
                tool: tool.into(),
                reason,
            }
        })?;

        let names: Vec<Value> = entries.iter().map(|e| Value::str(&e.path)).collect();
        let directories = entries.iter().filter(|e| e.is_dir).count();
        let files = entries.len() - directories;

        let summary = if truncated {
            format!("{} entries (truncated at {MAX_ENTRIES})", entries.len())
        } else {
            format!(
                "{} entr{} ({files} file{}, {directories} director{})",
                entries.len(),
                if entries.len() == 1 { "y" } else { "ies" },
                if files == 1 { "" } else { "s" },
                if directories == 1 { "y" } else { "ies" }
            )
        };

        let mut data = vec![
            ("path".to_string(), Value::str(requested)),
            ("entries".to_string(), Value::Array(names)),
            (
                "directories".to_string(),
                Value::Array(
                    entries
                        .iter()
                        .filter(|e| e.is_dir)
                        .map(|e| Value::str(&e.path))
                        .collect(),
                ),
            ),
            ("file_count".to_string(), Value::Number(files as f64)),
            (
                "directory_count".to_string(),
                Value::Number(directories as f64),
            ),
            ("truncated".to_string(), Value::Bool(truncated)),
        ];
        if truncated {
            data.push((
                "note".to_string(),
                Value::str(format!(
                    "only the first {MAX_ENTRIES} entries are shown; list a subdirectory for more"
                )),
            ));
        }

        Ok(ToolResult::success(tool, Value::Object(data), summary))
    }
}

/// One listing row.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Entry {
    /// Path relative to the workspace root, using `/` separators.
    path: String,
    is_dir: bool,
}

/// Depth-limited, symlink-avoiding walk.
fn collect(
    dir: &Path,
    display_prefix: &str,
    remaining_depth: i64,
    out: &mut Vec<Entry>,
    truncated: &mut bool,
) -> Result<(), String> {
    // `display_prefix` is empty for the workspace root, `src` for a subdirectory.
    let read =
        std::fs::read_dir(dir).map_err(|e| format!("cannot read `{}`: {e}", dir.display()))?;
    let mut children: Vec<(String, bool)> = Vec::new();
    for entry in read {
        let entry = entry.map_err(|e| format!("cannot read an entry: {e}"))?;
        let name = entry.file_name().to_string_lossy().to_string();
        // `file_type` does not follow symlinks, which is what we want here.
        let file_type = entry
            .file_type()
            .map_err(|e| format!("cannot stat `{name}`: {e}"))?;
        children.push((name, file_type.is_dir()));
    }
    // Directories first, then alphabetical: the order a human would list.
    children.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    for (name, is_dir) in children {
        if out.len() >= MAX_ENTRIES {
            *truncated = true;
            return Ok(());
        }
        let display = if display_prefix.is_empty() {
            name.clone()
        } else {
            format!("{display_prefix}/{name}")
        };
        out.push(Entry {
            path: display.clone(),
            is_dir,
        });
        if is_dir && remaining_depth > 1 {
            // Recurse with the real path so symlinked dirs are not followed
            // (they were reported as directories above but never descended).
            let next = dir.join(&name);
            if next.is_symlink() {
                continue;
            }
            collect(&next, &display, remaining_depth - 1, out, truncated)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::planner::WorkerRole;

    /// Builds a small tree: `src/{main.rs, lib.rs}`, `tests/it.rs`, `README.md`.
    fn fixture(name: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!("qwen-ws-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::create_dir_all(root.join("tests")).unwrap();
        std::fs::write(root.join("README.md"), "hi").unwrap();
        std::fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();
        std::fs::write(root.join("src/lib.rs"), "// lib").unwrap();
        std::fs::write(root.join("tests/it.rs"), "#[test] fn t() {}").unwrap();
        std::fs::write(root.join(".gitignore"), "/target").unwrap();
        root
    }

    fn run(root: &Path, args: &str) -> ToolResult {
        let values = crate::json::parse(args).unwrap();
        let validated = crate::tools::validation::validate(&ListFiles.definition(), &values)
            .expect("args must validate");
        let context = ToolContext::new(root, WorkerRole::Researcher);
        ListFiles
            .execute(&validated, &context)
            .unwrap_or_else(|e| panic!("tool failed: {e}"))
    }

    fn entries(result: &ToolResult) -> Vec<String> {
        result
            .data
            .as_ref()
            .unwrap()
            .get("entries")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn lists_the_top_level_by_default() {
        let root = fixture("top");
        let result = run(&root, "{}");
        let entries = entries(&result);
        assert_eq!(
            entries,
            vec!["src", "tests", ".gitignore", "README.md"],
            "directories first, then files, alphabetical"
        );
        // Directories are identified.
        let data = result.data.as_ref().unwrap();
        assert_eq!(data.get("directory_count").unwrap().as_f64(), Some(2.0));
        assert_eq!(data.get("file_count").unwrap().as_f64(), Some(2.0));
        assert_eq!(data.get("truncated").unwrap().as_bool(), Some(false));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn descends_when_asked() {
        let root = fixture("depth");
        let result = run(&root, r#"{"depth":2}"#);
        let entries = entries(&result);
        assert!(entries.contains(&"src".to_string()));
        assert!(entries.contains(&"src/main.rs".to_string()), "{entries:?}");
        assert!(entries.contains(&"tests/it.rs".to_string()), "{entries:?}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn lists_a_subdirectory() {
        let root = fixture("sub");
        let result = run(&root, r#"{"path":"src"}"#);
        let entries = entries(&result);
        assert_eq!(
            entries,
            vec!["src/lib.rs", "src/main.rs"],
            "paths stay usable as the next call's `path`"
        );
        assert_eq!(
            result.data.as_ref().unwrap().get("path").unwrap().as_str(),
            Some("src")
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_summary_is_informative() {
        let root = fixture("summary");
        let result = run(&root, "{}");
        assert!(result.summary.contains("4 entries"), "{}", result.summary);
        assert!(result.summary.contains("file"), "{}", result.summary);
        assert!(result.summary.contains("director"), "{}", result.summary);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn traversal_outside_the_workspace_is_refused_without_touching_the_disk() {
        let root = fixture("escape");
        let values = crate::json::parse(r#"{"path":"../"}"#).unwrap();
        let validated =
            crate::tools::validation::validate(&ListFiles.definition(), &values).unwrap();
        let err = ListFiles
            .execute(&validated, &ToolContext::new(&root, WorkerRole::Researcher))
            .unwrap_err();
        assert_eq!(err.kind(), "rejected");
        assert!(err.to_string().contains("escapes"), "{err}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn absolute_paths_are_refused() {
        let root = fixture("abs");
        // `/etc` is not `Path::is_absolute` on Windows but must still be
        // refused, so this asserts the refusal *reason*, not the platform's
        // notion of absoluteness.
        #[cfg_attr(not(windows), allow(unused_mut))]
        let mut cases = vec!["/etc", "/"];
        #[cfg(windows)]
        cases.extend([r"\\server\share", r"C:\Windows"]);
        for path in cases {
            let values = crate::json::parse(&format!(
                r#"{{"path":{}}}"#,
                crate::json::to_string(&Value::str(path))
            ))
            .unwrap();
            let validated =
                crate::tools::validation::validate(&ListFiles.definition(), &values).unwrap();
            let err = ListFiles
                .execute(&validated, &ToolContext::new(&root, WorkerRole::Researcher))
                .unwrap_err();
            assert_eq!(err.kind(), "rejected");
            assert!(err.to_string().contains("absolute"), "{path}: {err}");
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn missing_paths_are_reported_not_invented() {
        let root = fixture("missing");
        let values = crate::json::parse(r#"{"path":"nope"}"#).unwrap();
        let validated =
            crate::tools::validation::validate(&ListFiles.definition(), &values).unwrap();
        let err = ListFiles
            .execute(&validated, &ToolContext::new(&root, WorkerRole::Researcher))
            .unwrap_err();
        assert!(err.to_string().contains("does not exist"), "{err}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_file_is_not_a_directory() {
        let root = fixture("file");
        let values = crate::json::parse(r#"{"path":"README.md"}"#).unwrap();
        let validated =
            crate::tools::validation::validate(&ListFiles.definition(), &values).unwrap();
        let err = ListFiles
            .execute(&validated, &ToolContext::new(&root, WorkerRole::Researcher))
            .unwrap_err();
        assert!(err.to_string().contains("not a directory"), "{err}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn output_is_capped_and_says_so() {
        let root = std::env::temp_dir().join(format!("qwen-ws-{}-wide", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        for i in 0..(MAX_ENTRIES + 25) {
            std::fs::write(root.join(format!("file{i:03}.txt")), "x").unwrap();
        }
        let result = run(&root, "{}");
        let entries = entries(&result);
        assert_eq!(entries.len(), MAX_ENTRIES);
        let data = result.data.as_ref().unwrap();
        assert_eq!(data.get("truncated").unwrap().as_bool(), Some(true));
        assert!(data.get("note").is_some());
        assert!(result.summary.contains("truncated"), "{}", result.summary);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_model_receives_structured_json() {
        let root = fixture("model");
        let result = run(&root, "{}");
        let content = crate::json::parse(&result.to_model_content()).unwrap();
        assert_eq!(content.get("ok").unwrap().as_bool(), Some(true));
        let payload = content.get("result").unwrap();
        assert!(payload.get("entries").unwrap().as_array().unwrap().len() >= 4);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_definition_is_read_only_and_bounded() {
        let definition = ListFiles.definition();
        assert_eq!(definition.permission, Permission::WorkspaceRead);
        assert!(definition.required_param_names().is_empty());
        assert_eq!(definition.params.len(), 2);
        // Bounds are declared, so the validator enforces them.
        let schema = &definition.param("depth").unwrap().schema;
        let json = schema.to_json();
        assert_eq!(json.get("minimum").unwrap().as_f64(), Some(1.0));
        assert_eq!(
            json.get("maximum").unwrap().as_f64(),
            Some(MAX_DEPTH as f64)
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_directories_are_listed_but_not_descended() {
        let root = fixture("symlink");
        let outside = std::env::temp_dir().join(format!("qwen-ws-{}-outside", std::process::id()));
        let _ = std::fs::remove_dir_all(&outside);
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("secret.txt"), "classification: none").unwrap();
        let link = root.join("linked");
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(&outside, &link).unwrap();

        let result = run(&root, r#"{"depth":3}"#);
        let entries = entries(&result);
        assert!(entries.contains(&"linked".to_string()), "{entries:?}");
        assert!(
            !entries.iter().any(|e| e.contains("secret.txt")),
            "must not descend through a symlink: {entries:?}"
        );

        let _ = std::fs::remove_file(&link);
        let _ = std::fs::remove_dir_all(&outside);
        let _ = std::fs::remove_dir_all(&root);
    }
}
