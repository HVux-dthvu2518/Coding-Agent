//! Planning — an M2 slot that is already wired in.
//!
//! The architectural rule from the spec is *"don't force every request through a
//! planner"*: a question like "what does `?` do in Rust" must go straight to the
//! model. This module therefore ships only the **plan data model** and a
//! deterministic [`should_plan`] heuristic, so M2 can add the LLM-driven planner
//! behind the same types and the orchestrator needs no restructuring.
//!
//! In M0 the orchestrator calls [`should_plan`] only to decide whether to note,
//! in debug logs, that a request *would* have been planned. Nothing branches on
//! it yet — that is M2.

/// One step of a plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanStep {
    /// 1-based position.
    pub index: usize,
    pub description: String,
    /// Which worker role should run it (see `workers.rs`).
    pub worker: WorkerRole,
    /// Set once the step finished successfully.
    pub done: bool,
}

/// Logical worker roles from the hybrid architecture.
///
/// All roles share one model instance, so these are prompt/tool scoping
/// conveniences rather than separate agents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerRole {
    /// The orchestrator itself: conversation and simple answers.
    Orchestrator,
    Researcher,
    Coder,
    Reviewer,
}

impl WorkerRole {
    pub fn as_str(self) -> &'static str {
        match self {
            WorkerRole::Orchestrator => "orchestrator",
            WorkerRole::Researcher => "researcher",
            WorkerRole::Coder => "coder",
            WorkerRole::Reviewer => "reviewer",
        }
    }
}

/// A decomposition of an objective.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Plan {
    pub objective: String,
    pub steps: Vec<PlanStep>,
}

impl Plan {
    pub fn new(objective: impl Into<String>) -> Plan {
        Plan {
            objective: objective.into(),
            steps: Vec::new(),
        }
    }

    pub fn with_steps(objective: impl Into<String>, steps: Vec<PlanStep>) -> Plan {
        Plan {
            objective: objective.into(),
            steps,
        }
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// The first unfinished step.
    pub fn next_step(&self) -> Option<&PlanStep> {
        self.steps.iter().find(|s| !s.done)
    }

    pub fn is_complete(&self) -> bool {
        !self.steps.is_empty() && self.steps.iter().all(|s| s.done)
    }

    pub fn completed(&self) -> usize {
        self.steps.iter().filter(|s| s.done).count()
    }

    /// A compact, human-readable rendering for the REPL and logs.
    pub fn render(&self) -> String {
        let mut out = format!("Plan: {}\n", self.objective);
        for step in &self.steps {
            out.push_str(&format!(
                "  {}. [{}] {}\n",
                step.index,
                if step.done { "x" } else { " " },
                step.description
            ));
        }
        out
    }
}

/// Words that usually indicate a multi-step objective rather than a question.
const MULTI_STEP_MARKERS: &[&str] = &[
    "then",
    "after that",
    "finally",
    "step 1",
    "first,",
    "second,",
    "and also",
    "make sure",
    "verify",
    "refactor",
    "implement",
    "migrate",
    "set up",
    "debug",
    "investigate",
    "and test",
    "write tests",
];

/// Markers that indicate a plain question, which must bypass planning.
const QUESTION_PREFIXES: &[&str] = &[
    "what", "why", "how", "when", "where", "who", "which", "is ", "are ", "can ", "does ", "do ",
    "should ", "explain", "tell me", "define", "compare",
];

/// Heuristic: should this request go through the planner?
///
/// Deterministic and cheap so it can run on every message. Deliberately
/// conservative — a false negative just means the orchestrator answers
/// directly, which is the M0 behaviour anyway.
pub fn should_plan(request: &str) -> bool {
    let text = request.trim().to_ascii_lowercase();
    if text.is_empty() {
        return false;
    }

    let is_question = QUESTION_PREFIXES.iter().any(|p| text.starts_with(p));
    let marker_hits = MULTI_STEP_MARKERS
        .iter()
        .filter(|m| text.contains(**m))
        .count();

    // A question needs strong multi-step evidence to be planned ("how do I X,
    // then Y, then verify Z"); a bare question never is.
    let word_count = text.split_whitespace().count();
    if is_question {
        return marker_hits >= 2 && word_count >= 12;
    }
    marker_hits >= 1 || (text.ends_with('.') && word_count >= 20 && contains_verb(&text))
}

fn contains_verb(text: &str) -> bool {
    [
        "build", "create", "add", "fix", "change", "update", "run", "generate", "review",
    ]
    .iter()
    .any(|v| text.contains(v))
}

/// Helper for M2: renumbers steps and clears completion flags.
pub fn renumber(mut plan: Plan) -> Plan {
    for (i, step) in plan.steps.iter_mut().enumerate() {
        step.index = i + 1;
        step.done = false;
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(i: usize, desc: &str, worker: WorkerRole) -> PlanStep {
        PlanStep {
            index: i,
            description: desc.into(),
            worker,
            done: false,
        }
    }

    #[test]
    fn questions_bypass_the_planner() {
        assert!(!should_plan("What is Rust ownership?"));
        assert!(!should_plan("explain the borrow checker"));
        assert!(!should_plan("How do I read a file in Rust?"));
        assert!(!should_plan(""));
    }

    #[test]
    fn multi_step_tasks_are_planned() {
        assert!(should_plan("Read src/main.rs, then summarise it"));
        assert!(should_plan("Implement a retry policy and test it"));
        assert!(should_plan("Fix the parser bug and also add tests"));
    }

    #[test]
    fn long_questions_with_two_markers_are_planned() {
        assert!(should_plan(
            "How do I refactor this module so that it compiles, then verify the tests still pass?"
        ));
    }

    #[test]
    fn plan_progress_tracking() {
        let mut plan = Plan::with_steps(
            "add a health endpoint",
            vec![
                step(1, "read the router", WorkerRole::Coder),
                step(2, "add /health", WorkerRole::Coder),
                step(3, "test it", WorkerRole::Reviewer),
            ],
        );
        assert!(!plan.is_complete());
        assert_eq!(plan.next_step().unwrap().description, "read the router");
        plan.steps[0].done = true;
        assert_eq!(plan.next_step().unwrap().index, 2);
        assert_eq!(plan.completed(), 1);
        plan.steps[1].done = true;
        plan.steps[2].done = true;
        assert!(plan.is_complete());
        assert_eq!(plan.len(), 3);
    }

    #[test]
    fn render_is_readable() {
        let mut plan = Plan::with_steps("obj", vec![step(1, "first", WorkerRole::Orchestrator)]);
        plan.steps[0].done = true;
        let text = plan.render();
        assert!(text.contains("Plan: obj"));
        assert!(text.contains("[x] first"), "{text}");
    }

    #[test]
    fn renumber_resets_completion() {
        let mut plan = Plan::with_steps("o", vec![step(9, "a", WorkerRole::Coder)]);
        plan.steps[0].done = true;
        let plan = renumber(plan);
        assert_eq!(plan.steps[0].index, 1);
        assert!(!plan.steps[0].done);
    }

    #[test]
    fn empty_plan_is_not_complete() {
        assert!(!Plan::new("nothing").is_complete());
        assert!(Plan::new("nothing").is_empty());
    }
}
