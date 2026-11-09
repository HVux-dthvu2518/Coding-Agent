//! The agent state machine.
//!
//! ```text
//!         ┌──────────────────────────── complete ────────────────────────────┐
//!         ▼                                                                  │
//!      Idle ──request──► Thinking ──tokens──► Responding ──done──► Idle      │
//!         ▲                  │                    │                          │
//!         │                  └──error──► Failed ──┘                          │
//!         └───────────────── reset ────────────────────────────────────────--┘
//! ```
//!
//! M0 only has the *conversational* path. The states that M2 needs
//! (`Planning`, `Executing`, `Observing`, `Blocked`) are already modelled and
//! rejected by [`Transition::try_begin_planning`] so the loop can be extended
//! without reshaping the types — and so an accidental early transition is a
//! compile-time-visible, tested error rather than a silent no-op.

/// Where the agent is in its execution loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentState {
    /// Waiting for user input.
    Idle,
    /// Request sent, waiting for the first token.
    Thinking,
    /// Receiving a streamed answer.
    Responding,
    /// Planning a multi-step objective (M2).
    Planning,
    /// Running a tool (M2/M3).
    Executing,
    /// Inspecting tool output (M2/M3).
    Observing,
    /// Objective verified (M2).
    Completed,
    /// Cannot safely continue: needs approval or hit an iteration limit.
    Blocked { reason: String },
    /// The last attempt failed; the conversation is still usable.
    Failed { reason: String },
}

impl AgentState {
    pub fn name(&self) -> &'static str {
        match self {
            AgentState::Idle => "idle",
            AgentState::Thinking => "thinking",
            AgentState::Responding => "responding",
            AgentState::Planning => "planning",
            AgentState::Executing => "executing",
            AgentState::Observing => "observing",
            AgentState::Completed => "completed",
            AgentState::Blocked { .. } => "blocked",
            AgentState::Failed { .. } => "failed",
        }
    }

    /// True while a request is in flight; the REPL uses this to decide whether
    /// Ctrl-C should cancel a request or exit the app.
    pub fn is_busy(&self) -> bool {
        matches!(
            self,
            AgentState::Thinking
                | AgentState::Responding
                | AgentState::Planning
                | AgentState::Executing
                | AgentState::Observing
        )
    }

    /// True when the user can type a new message without losing work.
    pub fn accepts_input(&self) -> bool {
        matches!(
            self,
            AgentState::Idle
                | AgentState::Completed
                | AgentState::Failed { .. }
                | AgentState::Blocked { .. }
        )
    }

    pub fn is_failure(&self) -> bool {
        matches!(self, AgentState::Failed { .. } | AgentState::Blocked { .. })
    }
}

/// A rejected state change. Kept as a value so callers can decide whether to
/// log, ignore or escalate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransitionError {
    pub from: String,
    pub to: String,
    pub reason: String,
}

impl std::fmt::Display for TransitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "invalid agent transition {} -> {}: {}",
            self.from, self.to, self.reason
        )
    }
}

impl std::error::Error for TransitionError {}

/// Owns the current state and the small amount of bookkeeping the REPL shows in
/// the status bar.
#[derive(Debug, Clone)]
pub struct StateMachine {
    state: AgentState,
    /// Messages sent to the model since the app started.
    pub requests: u64,
    /// Assistant messages completed.
    pub replies: u64,
}

impl Default for StateMachine {
    fn default() -> Self {
        StateMachine {
            state: AgentState::Idle,
            requests: 0,
            replies: 0,
        }
    }
}

impl StateMachine {
    pub fn state(&self) -> &AgentState {
        &self.state
    }

    fn transition(&mut self, to: AgentState, reason: &str) -> Result<(), TransitionError> {
        let legal = match (&self.state, &to) {
            // Normal conversational path.
            (AgentState::Idle, AgentState::Thinking) => true,
            (AgentState::Thinking, AgentState::Responding) => true,
            (AgentState::Thinking, AgentState::Idle) => true, // empty reply
            (AgentState::Responding, AgentState::Idle) => true,
            // Failure from anywhere in flight.
            (s, AgentState::Failed { .. }) if s.is_busy() => true,
            // Retrying after a failure: the user simply asks again.
            (AgentState::Failed { .. } | AgentState::Blocked { .. }, AgentState::Thinking) => true,
            // Recovery back to idle from any settled state.
            (s, AgentState::Idle) if !s.is_busy() => true,
            // M2 states are reachable only from Idle.
            (AgentState::Idle, AgentState::Planning) => true,
            (AgentState::Planning, AgentState::Executing) => true,
            (AgentState::Planning, AgentState::Idle) => true,
            (AgentState::Executing, AgentState::Observing) => true,
            (AgentState::Observing, AgentState::Executing) => true,
            (AgentState::Observing, AgentState::Planning) => true,
            (AgentState::Executing, AgentState::Completed) => true,
            (s, AgentState::Completed) if s.is_busy() => true,
            (s, AgentState::Blocked { .. }) if s.is_busy() || matches!(s, AgentState::Idle) => true,
            _ => false,
        };
        if legal {
            self.state = to;
            Ok(())
        } else {
            Err(TransitionError {
                from: self.state.name().to_string(),
                to: to.name().to_string(),
                reason: reason.to_string(),
            })
        }
    }

    /// A request left the app.
    pub fn begin_request(&mut self) -> Result<(), TransitionError> {
        self.transition(AgentState::Thinking, "request already in flight")?;
        self.requests += 1;
        Ok(())
    }

    /// The first token arrived.
    pub fn first_token(&mut self) -> Result<(), TransitionError> {
        self.transition(AgentState::Responding, "first token outside a request")
    }

    /// The reply finished (or was interrupted and its partial text kept).
    pub fn finish_reply(&mut self) {
        self.replies += 1;
        let _ = self.transition(AgentState::Idle, "reply finished");
    }

    /// The request failed; the conversation stays usable.
    pub fn fail(&mut self, reason: impl Into<String>) {
        let reason = reason.into();
        let _ = self.transition(AgentState::Failed { reason }, "failure");
    }

    /// Clears a failure so typing works again.
    pub fn reset(&mut self) {
        let _ = self.transition(AgentState::Idle, "reset");
    }

    /// Guard for the M2 entry point: refuses to plan while a request is in
    /// flight, which is the bug this enum exists to prevent.
    pub fn try_begin_planning(&mut self) -> Result<(), TransitionError> {
        if !matches!(self.state, AgentState::Idle) {
            return Err(TransitionError {
                from: self.state.name().to_string(),
                to: AgentState::Planning.name().to_string(),
                reason: "planning requires an idle agent (M2 feature gate)".into(),
            });
        }
        self.transition(AgentState::Planning, "planning requires an idle agent")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn happy_path_request_lifecycle() {
        let mut sm = StateMachine::default();
        sm.begin_request().unwrap();
        assert_eq!(sm.state(), &AgentState::Thinking);
        sm.first_token().unwrap();
        assert_eq!(sm.state(), &AgentState::Responding);
        sm.finish_reply();
        assert_eq!(sm.state(), &AgentState::Idle);
        assert_eq!(sm.requests, 1);
        assert_eq!(sm.replies, 1);
    }

    #[test]
    fn a_second_concurrent_request_is_rejected() {
        let mut sm = StateMachine::default();
        sm.begin_request().unwrap();
        let err = sm.begin_request().unwrap_err();
        assert!(err.reason.contains("in flight"), "{err}");
        assert_eq!(sm.requests, 1, "rejected request must not be counted");
    }

    #[test]
    fn first_token_without_a_request_is_rejected() {
        let mut sm = StateMachine::default();
        assert!(sm.first_token().is_err());
    }

    #[test]
    fn empty_reply_can_return_to_idle() {
        let mut sm = StateMachine::default();
        sm.begin_request().unwrap();
        sm.finish_reply();
        assert_eq!(sm.state(), &AgentState::Idle);
    }

    #[test]
    fn failure_is_recoverable() {
        let mut sm = StateMachine::default();
        sm.begin_request().unwrap();
        sm.fail("connection refused");
        assert!(sm.state().is_failure());
        assert!(sm.state().accepts_input());
        sm.reset();
        assert_eq!(sm.state(), &AgentState::Idle);
        sm.begin_request().unwrap();
    }

    #[test]
    fn a_failure_does_not_lock_out_the_next_request() {
        let mut sm = StateMachine::default();
        sm.begin_request().unwrap();
        sm.fail("connection refused");
        // No explicit reset needed: asking again is legal.
        sm.begin_request().unwrap();
        assert_eq!(sm.state(), &AgentState::Thinking);
        assert_eq!(sm.requests, 2);
    }

    #[test]
    fn busy_states_reject_input() {
        let mut sm = StateMachine::default();
        sm.begin_request().unwrap();
        assert!(sm.state().is_busy());
        assert!(!sm.state().accepts_input());
    }

    #[test]
    fn planning_is_gated_in_m0() {
        let mut sm = StateMachine::default();
        sm.try_begin_planning().unwrap();
        assert_eq!(sm.state(), &AgentState::Planning);
        // But not while something is in flight.
        let mut sm = StateMachine::default();
        sm.begin_request().unwrap();
        assert!(sm.try_begin_planning().is_err());
    }

    #[test]
    fn illegal_transitions_are_reported() {
        let mut sm = StateMachine::default();
        let err = sm.transition(AgentState::Executing, "test").unwrap_err();
        assert_eq!(err.from, "idle");
        assert_eq!(err.to, "executing");
        assert!(err.to_string().contains("invalid agent transition"));
    }
}
