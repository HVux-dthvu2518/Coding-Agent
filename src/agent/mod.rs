//! Agent intelligence: context, state, planning, reflection, orchestration.
//!
//! M0 implements the conversational path end to end. `planner` and
//! `reflection` are present as typed seams so M2 adds behaviour without
//! reshaping the core.

pub mod context;
pub mod orchestrator;
pub mod planner;
pub mod reflection;
pub mod state;
pub mod workers;

pub use context::Context;
pub use orchestrator::{Event, LastTurn, Orchestrator, TurnOutcome};
pub use state::{AgentState, StateMachine, TransitionError};
pub use workers::WorkerProfile;
