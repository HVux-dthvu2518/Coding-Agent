//! Command-line interface: argument parsing, REPL, styling.

pub mod args;
pub mod commands;
pub mod doctor;
pub mod repl;
pub mod ui;

pub use args::{Cli, Command as CliCommand};
pub use repl::{Session, banner_lines, context_for};
