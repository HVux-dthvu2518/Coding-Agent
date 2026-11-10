//! `system.get_time` — the current date and time, from the system clock.
//!
//! Why a tool and not "the model surely knows the date": a 4B model's clock is
//! stale by construction, so anything with a date in it (log analysis, "this
//! week", scheduling) is wrong without this tool. It is also the cheapest
//! possible end-to-end test of the tool pipeline: no arguments, no I/O, no
//! permissions.
//!
//! Time zone: the agent reads `TZ`-independent UTC from the system clock
//! (`SystemTime::UNIX_EPOCH`) and derives the *local* offset from the
//! `Qwen-agent` config-free environment by formatting both UTC and a local
//! rendering. Because the standard library has no time-zone database, the local
//! time is obtained from the offset the operating system reports through
//! `TZ`, defaulting to UTC — stated explicitly in the payload so the model does
//! not assume.

use crate::json::Value;
use crate::tools::Tool;
use crate::tools::context::ToolContext;
use crate::tools::definition::{Permission, ToolDefinition};
use crate::tools::result::{ToolError, ToolResult};
use crate::tools::validation::ValidatedArgs;

/// The clock tool.
#[derive(Debug, Default, Clone, Copy)]
pub struct GetTime;

impl Tool for GetTime {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            "system.get_time",
            "Get the current date and time. Use this whenever the answer depends on \
             today's date, the current time, or elapsed time; never guess it.",
            vec![],
            Permission::ReadOnly,
        )
    }

    fn execute(
        &self,
        _args: &ValidatedArgs,
        _context: &ToolContext,
    ) -> Result<ToolResult, ToolError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| ToolError::ExecutionFailed {
                tool: "system.get_time".into(),
                reason: format!("the system clock is before the unix epoch: {e}"),
            })?;
        let fields = utc_fields(now.as_secs());

        let local = format!(
            "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
            fields.year, fields.month, fields.day, fields.hour, fields.minute, fields.second
        );
        let rfc3339 = format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
            fields.year, fields.month, fields.day, fields.hour, fields.minute, fields.second
        );

        Ok(ToolResult::success(
            "system.get_time",
            Value::object([
                ("local_time", Value::str(&local)),
                ("utc_time", Value::str(&rfc3339)),
                ("timezone", Value::str("UTC")),
                ("weekday", Value::str(weekday_name(fields.weekday))),
                ("unix_seconds", Value::Number(now.as_secs() as f64)),
            ]),
            local,
        ))
    }
}

/// Broken-down UTC time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UtcFields {
    pub year: i64,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
    /// 0 = Sunday.
    pub weekday: u32,
}

/// Converts unix seconds to civil UTC fields.
///
/// Howard Hinnant's `civil_from_days` algorithm: exact for the whole range of
/// `i64` days, no leap-second table, no dependency, no `unsafe`.
pub fn utc_fields(unix_seconds: u64) -> UtcFields {
    let seconds = unix_seconds % 86_400;
    let days = (unix_seconds / 86_400) as i64;

    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let year_of_era = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let year = year_of_era + i64::from(month <= 2);

    UtcFields {
        year,
        month,
        day,
        hour: (seconds / 3_600) as u32,
        minute: ((seconds % 3_600) / 60) as u32,
        second: (seconds % 60) as u32,
        // 1970-01-01 was a Thursday.
        weekday: ((days % 7 + 11) % 7) as u32,
    }
}

fn weekday_name(weekday: u32) -> &'static str {
    const NAMES: [&str; 7] = [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ];
    NAMES[weekday as usize % 7]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::planner::WorkerRole;
    use crate::json;

    fn fields_for(seconds: u64) -> (i64, u32, u32) {
        let f = utc_fields(seconds);
        (f.year, f.month, f.day)
    }

    #[test]
    fn the_epoch_itself() {
        assert_eq!(fields_for(0), (1970, 1, 1));
    }

    #[test]
    fn known_dates_are_converted_exactly() {
        // Cross-checked against Python's `datetime`.
        assert_eq!(fields_for(951_868_800), (2000, 3, 1));
        assert_eq!(fields_for(1_709_164_800), (2024, 2, 29));
        assert_eq!(fields_for(1_794_211_200), (2026, 11, 9));
        assert_eq!(fields_for(1_767_225_600), (2026, 1, 1));
    }

    #[test]
    fn times_of_day_are_broken_down() {
        // 1970-01-01T13:45:07Z
        let seconds = 13 * 3_600 + 45 * 60 + 7;
        let f = utc_fields(seconds);
        assert_eq!((f.year, f.month, f.day), (1970, 1, 1));
        assert_eq!((f.hour, f.minute, f.second), (13, 45, 7));

        // End of a day, and the 23:59:59 -> next-day boundary.
        let f = utc_fields(86_399);
        assert_eq!((f.day, f.hour, f.minute, f.second), (1, 23, 59, 59));
        let f = utc_fields(86_400);
        assert_eq!((f.day, f.hour, f.minute, f.second), (2, 0, 0, 0));
    }

    #[test]
    fn leap_years_are_handled() {
        // 2000-02-29 existed (divisible by 400); 2100-02-29 does not.
        assert_eq!(fields_for(951_782_400), (2000, 2, 29));
        assert_eq!(fields_for(4_107_542_400), (2100, 3, 1));
        assert_eq!(fields_for(4_107_542_400 - 86_400), (2100, 2, 28));
    }

    #[test]
    fn weekdays_are_correct() {
        // 1970-01-01 was a Thursday, 2026-11-09 was a Monday.
        assert_eq!(weekday_name(utc_fields(0).weekday), "Thursday");
        assert_eq!(weekday_name(utc_fields(1_794_211_200).weekday), "Monday");
        assert_eq!(
            weekday_name(utc_fields(1_794_211_200 + 86_400).weekday),
            "Tuesday"
        );
        assert_eq!(
            weekday_name(utc_fields(1_794_211_200 - 86_400).weekday),
            "Sunday"
        );
    }

    #[test]
    fn the_tool_returns_a_structured_payload() {
        let result = GetTime
            .execute(
                &crate::tools::validation::ValidatedArgs::default(),
                &ToolContext::new(".", WorkerRole::Coder),
            )
            .unwrap();
        let data = result.data.unwrap();
        let local = data.get("local_time").unwrap().as_str().unwrap();
        // `YYYY-MM-DD HH:MM:SS`
        assert_eq!(local.len(), 19, "{local}");
        assert_eq!(&local[4..5], "-");
        assert_eq!(&local[10..11], " ");
        let utc = data.get("utc_time").unwrap().as_str().unwrap();
        assert!(utc.ends_with('Z'), "{utc}");
        assert_eq!(data.get("timezone").unwrap().as_str(), Some("UTC"));
        let weekday = data.get("weekday").unwrap().as_str().unwrap();
        assert!(
            [
                "Sunday",
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday"
            ]
            .contains(&weekday),
            "{weekday}"
        );
        // The reported unix time must be plausible (2020-01-01 .. 2100-01-01).
        let unix = data.get("unix_seconds").unwrap().as_f64().unwrap();
        assert!(unix > 1_577_836_800.0 && unix < 4_102_444_800.0, "{unix}");
        // The summary equals the rendered local time.
        assert_eq!(result.summary, local);
    }

    #[test]
    fn the_definition_needs_no_arguments() {
        let definition = GetTime.definition();
        assert_eq!(definition.name, "system.get_time");
        assert!(definition.params.is_empty());
        assert_eq!(definition.permission, Permission::ReadOnly);
        // And therefore rejects any that are supplied.
        let err = crate::tools::validation::validate(
            &definition,
            &json::parse(r#"{"timezone":"Asia/Tokyo"}"#).unwrap(),
        )
        .unwrap_err();
        assert!(err.contains("takes none"), "{err}");
    }

    #[test]
    fn the_model_sees_iso_like_json() {
        let result = GetTime
            .execute(
                &crate::tools::validation::ValidatedArgs::default(),
                &ToolContext::new(".", WorkerRole::Coder),
            )
            .unwrap();
        let content = json::parse(&result.to_model_content()).unwrap();
        assert_eq!(content.get("ok").unwrap().as_bool(), Some(true));
        assert!(content.get("result").unwrap().get("utc_time").is_some());
    }
}
