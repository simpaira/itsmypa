//! Upcoming meetings from the user's secret iCal (ICS) address — Google,
//! Outlook and Apple calendars all provide one. Read-only, no account, no
//! OAuth. The home screen polls this and offers to record when a meeting
//! starts. Port of app.py `_parse_ics` / `/calendar/upcoming`.

use anyhow::{Context, Result};
use chrono::{DateTime, Duration, NaiveDateTime, TimeZone, Utc};
use rrule::{RRule, Tz, Unvalidated};
use serde::Serialize;
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Event {
    pub title: String,
    /// RFC 3339 with offset, e.g. "2026-07-09T10:00:00+00:00".
    pub start: String,
    pub end: String,
}

/// Fetch the calendar and return meetings overlapping the last hour through
/// the next 24 hours, soonest first.
pub fn upcoming(url: &str) -> Result<Vec<Event>> {
    let body = ureq::get(url)
        .timeout(std::time::Duration::from_secs(15))
        .call()
        .context("Could not fetch the calendar")?
        .into_string()
        .context("The calendar isn't readable text")?;
    let now = Utc::now();
    Ok(parse_ics(&body, now - Duration::hours(1), now + Duration::hours(24)))
}

/// One property line: NAME;PARAM=V;…:VALUE
struct Prop {
    name: String,
    params: HashMap<String, String>,
    value: String,
}

fn parse_prop(line: &str) -> Option<Prop> {
    // The value starts at the first ':' outside a quoted parameter value.
    let mut in_quotes = false;
    let split = line.char_indices().find(|&(_, c)| {
        if c == '"' {
            in_quotes = !in_quotes;
        }
        c == ':' && !in_quotes
    })?;
    let (head, value) = (&line[..split.0], &line[split.0 + 1..]);
    let mut parts = head.split(';');
    let name = parts.next()?.to_ascii_uppercase();
    let params = parts
        .filter_map(|p| p.split_once('='))
        .map(|(k, v)| (k.to_ascii_uppercase(), v.trim_matches('"').to_string()))
        .collect();
    Some(Prop { name, params, value: value.to_string() })
}

fn unescape(text: &str) -> String {
    text.replace("\\n", " ").replace("\\N", " ").replace("\\,", ",").replace("\\;", ";").replace("\\\\", "\\")
}

/// A DTSTART-style value. None for all-day dates (nothing to record) or
/// values that can't be read. Unknown TZIDs (e.g. Outlook's Windows zone
/// names) are read as the computer's local time.
fn parse_time(p: &Prop) -> Option<DateTime<Tz>> {
    let v = p.value.trim();
    if p.params.get("VALUE").is_some_and(|x| x.eq_ignore_ascii_case("DATE")) || v.len() == 8 {
        return None;
    }
    if let Some(utc) = v.strip_suffix('Z') {
        let naive = NaiveDateTime::parse_from_str(utc, "%Y%m%dT%H%M%S").ok()?;
        return Some(Tz::UTC.from_utc_datetime(&naive));
    }
    let naive = NaiveDateTime::parse_from_str(v, "%Y%m%dT%H%M%S").ok()?;
    let tz = p
        .params
        .get("TZID")
        .and_then(|id| id.parse::<chrono_tz::Tz>().ok())
        .map(Tz::from)
        .unwrap_or(Tz::LOCAL);
    tz.from_local_datetime(&naive).earliest()
}

/// "PT45M", "PT1H30M", "P1D" → duration.
fn parse_duration(v: &str) -> Option<Duration> {
    let v = v.trim().trim_start_matches('+');
    let rest = v.strip_prefix('P')?;
    let (mut total, mut num, mut in_time) = (Duration::zero(), String::new(), false);
    for c in rest.chars() {
        match c {
            'T' => in_time = true,
            d if d.is_ascii_digit() => num.push(d),
            unit => {
                let n: i64 = num.parse().ok()?;
                num.clear();
                total += match (unit, in_time) {
                    ('W', false) => Duration::weeks(n),
                    ('D', false) => Duration::days(n),
                    ('H', true) => Duration::hours(n),
                    ('M', true) => Duration::minutes(n),
                    ('S', true) => Duration::seconds(n),
                    _ => return None,
                };
            }
        }
    }
    Some(total)
}

#[derive(Default)]
struct VEvent {
    props: Vec<Prop>,
}

impl VEvent {
    fn get(&self, name: &str) -> Option<&Prop> {
        self.props.iter().find(|p| p.name == name)
    }
    fn all<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Prop> + 'a {
        self.props.iter().filter(move |p| p.name == name)
    }
}

/// Every timed event occurrence overlapping [from, to], recurring events
/// expanded, soonest first. All-day and cancelled events are skipped.
pub fn parse_ics(data: &str, from: DateTime<Utc>, to: DateTime<Utc>) -> Vec<Event> {
    // Unfold continuation lines (RFC 5545 §3.1).
    let mut lines: Vec<String> = Vec::new();
    for raw in data.lines() {
        match raw.strip_prefix([' ', '\t']) {
            Some(cont) if !lines.is_empty() => lines.last_mut().unwrap().push_str(cont),
            _ => lines.push(raw.trim_end_matches('\r').to_string()),
        }
    }

    // Collect VEVENTs, ignoring nested components (alarms).
    let (mut events, mut cur, mut depth) = (Vec::new(), None::<VEvent>, 0);
    for line in &lines {
        let upper = line.to_ascii_uppercase();
        if upper == "BEGIN:VEVENT" {
            cur = Some(VEvent::default());
            depth = 0;
        } else if upper == "END:VEVENT" {
            events.extend(cur.take());
        } else if let Some(ev) = cur.as_mut() {
            if upper.starts_with("BEGIN:") {
                depth += 1;
            } else if upper.starts_with("END:") {
                depth -= 1;
            } else if depth == 0 {
                ev.props.extend(parse_prop(line));
            }
        }
    }

    // Moved or cancelled single instances of a recurring meeting
    // (RECURRENCE-ID) replace that instance of the series.
    let mut overridden: HashMap<String, Vec<DateTime<Tz>>> = HashMap::new();
    for ev in &events {
        if let (Some(uid), Some(rid)) = (ev.get("UID"), ev.get("RECURRENCE-ID").and_then(parse_time)) {
            overridden.entry(uid.value.clone()).or_default().push(rid);
        }
    }

    let (from_tz, to_tz) = (from.with_timezone(&Tz::UTC), to.with_timezone(&Tz::UTC));
    let mut out = Vec::new();
    for ev in &events {
        if ev.get("STATUS").is_some_and(|s| s.value.eq_ignore_ascii_case("CANCELLED")) {
            continue;
        }
        let Some(start) = ev.get("DTSTART").and_then(parse_time) else { continue };
        let length = ev
            .get("DTEND")
            .and_then(parse_time)
            .map(|end| end - start)
            .or_else(|| ev.get("DURATION").and_then(|d| parse_duration(&d.value)))
            .unwrap_or_else(|| Duration::hours(1));
        let title = ev.get("SUMMARY").map(|s| unescape(&s.value)).filter(|t| !t.trim().is_empty()).unwrap_or_else(|| "Untitled meeting".into());

        let starts: Vec<DateTime<Tz>> = match ev.get("RRULE").filter(|_| ev.get("RECURRENCE-ID").is_none()) {
            None => vec![start],
            Some(rule) => {
                let Ok(rule) = rule.value.parse::<RRule<Unvalidated>>() else { continue };
                let Ok(mut set) = rule.build(start) else { continue };
                for ex in ev.all("EXDATE") {
                    let tzid = ex.params.clone();
                    for v in ex.value.split(',') {
                        if let Some(t) = parse_time(&Prop { name: "EXDATE".into(), params: tzid.clone(), value: v.into() }) {
                            set = set.exdate(t);
                        }
                    }
                }
                if let Some(uid) = ev.get("UID") {
                    for t in overridden.get(&uid.value).into_iter().flatten() {
                        set = set.exdate(*t);
                    }
                }
                set.after(from_tz - length).before(to_tz).all(500).dates
            }
        };
        for s in starts {
            let e = s + length;
            if s < to_tz && e > from_tz {
                out.push((s, Event { title: title.clone(), start: s.fixed_offset().to_rfc3339(), end: e.fixed_offset().to_rfc3339() }));
            }
        }
    }
    out.sort_by_key(|(s, _)| *s);
    out.into_iter().map(|(_, e)| e).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn utc(y: i32, m: u32, d: u32) -> DateTime<Utc> {
        Utc.from_utc_datetime(&NaiveDate::from_ymd_opt(y, m, d).unwrap().and_hms_opt(0, 0, 0).unwrap())
    }

    // Port of test_parse_ics_expands_recurrence_and_skips_allday
    #[test]
    fn expands_recurrence_and_skips_allday() {
        let ics = "BEGIN:VCALENDAR\r\nVERSION:2.0\r\nPRODID:-//test//test//EN\r\nBEGIN:VEVENT\r\nUID:weekly-standup@test\r\nDTSTART:20260101T100000Z\r\nDTEND:20260101T101500Z\r\nRRULE:FREQ=WEEKLY\r\nSUMMARY:Weekly standup\r\nEND:VEVENT\r\nBEGIN:VEVENT\r\nUID:allday@test\r\nDTSTART;VALUE=DATE:20260706\r\nSUMMARY:All-day thing\r\nEND:VEVENT\r\nEND:VCALENDAR\r\n";
        let start = utc(2026, 7, 6);
        let events = parse_ics(ics, start, start + Duration::days(7));
        assert_eq!(events.len(), 1, "one weekly occurrence, all-day skipped");
        assert_eq!(events[0].title, "Weekly standup");
        assert!(events[0].start.starts_with("2026-07-09T10:00"), "{}", events[0].start);
    }

    #[test]
    fn timezones_exdates_moves_cancellations_and_folding() {
        let ics = "BEGIN:VCALENDAR\n\
BEGIN:VEVENT\nUID:sync@x\nDTSTART;TZID=Europe/Berlin:20260601T090000\nDURATION:PT30M\nRRULE:FREQ=DAILY;COUNT=10\n\
EXDATE;TZID=Europe/Berlin:20260603T090000\nSUMMARY:Daily sync\\, team A\nBEGIN:VALARM\nTRIGGER:-PT5M\nEND:VALARM\nEND:VEVENT\n\
BEGIN:VEVENT\nUID:sync@x\nRECURRENCE-ID;TZID=Europe/Berlin:20260604T090000\nDTSTART;TZID=Europe/Berlin:20260604T150000\n\
DURATION:PT30M\nSUMMARY:Daily sync (moved)\nEND:VEVENT\n\
BEGIN:VEVENT\nUID:gone@x\nDTSTART:20260602T120000Z\nSTATUS:CANCELLED\nSUMMARY:Cancelled\nEND:VEVENT\n\
BEGIN:VEVENT\nUID:long@x\nDTSTART:20260602T130000Z\nDTEND:20260602T140000Z\nSUMMARY:A very long title that is\n  folded onto two lines\nEND:VEVENT\n\
END:VCALENDAR\n";
        let events = parse_ics(ics, utc(2026, 6, 2), utc(2026, 6, 5));
        let got: Vec<(&str, &str)> = events.iter().map(|e| (e.title.as_str(), &e.start[..16])).collect();
        assert_eq!(
            got,
            vec![
                ("Daily sync, team A", "2026-06-02T09:00"),
                ("A very long title that is folded onto two lines", "2026-06-02T13:00"),
                ("Daily sync (moved)", "2026-06-04T15:00"),
            ]
        );
        assert!(events[0].start.ends_with("+02:00"), "Berlin summer time: {}", events[0].start);
    }

    #[test]
    fn durations_parse() {
        assert_eq!(parse_duration("PT1H30M"), Some(Duration::minutes(90)));
        assert_eq!(parse_duration("P1D"), Some(Duration::days(1)));
        assert_eq!(parse_duration("nonsense"), None);
    }
}
