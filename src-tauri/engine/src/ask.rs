//! "Ask across all meetings": port of app.py `/ask` — the LLM turns the
//! question into search filters, BM25 ranks 250-word passages from every
//! saved meeting, and the LLM answers from the best passages only.
//!
//! The corpus (one person's meetings) is small enough that lexical BM25 over
//! in-memory passages beats a vector store: no embedding model to download,
//! no index files to keep in sync. Semantic recall comes from the LLM
//! expanding the question into synonyms before retrieval.

use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

use crate::chat::is_small_talk;
use crate::llm::{Llm, Message};
use crate::prompts::{self, fill};
use crate::store::{Note, Store};

pub const ASK_GREETING: &str =
    "Hi! Ask me anything about your saved meetings — decisions, action items, who said what, and when.";
pub const NOTHING_FOUND: &str = "I couldn't find anything about that in your saved meetings.";

const PASSAGE_WORDS: usize = 250;
const TOP_K: usize = 10;
const PER_MEETING: usize = 3;
const MAX_SOURCES: usize = 6;
const ANALYZE_MAX_TOKENS: usize = 220;
const ANSWER_MAX_TOKENS: usize = 900;

/// A meeting an answer drew on, for the clickable source chips.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Source {
    pub id: String,
    pub title: String,
    pub date: String,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Summary,
    Notes,
    Transcript,
}

struct Passage {
    id: String,
    title: String,
    date: String, // YYYY-MM-DD
    kind: Kind,
    text: String,
    tf: HashMap<String, u32>,
    len: usize,
}

#[derive(Default)]
struct Index {
    passages: Vec<Passage>,
    df: HashMap<String, u32>,
    avg_len: f64,
}

/// Search filters the LLM extracts from the question.
#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct Filters {
    pub keywords: Vec<String>,
    pub people: Vec<String>,
    pub date_from: String,
    pub date_to: String,
}

/// Keeps the passage index between questions; rebuilt only when the saved
/// meetings change.
#[derive(Default)]
pub struct Asker {
    cache: Mutex<Option<(u64, std::sync::Arc<Index>)>>,
}

impl Asker {
    fn index(&self, store: &Store) -> Result<std::sync::Arc<Index>> {
        let version = store.version();
        let mut cache = self.cache.lock().unwrap();
        if let Some((v, idx)) = cache.as_ref() {
            if *v == version {
                return Ok(idx.clone());
            }
        }
        let idx = std::sync::Arc::new(build_index(&store.list()?));
        *cache = Some((version, idx.clone()));
        Ok(idx)
    }

    /// Answer `question` from all saved meetings, streaming the answer to
    /// `on_piece`. Returns the answer and the meetings it drew on.
    /// `today` is "YYYY-MM-DD (Weekday)" so "last week" can be resolved.
    pub fn ask(
        &self,
        llm: &Llm,
        store: &Store,
        question: &str,
        today: &str,
        on_piece: &mut dyn FnMut(&str) -> bool,
    ) -> Result<(String, Vec<Source>)> {
        let question = question.trim();
        if is_small_talk(question) {
            on_piece(ASK_GREETING);
            return Ok((ASK_GREETING.into(), Vec::new()));
        }

        let q: String = question.chars().take(500).collect();
        let raw = llm.chat(
            &[Message::user(fill(prompts::ASK_ANALYZE_PROMPT, &[("TODAY", today), ("QUESTION", &q)]))],
            ANALYZE_MAX_TOKENS,
            0.1,
            &mut |_| true,
        )?;
        let filters = single_day_if_one_bound(parse_filters(&raw), question);

        let mut terms = tokens(question);
        for kw in &filters.keywords {
            terms.extend(tokens(kw));
        }
        let mut seen = HashSet::new();
        terms.retain(|t| seen.insert(t.clone()));

        let index = self.index(store)?;
        let dated = !filters.date_from.is_empty() || !filters.date_to.is_empty();
        let mut hits = retrieve(&index, &terms, &filters.date_from, &filters.date_to, &filters.people);
        if hits.is_empty() && dated {
            // "What was discussed on Sep 22?" has no topic words to match:
            // use the meetings from that period as they are.
            hits = in_period(&index, &filters.date_from, &filters.date_to);
        }
        if hits.is_empty() && dated {
            // The date guess may be off — retry without it.
            hits = retrieve(&index, &terms, "", "", &filters.people);
        }
        if hits.is_empty() {
            on_piece(NOTHING_FOUND);
            return Ok((NOTHING_FOUND.into(), Vec::new()));
        }

        let context = hits
            .iter()
            .map(|p| format!("[{} — {}]\n{}", p.title, p.date, p.text))
            .collect::<Vec<_>>()
            .join("\n\n");
        let prompt = fill(prompts::ASK_ANSWER_PROMPT, &[("TODAY", today), ("CONTEXT", &context), ("QUESTION", question)]);
        let answer = llm.chat(&[Message::user(prompt)], ANSWER_MAX_TOKENS, 0.15, on_piece)?;

        let mut sources: Vec<Source> = Vec::new();
        for p in &hits {
            if !sources.iter().any(|s| s.id == p.id) {
                sources.push(Source { id: p.id.clone(), title: p.title.clone(), date: p.date.clone() });
            }
        }
        sources.truncate(MAX_SOURCES);
        Ok((answer, sources))
    }
}

const STOP: &[&str] = &[
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "were", "be", "been", "am", "i",
    "we", "you", "he", "she", "it", "they", "at", "with", "about", "that", "this", "these", "those", "my", "our",
    "your", "his", "her", "its", "from", "as", "by", "what", "which", "who", "whom", "when", "where", "how", "did",
    "does", "do", "can", "could", "should", "would", "will",
];

fn tokens(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| w.len() > 1 && !STOP.contains(w))
        .map(str::to_string)
        .collect()
}

fn build_index(notes: &[Note]) -> Index {
    let mut passages = Vec::new();
    for n in notes.iter().filter(|n| n.status.is_empty() || n.status == "ready") {
        let title = if n.title.trim().is_empty() { "Untitled meeting".to_string() } else { n.title.clone() };
        let date: String = n.date.chars().take(10).collect();
        let notes_text = if n.enhanced.trim().is_empty() { &n.scratchpad } else { &n.enhanced };
        for (kind, text) in [(Kind::Summary, &n.summary), (Kind::Notes, notes_text), (Kind::Transcript, &n.transcript)] {
            let words: Vec<&str> = text.split_whitespace().collect();
            if words.len() < 8 {
                continue;
            }
            for w in words.chunks(PASSAGE_WORDS) {
                let seg = w.join(" ");
                let toks = tokens(&format!("{title} {seg}"));
                if toks.is_empty() {
                    continue;
                }
                let mut tf = HashMap::new();
                for t in &toks {
                    *tf.entry(t.clone()).or_insert(0) += 1;
                }
                passages.push(Passage { id: n.id.clone(), title: title.clone(), date: date.clone(), kind, text: seg, len: toks.len(), tf });
            }
        }
    }
    let mut df: HashMap<String, u32> = HashMap::new();
    for p in &passages {
        for t in p.tf.keys() {
            *df.entry(t.clone()).or_insert(0) += 1;
        }
    }
    let avg_len = if passages.is_empty() { 1.0 } else { passages.iter().map(|p| p.len).sum::<usize>() as f64 / passages.len() as f64 };
    Index { passages, df, avg_len }
}

/// BM25 over passages with the Python engine's boosts: passages naming a
/// person from the question ×1.6, summaries ×1.3 (they carry decisions and
/// action items). At most 3 passages per meeting, to spread context.
fn retrieve<'a>(idx: &'a Index, terms: &[String], date_from: &str, date_to: &str, people: &[String]) -> Vec<&'a Passage> {
    let n = idx.passages.len().max(1) as f64;
    let people_toks: HashSet<String> = people.iter().flat_map(|p| tokens(p)).collect();
    let mut scored: Vec<(f64, &Passage)> = Vec::new();
    for p in &idx.passages {
        if !date_from.is_empty() && !p.date.is_empty() && p.date.as_str() < date_from {
            continue;
        }
        if !date_to.is_empty() && !p.date.is_empty() && p.date.as_str() > date_to {
            continue;
        }
        let mut s = 0.0;
        for t in terms {
            let Some(&f) = p.tf.get(t) else { continue };
            let df = *idx.df.get(t).unwrap_or(&0) as f64;
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            let f = f as f64;
            s += idf * (f * 2.5) / (f + 1.5 * (0.25 + 0.75 * p.len as f64 / idx.avg_len));
        }
        if s == 0.0 {
            continue;
        }
        if !people_toks.is_empty() && people_toks.iter().any(|t| p.tf.contains_key(t)) {
            s *= 1.6;
        }
        if p.kind == Kind::Summary {
            s *= 1.3;
        }
        scored.push((s, p));
    }
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut per_meeting: HashMap<&str, usize> = HashMap::new();
    let mut out = Vec::new();
    for (_, p) in scored {
        let c = per_meeting.entry(p.id.as_str()).or_insert(0);
        if *c >= PER_MEETING {
            continue;
        }
        *c += 1;
        out.push(p);
        if out.len() >= TOP_K {
            break;
        }
    }
    out
}

/// Small models often fill only `date_from` for "on September 22". Unless the
/// question asks for an open range ("since…", "before…"), one bound means
/// that single day.
fn single_day_if_one_bound(mut f: Filters, question: &str) -> Filters {
    let q = question.to_lowercase();
    let open_range = ["since", "after", "before", "until", "till", "from", "onward", "prior"]
        .iter()
        .any(|w| q.split(|c: char| !c.is_alphanumeric()).any(|t| t == *w));
    if !open_range {
        if f.date_to.is_empty() {
            f.date_to = f.date_from.clone();
        } else if f.date_from.is_empty() {
            f.date_from = f.date_to.clone();
        }
    }
    f
}

/// Passages from meetings in the date range, when the question is only about
/// a period: summaries and notes first, then each transcript from its start.
fn in_period<'a>(idx: &'a Index, date_from: &str, date_to: &str) -> Vec<&'a Passage> {
    let in_range = |p: &&Passage| {
        !p.date.is_empty()
            && (date_from.is_empty() || p.date.as_str() >= date_from)
            && (date_to.is_empty() || p.date.as_str() <= date_to)
    };
    let mut ranked: Vec<&Passage> = idx.passages.iter().filter(in_range).collect();
    // Stable sort keeps each meeting's passages in order.
    ranked.sort_by_key(|p| match p.kind {
        Kind::Summary => 0,
        Kind::Notes => 1,
        Kind::Transcript => 2,
    });
    let mut per_meeting: HashMap<&str, usize> = HashMap::new();
    ranked
        .into_iter()
        .filter(|p| {
            let c = per_meeting.entry(p.id.as_str()).or_insert(0);
            *c += 1;
            *c <= PER_MEETING
        })
        .take(TOP_K)
        .collect()
}

/// The first {...} object in the model's reply; anything unusable → no filters.
fn parse_filters(raw: &str) -> Filters {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"(?s)\{.*\}").unwrap());
    let Some(m) = re.find(raw) else { return Filters::default() };
    let mut f: Filters = serde_json::from_str(m.as_str()).unwrap_or_default();
    // Only accept real YYYY-MM-DD dates.
    for d in [&mut f.date_from, &mut f.date_to] {
        let ok = d.len() >= 10 && d[..10].chars().enumerate().all(|(i, c)| if i == 4 || i == 7 { c == '-' } else { c.is_ascii_digit() });
        *d = if ok { d[..10].to_string() } else { String::new() };
    }
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(id: &str, title: &str, date: &str, summary: &str, transcript: &str) -> Note {
        Note {
            id: id.into(),
            title: title.into(),
            date: date.into(),
            summary: summary.into(),
            transcript: transcript.into(),
            status: "ready".into(),
            ..Default::default()
        }
    }

    fn corpus() -> Index {
        build_index(&[
            note("a", "Pricing review", "2026-09-01T10:00:00", "We decided the Pro plan costs twelve dollars per month and annual gets two months free.", "lots of talk about pricing tiers and the pro plan and discounts for students"),
            note("b", "Security sync", "2026-09-15T10:00:00", "Northwind Security starts the audit on April second with Maria as owner.", "the audit scope covers billing and login pages for every customer"),
            note("c", "Standup", "2026-09-20T09:00:00", "Short standup with no decisions about anything at all today really.", "tom is migrating the billing database to the new cluster"),
        ])
    }

    #[test]
    fn retrieval_ranks_the_matching_meeting_first() {
        let idx = corpus();
        let hits = retrieve(&idx, &tokens("what did we decide about pricing for the pro plan"), "", "", &[]);
        assert_eq!(hits[0].id, "a");
        assert!(hits.iter().all(|p| p.id == "a"));
    }

    #[test]
    fn summaries_are_boosted_over_equal_transcript_matches() {
        let same = "the launch is on march fourteen for the mobile app";
        let idx = build_index(&[note("a", "x", "2026-09-01", same, same)]);
        let hits = retrieve(&idx, &tokens("launch"), "", "", &[]);
        assert_eq!(hits[0].kind, Kind::Summary);
    }

    #[test]
    fn date_filters_and_people_boost() {
        let idx = corpus();
        let billing = tokens("billing");
        let all: Vec<&str> = retrieve(&idx, &billing, "", "", &[]).iter().map(|p| p.id.as_str()).collect();
        assert!(all.contains(&"b") && all.contains(&"c"));
        let late: Vec<&str> = retrieve(&idx, &billing, "2026-09-16", "", &[]).iter().map(|p| p.id.as_str()).collect();
        assert_eq!(late, vec!["c"]);
        let maria = retrieve(&idx, &tokens("audit owner"), "", "", &["Maria".into()]);
        assert_eq!(maria[0].id, "b");
    }

    #[test]
    fn at_most_three_passages_per_meeting() {
        let long = "pricing ".repeat(2000);
        let idx = build_index(&[note("a", "x", "2026-09-01", "", &long), note("b", "y", "2026-09-02", "", "pricing came up once here among some other unrelated words")]);
        let hits = retrieve(&idx, &tokens("pricing"), "", "", &[]);
        assert_eq!(hits.iter().filter(|p| p.id == "a").count(), 3);
        assert!(hits.iter().any(|p| p.id == "b"));
    }

    #[test]
    fn one_date_bound_means_one_day_unless_the_question_is_a_range() {
        let f = |from: &str, to: &str| Filters { date_from: from.into(), date_to: to.into(), ..Default::default() };
        let d = single_day_if_one_bound(f("2026-09-22", ""), "What was discussed on September 22?");
        assert_eq!((d.date_from.as_str(), d.date_to.as_str()), ("2026-09-22", "2026-09-22"));
        let d = single_day_if_one_bound(f("2026-09-22", ""), "What changed since September 22?");
        assert_eq!(d.date_to, "");
        let d = single_day_if_one_bound(f("", ""), "anything about pricing?");
        assert_eq!((d.date_from.as_str(), d.date_to.as_str()), ("", ""));
    }

    #[test]
    fn date_only_questions_use_meetings_from_that_period() {
        let idx = corpus();
        let ids: Vec<&str> = in_period(&idx, "2026-09-15", "2026-09-15").iter().map(|p| p.id.as_str()).collect();
        assert!(!ids.is_empty() && ids.iter().all(|&i| i == "b"));
        assert_eq!(in_period(&idx, "2026-09-15", "2026-09-15")[0].kind, Kind::Summary);
        assert!(in_period(&idx, "2025-01-01", "2025-01-31").is_empty());
    }

    #[test]
    fn filters_parse_from_messy_model_output() {
        let raw = "Sure! Here it is:\n```json\n{\"keywords\": [\"pricing\", \"price\"], \"people\": [\"Maria\"], \"date_from\": \"2026-09-01\", \"date_to\": \"last week\"}\n```";
        let f = parse_filters(raw);
        assert_eq!(f.keywords, vec!["pricing", "price"]);
        assert_eq!(f.people, vec!["Maria"]);
        assert_eq!((f.date_from.as_str(), f.date_to.as_str()), ("2026-09-01", ""));
        assert_eq!(parse_filters("no json here"), Filters::default());
        assert_eq!(parse_filters("{broken"), Filters::default());
    }

    #[test]
    fn unfinished_meetings_are_not_searched() {
        let mut n = note("a", "Live", "2026-09-01", "pricing pricing pricing pricing pricing pricing pricing pricing", "");
        n.status = "processing".into();
        assert!(build_index(&[n]).passages.is_empty());
    }
}
