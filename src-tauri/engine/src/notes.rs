//! Meeting notes from a transcript. Port of app.py `/summarize`, with prompts
//! budgeted in real tokens instead of word counts, so a long or non-English
//! meeting can never overflow the model's context window.
//!
//! Short meetings go straight to the summary prompt. Long ones are map-reduced:
//! each part is condensed to terse bullets ("map"), then the bullets are
//! synthesized into the one notes template ("reduce"). If even the bullets
//! don't fit, they are condensed again in rounds until they do.

use anyhow::Result;

use crate::chat::select_context;
use crate::llm::{Llm, Message};
use crate::prompts::{self, fill};

/// Room the final notes may take (the template is ~300-600 tokens).
const SUMMARY_MAX_TOKENS: usize = 1200;
/// Room each part's bullet notes may take.
const MAP_MAX_TOKENS: usize = 400;
/// Size of each part in the map step. Kept near the Python engine's 2,800
/// words: bigger parts fit the context but drown small facts (a test meeting
/// lost 4 of 6 planted decisions at ~7,600-token parts).
const MAP_PART_TOKENS: usize = 3_600;

/// Shown first while a long meeting is condensed in stages.
pub const LONG_MEETING_NOTICE: &str = "_Long meeting detected — condensing in stages, this takes a little longer…_\n\n";

/// Stream structured notes for `transcript`. `template` is a note style
/// ("general", "standup", "one_on_one", "sales", "interview"). `on_piece`
/// gets every piece of output as it's produced; returning false cancels.
pub fn summarize(llm: &Llm, transcript: &str, template: &str, on_piece: &mut dyn FnMut(&str) -> bool) -> Result<String> {
    let flavor = prompts::summary_flavor(template);
    let with_flavor = |p: String| if flavor.is_empty() { p } else { format!("{flavor}\n\n{p}") };
    let budget = llm.n_ctx() as usize - SUMMARY_MAX_TOKENS - 64; // 64 = chat-template overhead

    let direct = with_flavor(fill(&prompts::summary_prompt(), &[("TRANSCRIPT", transcript)]));
    let final_prompt = if llm.count_tokens(&direct) <= budget {
        direct
    } else {
        if !on_piece(LONG_MEETING_NOTICE) {
            return Ok(String::new());
        }
        let notes = condense(llm, transcript, budget)?;
        with_flavor(fill(&prompts::reduce_prompt(), &[("NOTES", &notes)]))
    };
    llm.chat(&[Message::user(final_prompt)], SUMMARY_MAX_TOKENS, 0.15, on_piece)
}

/// Map step, repeated until the combined bullet notes fit the reduce prompt.
fn condense(llm: &Llm, text: &str, budget: usize) -> Result<String> {
    let reduce_overhead = llm.count_tokens(&prompts::reduce_prompt());
    let map_overhead = llm.count_tokens(prompts::MAP_PROMPT) + 16;
    // Each part, plus its prompt and its answer, must fit the context.
    let part_tokens = MAP_PART_TOKENS.min(llm.n_ctx() as usize - map_overhead - MAP_MAX_TOKENS - 64);

    let mut current = text.to_string();
    loop {
        let parts = split_by_tokens(llm, &current, part_tokens);
        let mut notes = Vec::with_capacity(parts.len());
        for (i, part) in parts.iter().enumerate() {
            let prompt = fill(
                prompts::MAP_PROMPT,
                &[("IDX", &(i + 1).to_string()), ("TOTAL", &parts.len().to_string()), ("CHUNK", part)],
            );
            notes.push(llm.chat(&[Message::user(prompt)], MAP_MAX_TOKENS, 0.1, &mut |_| true)?);
        }
        let combined = notes.join("\n\n");
        // Done when the bullets fit — or when another round can't shrink them.
        if reduce_overhead + llm.count_tokens(&combined) <= budget || parts.len() == 1 {
            return Ok(combined);
        }
        current = combined;
    }
}

const EMAIL_MAX_TOKENS: usize = 400;
const ENHANCE_NOTE_MAX_TOKENS: usize = 300;
const MAX_ENHANCE_NOTES: usize = 25;
/// Users' scratchpad notes beyond this are cut (they're rough bullets).
const ENHANCE_NOTES_MAX_TOKENS: usize = 1000;
/// The title only needs the start of the meeting.
const TITLE_WORDS: usize = 400;

/// Draft a follow-up email from the meeting notes (or, if there are none
/// yet, the transcript). Starts with "Subject: ".
pub fn email(llm: &Llm, notes_or_transcript: &str, on_piece: &mut dyn FnMut(&str) -> bool) -> Result<String> {
    let budget = llm.n_ctx() as usize - EMAIL_MAX_TOKENS - llm.count_tokens(prompts::EMAIL_PROMPT) - 64;
    let notes = truncate_to_tokens(llm, notes_or_transcript.trim(), budget);
    let prompt = fill(prompts::EMAIL_PROMPT, &[("NOTES", &notes)]);
    llm.chat(&[Message::user(prompt)], EMAIL_MAX_TOKENS, 0.2, on_piece)
}

/// A short, specific title (3-7 words) from the start of the meeting.
pub fn title(llm: &Llm, transcript: &str) -> Result<String> {
    let start = transcript.split_whitespace().take(TITLE_WORDS).collect::<Vec<_>>().join(" ");
    let raw = llm.chat(&[Message::user(fill(prompts::TITLE_PROMPT, &[("TRANSCRIPT", &start)]))], 24, 0.2, &mut |_| true)?;
    Ok(clean_title(&raw))
}

/// Models sometimes wrap the title in quotes, add a full stop, or keep talking.
fn clean_title(raw: &str) -> String {
    let line = raw.trim().lines().next().unwrap_or("").trim();
    let line = line.trim_matches(|c| matches!(c, '"' | '\'' | '*' | '#' | '`')).trim();
    let line = line.trim_end_matches(['.', '!']).trim();
    line.chars().take(80).collect()
}

/// "Enhance my notes": expand each of the user's rough notes with specifics
/// from the transcript, one note at a time, then add what the notes missed.
/// Output: "### <note>" + bullets per note, in the user's order, then an
/// optional "### Also worth noting" section. Each finished block is sent to
/// `on_piece` as soon as it's ready.
pub fn enhance(llm: &Llm, transcript: &str, user_notes: &str, on_piece: &mut dyn FnMut(&str) -> bool) -> Result<String> {
    let notes = parse_notes(&truncate_to_tokens(llm, user_notes.trim(), ENHANCE_NOTES_MAX_TOKENS));
    let mut out = String::new();
    let mut emit = |block: String, out: &mut String| {
        out.push_str(&block);
        on_piece(&block)
    };

    for note in &notes {
        let bullets = match note_excerpt(transcript, note) {
            // Nothing in the meeting mentions this note's words.
            None => vec![NOT_DISCUSSED.to_string()],
            Some(excerpt) => {
                // Only the lines about this note: a small model given the
                // whole meeting lists every fact under every note.
                let prompt = fit_transcript(llm, prompts::ENHANCE_NOTE_PROMPT, &[("NOTE", note)], &excerpt, note, ENHANCE_NOTE_MAX_TOKENS);
                let raw = llm.chat(&[Message::user(prompt)], ENHANCE_NOTE_MAX_TOKENS, 0.15, &mut |_| true)?;
                let bullets = clean_bullets(&raw, 5);
                // "Not discussed" plus facts contradicts itself; trust the facts.
                let facts: Vec<String> = bullets.into_iter().filter(|b| !b.starts_with("Not discussed") && !echoes_note(b)).collect();
                if facts.is_empty() { vec![NOT_DISCUSSED.to_string()] } else { on_topic(facts, note) }
            }
        };
        let block = format!("### {note}\n{}\n\n", bullets.iter().map(|b| format!("- {b}")).collect::<Vec<_>>().join("\n"));
        if !emit(block, &mut out) {
            return Ok(out);
        }
    }

    // "What did the notes miss?" is a comparison small models get wrong
    // (they answer NONE). Instead ask for the meeting's key points — an easy
    // task — and drop the ones that mention any of the user's notes.
    let all = notes.join("\n");
    let prompt = fit_transcript(llm, prompts::KEY_POINTS_PROMPT, &[], transcript, &all, ENHANCE_NOTE_MAX_TOKENS);
    let raw = llm.chat(&[Message::user(prompt)], ENHANCE_NOTE_MAX_TOKENS, 0.15, &mut |_| true)?;
    let missed: Vec<String> = clean_bullets(&raw, 10)
        .into_iter()
        .filter(|b| !notes.iter().any(|n| mentions_note(b, n)))
        .take(4)
        .collect();
    if !missed.is_empty() {
        let block = format!("### Also worth noting\n{}\n", missed.iter().map(|b| format!("- {b}")).collect::<Vec<_>>().join("\n"));
        emit(block, &mut out);
    }
    Ok(out)
}

/// Fill `template` with `values` plus as much of the transcript (most
/// relevant to `focus` first) as fits beside `answer_tokens` of output.
fn fit_transcript(llm: &Llm, template: &str, values: &[(&str, &str)], transcript: &str, focus: &str, answer_tokens: usize) -> String {
    let mut vals: Vec<(&str, &str)> = values.to_vec();
    vals.push(("TRANSCRIPT", ""));
    let overhead = llm.count_tokens(&fill(template, &vals));
    let budget = (llm.n_ctx() as usize).saturating_sub(answer_tokens + overhead + 64);
    let context = select_context(transcript, focus, budget, |s| llm.count_tokens(s));
    vals.pop();
    vals.push(("TRANSCRIPT", &context));
    fill(template, &vals)
}

const NOT_DISCUSSED: &str = "Not discussed in the transcript.";

/// Transcript blocks that mention the note's words, each with its neighbours
/// for context, in meeting order. None when nothing mentions them.
fn note_excerpt(transcript: &str, note: &str) -> Option<String> {
    let stems: Vec<String> = words(note).filter(|w| !STOP_WORDS.contains(&w.as_str())).map(|w| stem(&w)).collect();
    if stems.is_empty() {
        return None;
    }
    // Speaker blocks are separated by blank lines; plain text falls back to lines.
    let blocks: Vec<&str> = transcript.split("\n\n").map(str::trim).filter(|b| !b.is_empty()).collect();
    let hits: Vec<usize> = blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| words(b).any(|w| stems.iter().any(|s| w.starts_with(s.as_str()))))
        .map(|(i, _)| i)
        .collect();
    if hits.is_empty() {
        return None;
    }
    let mut keep: Vec<usize> = hits.iter().flat_map(|&i| i.saturating_sub(1)..=(i + 1).min(blocks.len() - 1)).collect();
    keep.sort_unstable();
    keep.dedup();
    Some(keep.into_iter().map(|i| blocks[i]).collect::<Vec<_>>().join("\n\n"))
}

/// Keep the bullets that mention the note's words (the neighbouring lines
/// given for context tempt the model into listing unrelated facts). If none
/// do, the first bullet is the model's direct answer — keep just that.
fn on_topic(bullets: Vec<String>, note: &str) -> Vec<String> {
    let kept: Vec<String> = bullets.iter().filter(|b| mentions_note(b, note)).cloned().collect();
    if kept.is_empty() { bullets.into_iter().take(1).collect() } else { kept }
}

/// Small models sometimes "answer" by restating that the note exists
/// ("You mentioned \"budget?\"", "…is mentioned in the user's note").
fn echoes_note(bullet: &str) -> bool {
    let b = bullet.to_lowercase();
    b.contains("user's note") || b.contains("your note") || b.starts_with("you mentioned") || b.starts_with("the user mentioned")
}

/// Does `text` use any of the note's (stemmed) words?
fn mentions_note(text: &str, note: &str) -> bool {
    let stems: Vec<String> = words(note).filter(|w| !STOP_WORDS.contains(&w.as_str())).map(|w| stem(&w)).collect();
    words(text).any(|w| stems.iter().any(|s| w.starts_with(s.as_str())))
}

const STOP_WORDS: &[&str] = &["the", "and", "for", "with", "about", "what", "who", "when", "how", "our", "their", "this", "that", "from", "into", "are", "was", "were", "will", "any", "some", "need", "check", "todo", "note", "notes"];

fn words(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric()).filter(|w| w.chars().count() >= 3).map(str::to_lowercase)
}

/// Crude suffix stripping so "hiring" finds "hire" and "launched" finds "launch".
fn stem(word: &str) -> String {
    for suffix in ["ing", "ed", "es", "s"] {
        if let Some(base) = word.strip_suffix(suffix) {
            if base.chars().count() >= 3 {
                return base.to_string();
            }
        }
    }
    word.to_string()
}

/// The user's notes, one per line, without bullet or numbering marks.
/// Capped so a pasted wall of text can't turn into hundreds of model calls.
fn parse_notes(text: &str) -> Vec<String> {
    text.lines()
        .map(strip_marker)
        .filter(|l| !l.is_empty())
        .take(MAX_ENHANCE_NOTES)
        .map(str::to_string)
        .collect()
}

/// Bullet texts from a model answer: list lines only, markers removed, empty
/// and "nothing found" lines dropped, at most `max`.
fn clean_bullets(raw: &str, max: usize) -> Vec<String> {
    raw.lines()
        .map(str::trim)
        .filter(|l| l.starts_with(['-', '*', '•']) || l.chars().next().is_some_and(|c| c.is_ascii_digit()))
        .map(strip_marker)
        .filter(|l| !l.is_empty() && !l.eq_ignore_ascii_case("none"))
        .take(max)
        .map(str::to_string)
        .collect()
}

/// "- note", "* note", "• note", "1. note", "2) note", "- - note" → "note".
fn strip_marker(line: &str) -> &str {
    let mut l = line.trim();
    loop {
        let before = l;
        l = l.trim_start_matches(['-', '*', '•']).trim_start();
        let digits = l.len() - l.trim_start_matches(|c: char| c.is_ascii_digit()).len();
        if digits > 0 && l[digits..].starts_with(['.', ')']) {
            l = l[digits + 1..].trim_start();
        }
        if l == before {
            return l.trim_matches('#').trim();
        }
    }
}

/// The leading part of `text` that fits in `max_tokens`.
fn truncate_to_tokens(llm: &Llm, text: &str, max_tokens: usize) -> String {
    if llm.count_tokens(text) <= max_tokens {
        return text.to_string();
    }
    split_by_tokens(llm, text, max_tokens).into_iter().next().unwrap_or_default()
}

/// Split on whitespace into pieces of at most `max_tokens` tokens each.
pub fn split_by_tokens(llm: &Llm, text: &str, max_tokens: usize) -> Vec<String> {
    split_with(text, max_tokens, |s| llm.count_tokens(s))
}

/// Greedy word packing. Counting each word on its own over-estimates slightly
/// (no merges across word boundaries), which only errs on the safe side.
fn split_with(text: &str, max_tokens: usize, count: impl Fn(&str) -> usize) -> Vec<String> {
    let mut parts = Vec::new();
    let (mut cur, mut cur_tokens) = (String::new(), 0usize);
    for word in text.split_whitespace() {
        let t = count(&format!(" {word}"));
        if cur_tokens + t > max_tokens && !cur.is_empty() {
            parts.push(std::mem::take(&mut cur));
            cur_tokens = 0;
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(word);
        cur_tokens += t;
    }
    if !cur.is_empty() {
        parts.push(cur);
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_respects_budget_and_keeps_every_word() {
        let text = (0..1000).map(|i| format!("w{i}")).collect::<Vec<_>>().join(" ");
        // Pretend every word costs 3 tokens.
        let parts = split_with(&text, 30, |_| 3);
        assert_eq!(parts.len(), 100);
        assert!(parts.iter().all(|p| p.split_whitespace().count() == 10));
        let rejoined = parts.join(" ");
        assert_eq!(rejoined, text);
    }

    #[test]
    fn excerpt_finds_only_blocks_about_the_note() {
        let t = "[You 00:00] Hello all.\n\n[A 00:05] Dark mode moves to 1.1.\n\n[B 00:09] Budget is 5k.\n\n[A 00:12] Fine.\n\n[B 00:15] We will hire two engineers.";
        let dark = note_excerpt(t, "dark mode").unwrap();
        assert!(dark.contains("Dark mode") && dark.contains("Hello all") && dark.contains("Budget"));
        assert!(!dark.contains("hire"));
        assert!(note_excerpt(t, "hiring").unwrap().contains("hire two engineers"));
        assert!(note_excerpt(t, "pricing").is_none());
        assert!(note_excerpt(t, "the and").is_none());
    }

    #[test]
    fn off_topic_bullets_are_dropped() {
        let b = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let kept = on_topic(b(&["Dark Mode moves to 1.1.", "Budget is $5,000.", "The dark theme ships later."]), "dark mode");
        assert_eq!(kept, b(&["Dark Mode moves to 1.1.", "The dark theme ships later."]));
        // Nothing mentions the note: keep the model's first, direct answer.
        assert_eq!(on_topic(b(&["Ships on Friday.", "Karen owns it."]), "launch"), b(&["Ships on Friday."]));
    }

    #[test]
    fn echo_bullets_are_recognized() {
        assert!(echoes_note("\"android tablet\" is mentioned in the user's note."));
        assert!(echoes_note("You mentioned \"budget?\" during the meeting."));
        assert!(!echoes_note("The beta ads budget is capped at $5,000."));
    }

    #[test]
    fn notes_lose_their_markers() {
        let notes = parse_notes("- launch date\n\n* budget?\n1. dark mode\n2) hiring\n• security\n## pricing");
        assert_eq!(notes, vec!["launch date", "budget?", "dark mode", "hiring", "security", "pricing"]);
    }

    #[test]
    fn model_bullets_are_normalized() {
        let raw = "Sure! Here you go:\n- - The budget is $5,000.\n* Daniel reports weekly.\n\n1. Starts Friday.\nThanks!";
        assert_eq!(clean_bullets(raw, 5), vec!["The budget is $5,000.", "Daniel reports weekly.", "Starts Friday."]);
        assert_eq!(clean_bullets(raw, 2).len(), 2);
        assert!(clean_bullets("NONE", 4).is_empty());
        assert!(clean_bullets("- none", 4).is_empty());
    }

    #[test]
    fn titles_are_cleaned_up() {
        assert_eq!(clean_title("\"Mobile App Launch Planning.\"\nHope that helps!"), "Mobile App Launch Planning");
        assert_eq!(clean_title("**Q3 Budget Review**"), "Q3 Budget Review");
        assert_eq!(clean_title(&"x".repeat(200)).len(), 80);
        assert_eq!(clean_title("   "), "");
    }

    #[test]
    fn split_never_returns_empty_parts() {
        assert!(split_with("", 10, |_| 1).is_empty());
        // A single word larger than the budget still goes out on its own.
        assert_eq!(split_with("huge", 1, |_| 5), vec!["huge"]);
    }
}
