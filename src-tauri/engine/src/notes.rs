//! Meeting notes from a transcript. Port of app.py `/summarize`, with prompts
//! budgeted in real tokens instead of word counts, so a long or non-English
//! meeting can never overflow the model's context window.
//!
//! Short meetings go straight to the summary prompt. Long ones are map-reduced:
//! each part is condensed to terse bullets ("map"), then the bullets are
//! synthesized into the one notes template ("reduce"). If even the bullets
//! don't fit, they are condensed again in rounds until they do.

use anyhow::Result;

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
    fn split_never_returns_empty_parts() {
        assert!(split_with("", 10, |_| 1).is_empty());
        // A single word larger than the budget still goes out on its own.
        assert_eq!(split_with("huge", 1, |_| 5), vec!["huge"]);
    }
}
