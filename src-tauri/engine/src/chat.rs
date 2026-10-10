//! "Ask about this meeting" chat. Port of app.py `/chat` and
//! `select_chat_context`, with the transcript excerpt sized in real tokens.

use anyhow::Result;
use regex::Regex;
use std::sync::OnceLock;

use crate::llm::{Llm, Message};
use crate::prompts::{self, fill};

const ANSWER_MAX_TOKENS: usize = 768;
/// Earlier turns kept for follow-up questions ("and who owns that?").
const HISTORY_TURNS: usize = 6;
/// Transcript is scored for relevance in windows of this many words.
const WINDOW_WORDS: usize = 200;

pub const CHAT_GREETING: &str =
    "Hi! Ask me anything about this meeting — decisions, action items, or who said what.";

/// Greetings and pleasantries must never reach the model: a grounded prompt
/// turns "hi" into an unsolicited meeting summary (or worse, a citation of a
/// transcript line where somebody happened to say "hi").
pub fn is_small_talk(message: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)^(hi+|hii+|hello+|hey+|yo|sup|howdy|good\s*(morning|afternoon|evening)|thanks?|thank\s*you|thx|ty|ok(ay)?|cool|nice|great|got\s*it|bye|goodbye|test(ing)?|how\s*are\s*you\??)[\s!.?,]*$",
        )
        .unwrap()
    })
    .is_match(message.trim())
}

pub struct Turn {
    pub from_user: bool,
    pub text: String,
}

/// Answer `question` from the meeting only, streaming pieces to `on_piece`.
pub fn ask_meeting(
    llm: &Llm,
    transcript: &str,
    question: &str,
    history: &[Turn],
    on_piece: &mut dyn FnMut(&str) -> bool,
) -> Result<String> {
    if is_small_talk(question) {
        on_piece(CHAT_GREETING);
        return Ok(CHAT_GREETING.to_string());
    }

    let recent = &history[history.len().saturating_sub(HISTORY_TURNS)..];
    let history_msgs: Vec<Message> = recent
        .iter()
        .filter(|t| !t.text.trim().is_empty())
        .map(|t| if t.from_user { Message::user(t.text.trim()) } else { Message::assistant(t.text.trim()) })
        .collect();

    // Whatever the instructions, history and question don't use is left for
    // the transcript (64 tokens of slack for the chat template's markup).
    let fixed = llm.count_tokens(&fill(prompts::CHAT_SYSTEM, &[("TRANSCRIPT", "")]))
        + history_msgs.iter().map(|m| llm.count_tokens(&m.content) + 8).sum::<usize>()
        + llm.count_tokens(question);
    let budget = (llm.n_ctx() as usize).saturating_sub(ANSWER_MAX_TOKENS + fixed + 64);
    let context = select_context(transcript, question, budget, |s| llm.count_tokens(s));

    let mut messages = vec![Message::system(fill(prompts::CHAT_SYSTEM, &[("TRANSCRIPT", &context)]))];
    messages.extend(history_msgs);
    messages.push(Message::user(question));
    llm.chat(&messages, ANSWER_MAX_TOKENS, 0.2, on_piece)
}

/// The whole transcript if it fits `max_tokens`; otherwise the 200-word
/// windows that mention the question's words most, kept in meeting order
/// with "[…]" marking the gaps. Also used to focus "enhance my notes".
pub fn select_context(transcript: &str, question: &str, max_tokens: usize, count: impl Fn(&str) -> usize) -> String {
    if count(transcript) <= max_tokens {
        return transcript.to_string();
    }
    const HEADER: &str = "(Excerpts of a long transcript, selected for relevance to the question)\n\n";
    let words: Vec<&str> = transcript.split_whitespace().collect();
    let windows: Vec<String> = words.chunks(WINDOW_WORDS).map(|w| w.join(" ")).collect();
    let terms: Vec<String> = question
        .to_lowercase()
        .split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .filter(|w| w.chars().count() > 3)
        .map(str::to_string)
        .collect();

    let mut ranked: Vec<(usize, usize)> = windows
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let lower = w.to_lowercase();
            (terms.iter().map(|t| lower.matches(t.as_str()).count()).sum(), i)
        })
        .collect();
    // Most relevant first; ties go to the later window, as the Python did.
    ranked.sort_by(|a, b| b.cmp(a));

    let mut used = count(HEADER);
    let mut keep = Vec::new();
    for (_, i) in ranked {
        let cost = count(&windows[i]) + 2;
        if used + cost > max_tokens {
            continue;
        }
        used += cost;
        keep.push(i);
    }
    keep.sort_unstable();

    let mut out = String::from(HEADER);
    let mut prev: Option<usize> = None;
    for i in keep {
        if prev.is_some_and(|p| i != p + 1) {
            out.push_str("[…]\n");
        }
        out.push_str(&windows[i]);
        out.push('\n');
        prev = Some(i);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ~1.3 tokens per word, close enough for budgeting tests.
    fn approx(s: &str) -> usize {
        s.split_whitespace().count() * 13 / 10
    }

    // Port of test_chat_smalltalk_never_reaches_llm / test_ask_smalltalk_…
    #[test]
    fn greetings_are_small_talk() {
        for m in ["hi", "Hello!", "thanks", "how are you?", "ok", "hey!!", "thank you", "good morning"] {
            assert!(is_small_talk(m), "{m}");
        }
    }

    // Port of test_real_questions_bypass_smalltalk_guard
    #[test]
    fn real_questions_are_not_small_talk() {
        for q in ["hi, what did we decide about pricing?", "who said thanks to the client?", "what are the action items"] {
            assert!(!is_small_talk(q), "{q}");
        }
    }

    // Port of test_select_chat_context_passthrough_when_short
    #[test]
    fn short_transcript_passes_through() {
        assert_eq!(select_context("short transcript", "anything?", 5000, approx), "short transcript");
    }

    // Port of test_select_chat_context_picks_relevant_chunks
    #[test]
    fn long_transcript_keeps_relevant_windows_within_budget() {
        let filler = "word ".repeat(200);
        let needle = "the budget approval deadline is friday ".repeat(5);
        let transcript = format!("{}{}{}", filler.repeat(30), needle, filler.repeat(30));
        let out = select_context(&transcript, "when is the budget approval deadline?", 1300, approx);
        assert!(out.contains("budget approval"));
        assert!(approx(&out) <= 1300);
    }
}
