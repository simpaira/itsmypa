//! Every LLM prompt, copied verbatim from app.py so notes come out exactly as
//! before. `{{NAME}}` placeholders are filled with [`fill`].
//!
//! Every summary — direct or map-reduced — is forced into one template so the
//! UI can parse it into Executive Summary / Decisions / Action Items / Topics
//! cards instead of a single wall of markdown.

pub const SUMMARY_TEMPLATE_RULES: &str = r#"Follow this EXACT markdown template, with these exact section headers, in this exact order, every time:

## Executive Summary
2-4 sentences: what the meeting was about and its outcome.

## Key Decisions
- One bullet per decision actually made. If none were made, write a single bullet: "No decisions were recorded."

## Action Items
- [ ] Task description — @Owner
(One bullet per task. Use @Unassigned if no owner is stated in the transcript. If there are no action items, write "- [ ] No action items were recorded. — @Unassigned")

## Topics Discussed
- Short topic label (3-8 labels, not sentences)

Rules:
- Be concise, factual, and specific. Never invent names, numbers, or facts not present in the source.
- Output ONLY the template above — no preamble, no closing remarks, no extra headers.
"#;

pub fn summary_prompt() -> String {
    format!(
        "You are an expert meeting analyst. Read the transcript below and produce structured notes.\n\n{SUMMARY_TEMPLATE_RULES}\n---\nTRANSCRIPT:\n{{{{TRANSCRIPT}}}}\n---"
    )
}

pub const MAP_PROMPT: &str = "Extract the key facts, decisions, and action items from this excerpt \
(part {{IDX}} of {{TOTAL}}) of a longer meeting transcript. \
Be terse: short bullet points only, no commentary, no headers.\n\nEXCERPT:\n{{CHUNK}}";

pub fn reduce_prompt() -> String {
    format!(
        "You are an expert meeting analyst. Below are bullet-point notes extracted independently \
from sequential parts of one long meeting transcript. Synthesize them into a single set \
of meeting notes, removing duplicates and resolving the full picture.\n\n{SUMMARY_TEMPLATE_RULES}\n---\nNOTES FROM TRANSCRIPT PARTS:\n{{{{NOTES}}}}\n---"
    )
}

/// Note styles: a flavor line steers what the summarizer weights. Keys must
/// match the <select> options in the UI.
pub fn summary_flavor(template: &str) -> &'static str {
    match template {
        "standup" => "Context: this is a daily standup. Weight per-person updates, blockers, and who is waiting on whom.",
        "one_on_one" => "Context: this is a 1:1 conversation. Weight feedback, career/personal topics, agreements, and follow-ups; keep the tone neutral and discreet.",
        "sales" => "Context: this is a sales or client call. Weight the client's needs and pain points, objections raised, budget/timeline signals, and concrete next steps.",
        "interview" => "Context: this is a job interview. Weight the candidate's background, demonstrated skills, notable answers, concerns, and suggested follow-up questions.",
        _ => "",
    }
}

/// "Enhance my notes": the user's rough bullets guide the AI; the output
/// repeats each user note verbatim followed by expansions from the transcript.
pub const ENHANCE_PROMPT: &str = "You are enhancing a user's rough meeting notes using the meeting transcript.\n\
For EACH line or bullet of the user's notes, output exactly:\n\
### <the user's note, repeated verbatim>\n\
- 2 to 5 factual bullets expanding that note with specifics from the transcript: \
decisions, numbers, owners, dates, and short quotes.\n\
Keep the user's original order. If a note has no match in the transcript, output a \
single bullet: 'Not discussed in the transcript.'\n\
After covering every user note, add one final section:\n\
### Also worth noting\n\
- up to 4 bullets for important points the user's notes missed. If none, skip this section.\n\
Rules: never invent facts, names, or numbers not present in the transcript. \
Output ONLY the format above — no preamble, no closing remarks.\n\n\
---\nUSER NOTES:\n{{NOTES}}\n---\nTRANSCRIPT:\n{{TRANSCRIPT}}\n---";

pub const EMAIL_PROMPT: &str = "Write a short, professional follow-up email for the meeting below.\n\
Rules:\n\
- Start with 'Subject: ' on the first line.\n\
- Briefly thank attendees, recap the key decisions, then list action items with owners.\n\
- Use only facts from the meeting notes/transcript — never invent names, dates, or commitments.\n\
- Keep it under 180 words. Plain text only, no markdown.\n\n\
---\nMEETING NOTES:\n{{NOTES}}\n---";

pub const TITLE_PROMPT: &str = "Give this meeting a short, specific title (3-7 words). \
Output ONLY the title — no quotes, no punctuation at the end, no commentary.\n\n\
---\nTRANSCRIPT (start of meeting):\n{{TRANSCRIPT}}\n---";

pub const CHAT_SYSTEM: &str = "You are a helpful assistant answering questions about a meeting. \
Answer ONLY using the meeting transcript provided below — never invent facts, names, or numbers. \
If the answer isn't in the transcript, say plainly that it wasn't discussed. \
If the user's message is small talk or not a question about the meeting, reply in one short \
friendly sentence and offer to answer questions about the meeting — never summarize unprompted. \
Reference speaker labels (e.g. 'Speaker 2') when relevant. Be concise and direct.\n\n\
---\nTRANSCRIPT:\n{{TRANSCRIPT}}\n---";

pub const ASK_ANALYZE_PROMPT: &str = "Extract search filters from a question about the user's meeting-notes archive.\n\
Today is {{TODAY}}.\n\
Reply with ONLY a JSON object, no other text:\n\
{\"keywords\": [4-8 search words: the key nouns/verbs of the question PLUS close synonyms \
(e.g. for \"front end\" also \"frontend\", \"UI\", \"web\")], \
\"people\": [person names mentioned in the question, else []], \
\"date_from\": \"YYYY-MM-DD or empty string\", \"date_to\": \"YYYY-MM-DD or empty string\"}\n\
Set dates ONLY when the question names a specific day or period (\"on July 8\", \"last week\"); \
for a single day set date_from and date_to to that same day. Otherwise use empty strings.\n\n\
Question: {{QUESTION}}";

pub const ASK_ANSWER_PROMPT: &str = "You answer questions from the user's saved meeting notes. Today is {{TODAY}}.\n\
Use ONLY the meeting excerpts below; each is headed [Meeting title — date].\n\
- Answer directly and concisely, in Markdown.\n\
- Mention which meeting each fact comes from naturally in prose (e.g. \"In the Q3 review \
on 2026-07-01, …\") — never reproduce the bracketed [ … ] header format itself.\n\
- If the excerpts don't answer the question, say so plainly — never invent details.\n\
- If the message is small talk rather than a question about meetings, reply in one short \
friendly sentence inviting a question — do not mention or cite any meeting.\n\n\
EXCERPTS:\n{{CONTEXT}}\n\nQuestion: {{QUESTION}}";

/// Replace each `{{KEY}}` with its value. Values are inserted once, in order,
/// so text inside a value (a transcript containing "{{NOTES}}") is never
/// substituted again.
pub fn fill(template: &str, values: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(open) = rest.find("{{") {
        let Some(close) = rest[open..].find("}}") else { break };
        let key = &rest[open + 2..open + close];
        match values.iter().find(|(k, _)| *k == key) {
            Some((_, v)) => {
                out.push_str(&rest[..open]);
                out.push_str(v);
            }
            None => out.push_str(&rest[..open + close + 2]),
        }
        rest = &rest[open + close + 2..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_replaces_keys_once_and_keeps_unknown() {
        let out = fill("a {{X}} b {{Y}} c {{Z}}", &[("X", "{{Y}}"), ("Y", "2")]);
        assert_eq!(out, "a {{Y}} b 2 c {{Z}}");
    }

    #[test]
    fn summary_prompt_keeps_its_placeholder() {
        assert!(summary_prompt().contains("TRANSCRIPT:\n{{TRANSCRIPT}}\n---"));
        assert!(reduce_prompt().ends_with("{{NOTES}}\n---"));
    }
}
