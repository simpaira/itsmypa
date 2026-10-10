//! The AI features that stream their answer as it's written: meeting notes,
//! "enhance my notes", follow-up email, chat about a meeting, and "ask across
//! all meetings". Each runs on the one notes model, one at a time.

use itsmypa_engine::chat::{self, Turn};
use itsmypa_engine::notes;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::api::ApiError;
use crate::state::AppState;

const NOT_LOADED: &str = "⚠️ The notes AI model isn't loaded — check Settings.";

/// Same placeholder the Python engine produced, so the notes cards still render.
const SUMMARY_NOT_LOADED: &str = "## Executive Summary\n⚠️ The notes AI model isn't loaded — check Settings.\n\n## Key Decisions\n- No decisions were recorded.\n\n## Action Items\n- [ ] No action items were recorded. — @Unassigned\n\n## Topics Discussed\n- N/A";

/// Trailer the "Ask" screen parses into clickable source chips.
const SOURCES_MARKER: &str = "\n@@SOURCES@@";

fn parse<T: serde::de::DeserializeOwned>(body: Option<Value>) -> Result<T, ApiError> {
    serde_json::from_value(body.unwrap_or(Value::Null)).map_err(|e| ApiError::new(400, format!("Invalid request: {e}")))
}

fn required(v: &str, what: &str) -> Result<(), ApiError> {
    if v.trim().is_empty() {
        Err(ApiError::new(400, format!("No {what} provided")))
    } else {
        Ok(())
    }
}

/// Run a streaming request. Input errors are returned before anything is
/// sent; after that, problems are reported inside the text, as before.
pub fn stream(state: &AppState, path: &str, body: Option<Value>, send: &mut dyn FnMut(&str) -> bool) -> Result<(), ApiError> {
    #[derive(Deserialize, Default)]
    #[serde(default)]
    struct Req {
        transcript: String,
        summary: String,
        notes: String,
        template: String,
        question: String,
        history: Vec<HistoryTurn>,
    }
    #[derive(Deserialize, Default)]
    #[serde(default)]
    struct HistoryTurn {
        role: String,
        content: String,
    }
    let req: Req = parse(body)?;

    // Validate first, so the screen gets a proper error instead of a stream.
    match path {
        "/summarize" => required(&req.transcript, "transcript")?,
        "/enhance" => {
            required(&req.transcript, "transcript")?;
            required(&req.notes, "notes")?;
        }
        "/generate/email" if req.summary.trim().is_empty() => required(&req.transcript, "summary or transcript")?,
        "/chat" => {
            required(&req.transcript, "transcript")?;
            required(&req.question, "question")?;
        }
        "/ask" => required(&req.question, "question")?,
        "/generate/email" => {}
        _ => return Err(ApiError::new(404, format!("{path} is not available"))),
    }

    // Greetings never wait for (or need) the model.
    if (path == "/chat" || path == "/ask") && chat::is_small_talk(&req.question) {
        send(if path == "/chat" { chat::CHAT_GREETING } else { itsmypa_engine::ask::ASK_GREETING });
        return Ok(());
    }

    let llm = state.models.llm.lock().unwrap();
    let Some(llm) = llm.as_ref() else {
        send(if path == "/summarize" { SUMMARY_NOT_LOADED } else { NOT_LOADED });
        return Ok(());
    };

    let result = match path {
        "/summarize" => {
            let template = if req.template.is_empty() { "general" } else { &req.template };
            notes::summarize(llm, &req.transcript, template, send).map(drop)
        }
        "/enhance" => notes::enhance(llm, &req.transcript, &req.notes, send).map(drop),
        "/generate/email" => {
            let source = if req.summary.trim().is_empty() { &req.transcript } else { &req.summary };
            notes::email(llm, source, send).map(drop)
        }
        "/chat" => {
            let history: Vec<Turn> = req
                .history
                .into_iter()
                .filter(|t| t.role == "user" || t.role == "assistant")
                .map(|t| Turn { from_user: t.role == "user", text: t.content })
                .collect();
            chat::ask_meeting(llm, &req.transcript, &req.question, &history, send).map(drop)
        }
        "/ask" => {
            let today = chrono::Local::now().format("%Y-%m-%d (%A)").to_string();
            state.asker.ask(llm, &state.store, &req.question, &today, send).map(|(_, sources)| {
                if !sources.is_empty() {
                    send(&format!("{SOURCES_MARKER}{}", json!(sources)));
                }
            })
        }
        _ => unreachable!("validated above"),
    };
    if let Err(e) = result {
        send(&format!("\n\n_Error: {e:#}_"));
    }
    Ok(())
}

/// A short title for a meeting (not streamed).
pub fn title(state: &AppState, body: Option<Value>) -> Result<Value, ApiError> {
    #[derive(Deserialize, Default)]
    #[serde(default)]
    struct Req {
        transcript: String,
    }
    let req: Req = parse(body)?;
    required(&req.transcript, "transcript")?;
    let llm = state.models.llm.lock().unwrap();
    let title = match llm.as_ref() {
        Some(llm) => notes::title(llm, &req.transcript)?,
        None => String::new(),
    };
    Ok(json!({"title": title}))
}
