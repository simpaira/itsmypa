"""ItsMyPA API + pipeline unit tests.

Run:  ITSMYPA_SKIP_MODELS=1 .venv/bin/python -m pytest tests/ -q

ITSMYPA_SKIP_MODELS=1 keeps imports fast: no model downloads, no ONNX or
GGUF loading. Endpoints that need models are tested for their 503 behavior.
"""
import json
import os
import uuid

os.environ["ITSMYPA_SKIP_MODELS"] = "1"

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as ms


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "NOTES_FILE", tmp_path / "notes.json")
    monkeypatch.setattr(ms, "_notes", [])
    monkeypatch.setattr(ms, "RECORDINGS_DIR", tmp_path / "recordings")
    monkeypatch.setattr(ms, "ENV_FILE", tmp_path / ".env")
    ms.RECORDINGS_DIR.mkdir()
    return TestClient(ms.app)


# ── health ─────────────────────────────────────────────────────────────────────
def test_health_reports_model_state(client):
    d = client.get("/health").json()
    assert d["asr_ready"] is False          # models skipped in tests
    assert d["model_ready"] is False
    assert "ffmpeg" in d
    assert d["engine"] == "sherpa-onnx"


# ── setup progress ─────────────────────────────────────────────────────────────
def test_setup_status_reports_pending_when_models_skipped(client):
    d = client.get("/setup/status").json()
    assert d["ready"] is False              # asr/llm never loaded in test mode
    ids = {c["id"] for c in d["components"]}
    assert ids == {"vad", "asr", "seg", "emb", "llm"}
    assert all(c["status"] == "pending" for c in d["components"])
    assert d["overall_percent"] == 0.0


def test_setup_status_ready_when_required_components_terminal(client):
    import app as ms
    try:
        for key in ("asr", "llm"):
            ms._setup_set(key, status="ready")
        d = client.get("/setup/status").json()
        assert d["ready"] is True           # only asr + llm are required
        assert d["has_error"] is False
    finally:
        for key in ("asr", "llm"):
            ms._setup_set(key, status="pending", bytes=0, total=0, error=None)


def test_setup_status_flags_errors_without_blocking_forever(client):
    import app as ms
    try:
        ms._setup_set("asr", status="error", error="boom")
        ms._setup_set("llm", status="ready")
        d = client.get("/setup/status").json()
        assert d["ready"] is True           # error is terminal — don't hang the UI
        assert d["has_error"] is True
    finally:
        ms._setup_set("asr", status="pending", bytes=0, total=0, error=None)
        ms._setup_set("llm", status="pending", bytes=0, total=0, error=None)


# ── notes CRUD ─────────────────────────────────────────────────────────────────
def test_notes_crud_roundtrip(client):
    note = client.post("/notes", json={"title": "T1", "transcript": "hello world"}).json()
    assert note["title"] == "T1"

    assert [n["id"] for n in client.get("/notes").json()] == [note["id"]]

    r = client.patch(f"/notes/{note['id']}", json={"title": "T2", "summary": "## Executive Summary\nx"})
    assert r.json()["title"] == "T2"
    assert r.json()["summary"].startswith("## Executive")

    # transcript/segments edits persist
    segs = [{"speaker": "Speaker 1", "text": "hi", "start": 0.0, "end": 1.0}]
    r = client.patch(f"/notes/{note['id']}", json={"transcript": "hi", "segments": segs})
    assert r.json()["segments"] == segs

    # disallowed fields are ignored
    r = client.patch(f"/notes/{note['id']}", json={"id": "evil", "date": "1999"})
    assert r.json()["id"] == note["id"]

    assert client.delete(f"/notes/{note['id']}").json() == {"ok": True}
    assert client.get("/notes").json() == []


def test_patch_unknown_note_404(client):
    assert client.patch("/notes/nope", json={"title": "x"}).status_code == 404


# ── jobs ───────────────────────────────────────────────────────────────────────
def test_transcribe_without_models_returns_503(client):
    r = client.post("/transcribe", files={"audio": ("a.wav", b"RIFF0000", "audio/wav")})
    assert r.status_code == 503


def test_unknown_job_404(client):
    assert client.get(f"/jobs/{uuid.uuid4()}").status_code == 404


# ── crash-safe recordings ──────────────────────────────────────────────────────
def test_recording_backup_lifecycle(client):
    rec_id = str(uuid.uuid4())

    # two appended chunks accumulate on disk
    for chunk in (b"aaaa", b"bbbb"):
        assert client.post(f"/recordings/{rec_id}/chunk",
                           files={"audio": ("c.webm", chunk, "audio/webm")}).status_code == 200
    listing = client.get("/recordings").json()
    assert listing[0]["id"] == rec_id and listing[0]["bytes"] == 8

    # recover requires models — 503 here, and the file must survive
    assert client.post(f"/recordings/{rec_id}/recover").status_code == 503
    assert client.get("/recordings").json() != []

    # finish (or discard) removes the backup
    client.post(f"/recordings/{rec_id}/finish")
    assert client.get("/recordings").json() == []


def test_recording_bad_id_rejected(client):
    r = client.post("/recordings/../../etc/passwd/chunk",
                    files={"audio": ("c.webm", b"x", "audio/webm")})
    assert r.status_code in (400, 404)  # path traversal must not reach the filesystem


# ── settings ───────────────────────────────────────────────────────────────────
def test_settings_roundtrip_writes_env(client):
    d = client.get("/settings").json()
    assert d["whisper_model"] == ms.WHISPER_MODEL

    r = client.post("/settings", json={"whisper_model": "medium", "diarize": False,
                                       "num_speakers": 4, "whisper_language": "en"})
    assert r.json()["restart_required"] is True
    env = ms.ENV_FILE.read_text()
    assert "WHISPER_MODEL=medium" in env
    assert "DIARIZE=0" in env
    assert "NUM_SPEAKERS=4" in env
    assert "WHISPER_LANGUAGE=en" in env

    # updating a key rewrites it in place instead of appending a duplicate
    client.post("/settings", json={"whisper_model": "small"})
    env = ms.ENV_FILE.read_text()
    assert env.count("WHISPER_MODEL=") == 1
    assert "WHISPER_MODEL=small" in env


def test_settings_rejects_bad_values(client):
    assert client.post("/settings", json={"whisper_model": "gpt-5"}).status_code == 400
    assert client.post("/settings", json={"whisper_language": "not-a-code"}).status_code == 400
    assert client.post("/settings", json={"num_speakers": 99}).status_code == 400
    assert client.post("/settings", json={}).status_code == 400


# ── backup / restore ───────────────────────────────────────────────────────────
def test_export_import_roundtrip(client):
    a = client.post("/notes", json={"title": "A", "transcript": "aaa"}).json()
    b = client.post("/notes", json={"title": "B", "transcript": "bbb"}).json()

    r = client.get("/export")
    assert "attachment" in r.headers["content-disposition"]
    backup = r.json()
    assert {n["title"] for n in backup["notes"]} == {"A", "B"}

    # wipe, then import restores both; importing again skips duplicates
    client.delete(f"/notes/{a['id']}")
    client.delete(f"/notes/{b['id']}")
    r = client.post("/import", files={"backup": ("b.json", json.dumps(backup), "application/json")})
    assert r.json() == {"ok": True, "imported": 2, "skipped": 0}
    r = client.post("/import", files={"backup": ("b.json", json.dumps(backup), "application/json")})
    assert r.json()["imported"] == 0 and r.json()["skipped"] == 2
    assert len(client.get("/notes").json()) == 2


def test_import_rejects_garbage(client):
    assert client.post("/import", files={"backup": ("x.json", b"not json", "application/json")}).status_code == 400
    assert client.post("/import", files={"backup": ("x.json", b'{"foo": 1}', "application/json")}).status_code == 400


# ── scratchpad / enhance ───────────────────────────────────────────────────────
def test_notes_store_scratchpad_and_enhanced(client):
    n = client.post("/notes", json={"title": "T", "scratchpad": "pricing concerns",
                                    "enhanced": "### pricing concerns\n- discussed"}).json()
    assert n["scratchpad"] == "pricing concerns"
    r = client.patch(f"/notes/{n['id']}", json={"enhanced": "### updated"})
    assert r.json()["enhanced"] == "### updated"


def test_enhance_requires_inputs(client):
    assert client.post("/enhance", json={"notes": "x"}).status_code == 400
    assert client.post("/enhance", json={"transcript": "x"}).status_code == 400


# ── calendar ───────────────────────────────────────────────────────────────────
SAMPLE_ICS = b"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//test//test//EN
BEGIN:VEVENT
UID:weekly-standup@test
DTSTART:20260101T100000Z
DTEND:20260101T101500Z
RRULE:FREQ=WEEKLY
SUMMARY:Weekly standup
END:VEVENT
BEGIN:VEVENT
UID:allday@test
DTSTART;VALUE=DATE:20260706
SUMMARY:All-day thing
END:VEVENT
END:VCALENDAR
"""

def test_parse_ics_expands_recurrence_and_skips_allday():
    from datetime import datetime, timedelta, timezone
    start = datetime(2026, 7, 6, tzinfo=timezone.utc)
    events = ms._parse_ics(SAMPLE_ICS, start, start + timedelta(days=7))
    assert len(events) == 1                      # one weekly occurrence, all-day skipped
    assert events[0]["title"] == "Weekly standup"
    assert events[0]["start"].startswith("2026-07-09T10:00")


def test_calendar_unconfigured(client, monkeypatch):
    monkeypatch.setattr(ms, "CALENDAR_ICS_URL", "")
    assert client.get("/calendar/upcoming").json()["configured"] is False


def test_settings_calendar_url_applies_live(client):
    r = client.post("/settings", json={"calendar_ics_url": "webcal://example.com/cal.ics"})
    assert r.json()["restart_required"] is False              # calendar applies immediately
    assert ms.CALENDAR_ICS_URL == "https://example.com/cal.ics"
    assert "CALENDAR_ICS_URL=https://example.com/cal.ics" in ms.ENV_FILE.read_text()
    assert client.post("/settings", json={"calendar_ics_url": "ftp://nope"}).status_code == 400
    client.post("/settings", json={"calendar_ics_url": ""})   # reset


# ── pure pipeline helpers ──────────────────────────────────────────────────────
def test_chunk_words_splits_evenly():
    text = " ".join(str(i) for i in range(10))
    assert ms.chunk_words(text, 4) == ["0 1 2 3", "4 5 6 7", "8 9"]


def test_select_chat_context_passthrough_when_short():
    assert ms.select_chat_context("short transcript", "anything?") == "short transcript"


def test_select_chat_context_picks_relevant_chunks():
    filler = "word " * 200
    needle = "the budget approval deadline is friday " * 5
    transcript = filler * 30 + needle + filler * 30
    out = ms.select_chat_context(transcript, "when is the budget approval deadline?", max_words=1000)
    assert "budget approval" in out
    assert len(out.split()) < 1500


def test_group_spans_merges_same_speaker_and_respects_cap():
    sr = ms.SAMPLE_RATE
    samples = np.zeros(sr * 100, dtype=np.float32)
    spans = [(0.0, 5.0), (5.5, 10.0), (11.0, 15.0), (16.0, 40.0), (40.5, 50.0)]
    speakers = [0, 0, 1, 1, 1]
    chunks = ms._group_spans(spans, speakers, samples)

    # speaker change forces a split; same speaker within gap+cap merges
    assert [c[0] for c in chunks][:2] == [0, 1]
    for sp, start, end, _ in chunks:
        assert end - start <= ms.MAX_CHUNK_SEC + 1e-6
    # merged windows slice the right samples
    sp, start, end, chunk = chunks[0]
    assert len(chunk) == int(end * sr) - int(start * sr)


def test_chat_smalltalk_never_reaches_llm(client):
    for msg in ("hi", "Hello!", "thanks", "how are you?", "ok"):
        r = client.post("/chat", json={"transcript": "Speaker 1: we ship Friday.", "question": msg})
        assert r.status_code == 200
        assert "Ask me anything about this meeting" in r.text
        # must NOT summarize or error about the missing model
        assert "model" not in r.text.lower()


def test_ask_smalltalk_never_reaches_retrieval(client):
    for msg in ("hi", "hey!!", "thank you", "good morning"):
        r = client.post("/ask", json={"question": msg})
        assert r.status_code == 200
        assert "your saved meetings" in r.text
        assert "@@SOURCES@@" not in r.text


def test_real_questions_bypass_smalltalk_guard():
    # These must go to the pipeline, not the canned greeting.
    for q in ("hi, what did we decide about pricing?", "who said thanks to the client?",
              "what are the action items"):
        assert not ms._SMALLTALK_RE.match(q)
