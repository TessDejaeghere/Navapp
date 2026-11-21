"""What the chatbot remembers: corrections by the research team, and saved conversations.

Corrections (data/memory/notes.jsonl)
    When a researcher corrects the bot on a fact ("that letter is by Alexandre Robert, not
    Louis-Léopold"), the agent stores it with remember_correction. Every later answer gets the
    corrections relevant to the question and to the letters it read, and they take precedence over
    metadata, automatic links and transcriptions. They can be edited or deleted on the Chatbot data page.

Conversations (data/memory/conversations.sqlite)
    Every conversation in the Navapp chat page is saved and can be reopened from the sidebar.
"""
from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEMORY = ROOT / "data" / "memory"
NOTES = MEMORY / "notes.jsonl"
CONVERSATIONS = MEMORY / "conversations.sqlite"
ALL_NOTES_UP_TO = 30     # with few corrections, every answer simply sees all of them
MAX_NOTES = 15

_STOP = set("""de het een en van dat die niet is was zijn met voor aan als ook door over naar bij maar dan
wie wat waar welke brief brieven navez the and of to in that this letter letters was not is are from with
le la les de des du un une et est que qui dans pour par pas sur lettre lettres""".split())


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def _terms(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{3,}", _fold(s)) if w not in _STOP}


# -- corrections ---------------------------------------------------------------------
def load_notes(include_deleted: bool = False) -> list[dict]:
    if not NOTES.exists():
        return []
    with NOTES.open(encoding="utf-8") as fh:
        notes = [json.loads(line) for line in fh if line.strip()]
    return notes if include_deleted else [n for n in notes if not n.get("deleted")]


def _write(notes: list[dict]) -> None:
    MEMORY.mkdir(parents=True, exist_ok=True)
    tmp = NOTES.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for n in notes:
            fh.write(json.dumps(n, ensure_ascii=False) + "\n")
    tmp.replace(NOTES)


def add_note(fact: str, letter_ids: list[str] | None = None, persons: list[str] | None = None,
             author: str = "", question: str = "", conversation_id: str = "") -> dict:
    fact = " ".join((fact or "").split())
    if len(fact) < 10:
        raise ValueError("A correction needs a full sentence.")
    note = {"id": uuid.uuid4().hex[:10], "created": _now(), "updated": "", "author": author or "research team",
            "fact": fact, "letter_ids": [x.strip() for x in (letter_ids or []) if x and x.strip()],
            "persons": [x.strip() for x in (persons or []) if x and x.strip()],
            "question": question[:500], "conversation_id": conversation_id}
    _write(load_notes(include_deleted=True) + [note])
    return note


def update_note(note_id: str, **fields) -> None:
    notes = load_notes(include_deleted=True)
    for n in notes:
        if n["id"] == note_id:
            n.update({k: v for k, v in fields.items() if k in ("fact", "letter_ids", "persons", "deleted")})
            n["updated"] = _now()
    _write(notes)


def delete_note(note_id: str) -> None:
    update_note(note_id, deleted=True)   # kept in the file, so a mistaken delete can be undone by hand


def relevant_notes(text: str, letter_ids: list[str] | None = None, limit: int = MAX_NOTES) -> list[dict]:
    """Corrections that concern this question or these letters (all of them while there are few)."""
    notes = load_notes()
    if len(notes) <= ALL_NOTES_UP_TO:
        return notes
    q, letters = _terms(text), {_fold(x) for x in (letter_ids or [])}
    scored = []
    for n in notes:
        s = 5 * len(letters & {_fold(x) for x in n["letter_ids"]})
        s += len(q & _terms(" ".join([n["fact"], *n["persons"], *n["letter_ids"]])))
        if s:
            scored.append((s, n["created"], n))
    return [n for *_, n in sorted(scored, key=lambda t: (t[0], t[1]), reverse=True)[:limit]]


def notes_for_letter(letter_id: str) -> list[dict]:
    key = _fold(letter_id)
    return [n for n in load_notes() if key in {_fold(x) for x in n["letter_ids"]}]


def notes_prompt(notes: list[dict]) -> str:
    if not notes:
        return ""
    lines = [f"- [{n['created'][:10]}, {n['author']}] {n['fact']}"
             + (f" (brieven: {', '.join(n['letter_ids'])})" if n["letter_ids"] else "") for n in notes]
    return ("\n\nCORRECTIES VAN HET ONDERZOEKSTEAM (onthouden uit eerdere gesprekken). Deze gaan voor op de "
            "metadata, de automatische koppelingen en de transcripties. Zeg het als je antwoord op zo'n "
            "correctie steunt:\n" + "\n".join(lines))


# -- conversations ---------------------------------------------------------------------
def _db() -> sqlite3.Connection:
    MEMORY.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(CONVERSATIONS)
    con.execute("CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, started TEXT, updated TEXT, "
                "title TEXT, turns TEXT)")
    return con


def new_conversation_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]


def save_conversation(conv_id: str, turns: list[dict]) -> None:
    if not turns:
        return
    title = next((t["content"] for t in turns if t["role"] == "user"), "")[:80]
    with _db() as con:
        con.execute("INSERT INTO conversations VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                    "updated=excluded.updated, title=excluded.title, turns=excluded.turns",
                    (conv_id, _now(), _now(), title, json.dumps(turns, ensure_ascii=False, default=str)))


def list_conversations(limit: int = 50) -> list[dict]:
    with _db() as con:
        rows = con.execute("SELECT id, updated, title FROM conversations ORDER BY updated DESC LIMIT ?",
                           (limit,)).fetchall()
    return [{"id": i, "updated": u, "title": t} for i, u, t in rows]


def load_conversation(conv_id: str) -> list[dict]:
    with _db() as con:
        row = con.execute("SELECT turns FROM conversations WHERE id=?", (conv_id,)).fetchone()
    return json.loads(row[0]) if row else []


def delete_conversation(conv_id: str) -> None:
    with _db() as con:
        con.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
