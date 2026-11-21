"""Letters added from Navapp, kept apart from the research corpus.

data/corpus/uploads.jsonl holds the chunk records of every letter Fien sends to
the chatbot. A letter sent again replaces its earlier version, and an uploaded
letter overrides the research-corpus version with the same letter id.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "data" / "corpus" / "uploads.jsonl"


def load_uploads() -> list[dict]:
    if not UPLOADS.exists():
        return []
    with UPLOADS.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write(records: list[dict]) -> None:
    UPLOADS.parent.mkdir(parents=True, exist_ok=True)
    tmp = UPLOADS.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(UPLOADS)


def add_letters(chunks: list[dict]) -> list[str]:
    """Add or replace letters; returns the letter ids that were written."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    new_ids = {c["letter_id"] for c in chunks}
    kept = [r for r in load_uploads() if r["letter_id"] not in new_ids]
    for c in chunks:
        c["added"] = stamp
    _write(kept + chunks)
    return sorted(new_ids)


def remove_letters(letter_ids: list[str]) -> int:
    drop = set(letter_ids)
    records = load_uploads()
    kept = [r for r in records if r["letter_id"] not in drop]
    _write(kept)
    return len({r["letter_id"] for r in records}) - len({r["letter_id"] for r in kept})


def uploaded_letters() -> list[dict]:
    """One summary row per uploaded letter."""
    rows: dict[str, dict] = {}
    for r in load_uploads():
        row = rows.setdefault(r["letter_id"], {
            "letter_id": r["letter_id"], "sender": r["sender"], "recipient": r["recipient"],
            "date": r["date"], "transcription": r["text_source"], "chunks": 0, "added": r.get("added", "")})
        row["chunks"] += 1
    return sorted(rows.values(), key=lambda x: x["added"], reverse=True)
