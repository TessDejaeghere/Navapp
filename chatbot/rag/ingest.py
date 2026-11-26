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


# -- export of full transcriptions -------------------------------------------------
TEXT_STATUS = {"corrected_htr": "HTR, corrected", "manual": "manual transcription",
               "opus_htr": "HTR, not corrected"}


def letter_texts(chunks: list[dict]) -> list[dict]:
    """Put the overlapping chunks of each letter back together into its full text.

    Each chunk carries its offset in the letter ("start"), so overlapping chunks are
    joined at that offset. Older records without an offset are joined with a blank line.
    """
    by_letter: dict[str, list[dict]] = {}
    for c in chunks:
        by_letter.setdefault(c["letter_id"], []).append(c)
    out = []
    for lid, cs in by_letter.items():
        cs.sort(key=lambda c: (c.get("start", c.get("chunk", 0)), c.get("chunk", 0)))
        text = ""
        for c in cs:
            start = c.get("start")
            if start is None:
                text = f"{text}\n\n{c['text']}" if text else c["text"]
            elif start <= len(text):
                text = text[:start] + c["text"]
            else:                       # whitespace between two chunks was stripped
                text = text + "\n" * min(start - len(text), 2) + c["text"]
        first = cs[0]
        out.append({"letter_id": lid, "sender": first.get("sender", ""), "recipient": first.get("recipient", ""),
                    "date": first.get("date", ""), "archive": first.get("archive", ""),
                    "transcription": TEXT_STATUS.get(first.get("text_source", ""), first.get("text_source", "")),
                    "added": first.get("added", ""), "permalink": first.get("permalink", ""),
                    "text": text.strip()})
    return sorted(out, key=lambda r: r["letter_id"])


def _file_name(letter_id: str) -> str:
    keep = "".join(ch if ch.isalnum() or ch in " .-_" else "_" for ch in letter_id)
    return keep.strip() or "letter"


def transcriptions_zip(letters: list[dict]) -> bytes:
    """ZIP with one .txt per letter plus transcriptions.csv (metadata and full text)."""
    import csv
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        used: set[str] = set()
        for r in letters:
            name, n = _file_name(r["letter_id"]), 2
            base = name
            while name in used:
                name, n = f"{base}_{n}", n + 1
            used.add(name)
            z.writestr(f"transcriptions/{name}.txt", r["text"] + "\n")
        sheet = io.StringIO()
        cols = ["letter_id", "sender", "recipient", "date", "archive", "transcription", "added", "permalink", "text"]
        w = csv.DictWriter(sheet, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(letters)
        z.writestr("transcriptions.csv", "﻿" + sheet.getvalue())
    return buf.getvalue()
