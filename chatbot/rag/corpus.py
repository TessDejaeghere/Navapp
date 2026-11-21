"""Turn one letter (text + metadata + entities) into retrieval chunks.

Shared by scripts/build_corpus.py (the whole research corpus) and by Navapp's
"send to chatbot" step (letters Fien processes herself), so both produce
identical records.
"""
from __future__ import annotations

import re

CHUNK_CHARS = 1600      # letters are short: most fit in one chunk
OVERLAP_CHARS = 250
OTHER_LABELS = {"LOC": "places", "ORG": "organisations", "ART": "artworks", "EVT": "events"}
LIST_FIELDS = ["persons_linked", "persons_checked", "persons_mentioned", "places", "organisations", "artworks", "events"]


def year_of(date: str) -> int | None:
    m = re.search(r"\b(1[78]\d\d)\b", str(date or ""))
    return int(m.group(1)) if m else None


def clean(s: str) -> str:
    return " ".join((s or "").split()).strip(" ,;:.")


# Split NER spans leave articles and possessives behind as "entities" ("l", "son", "de").
FRAGMENT_WORDS = {"l", "le", "la", "les", "d", "de", "du", "des", "un", "une", "son", "sa", "ses", "mon", "ma",
                  "mes", "ton", "ta", "tes", "votre", "vos", "notre", "nos", "leur", "leurs", "ce", "cet", "cette"}


def useful_entity(s: str) -> bool:
    words = re.findall(r"\w+", (s or "").lower())
    return bool(words) and len("".join(words)) >= 3 and not all(w in FRAGMENT_WORDS for w in words)


def unique(items) -> list[str]:
    seen, out = set(), []
    for x in items:
        k = (x or "").lower()
        if x and k not in seen:
            seen.add(k)
            out.append(x)
    return out


def split_text(text: str) -> list[tuple[int, int]]:
    """Character ranges of overlapping chunks, cut at line breaks where possible."""
    if len(text) <= CHUNK_CHARS + OVERLAP_CHARS:
        return [(0, len(text))]
    ranges, start = [], 0
    while start < len(text):
        end = min(len(text), start + CHUNK_CHARS)
        if end < len(text):
            cut = text.rfind("\n", start + CHUNK_CHARS // 2, end)
            if cut == -1:
                cut = text.rfind(" ", start + CHUNK_CHARS // 2, end)
            end = cut if cut > start else end
        ranges.append((start, end))
        if end >= len(text):
            break
        start = max(end - OVERLAP_CHARS, start + 1)
        nxt = text.find(" ", start)
        start = nxt + 1 if 0 <= nxt < end else start
    return ranges


def make_chunks(letter_id: str, text: str, meta: dict, persons: list[dict],
                other_spans: list[tuple[int, int, str]], text_source: str) -> list[dict]:
    """Chunk records for one letter.

    meta         {sender, recipient, date, archive, permalink}
    persons      [{start, end, surface, label ('PER'/'PER_NOM'), person (linked name or '')}]
    other_spans  [(start, end, label)] with labels from OTHER_LABELS
    """
    out = []
    for n, (a, b) in enumerate(split_text(text)):
        inside = lambda s, e: s >= a and e <= b  # noqa: E731
        per = [p for p in persons if p["label"] == "PER" and inside(p["start"], p["end"])]
        rec = {
            "id": f"{letter_id}#{n}", "letter_id": letter_id, "chunk": n,
            "sender": clean(meta.get("sender", "")), "recipient": clean(meta.get("recipient", "")),
            "date": clean(str(meta.get("date", "") or "")), "year": year_of(meta.get("date", "")),
            "archive": clean(meta.get("archive", "")), "permalink": meta.get("permalink", "") or "",
            "text_source": text_source, "text": text[a:b].strip(),
            "start": a + len(text[a:b]) - len(text[a:b].lstrip()),   # offset of the chunk in the letter
            # suggested links are machine output: shown as such in the apps
            "persons_linked": unique(p["person"] for p in per if p.get("person")),
            "persons_mentioned": unique(clean(p["surface"]) for p in per),
        }
        for lab, field in OTHER_LABELS.items():
            rec[field] = unique(x for x in (clean(text[s:e]) for s, e, l in other_spans if l == lab and inside(s, e))
                                if useful_entity(x))
        out.append(rec)
    return out


def header(c: dict) -> str:
    """Metadata line prepended to each chunk for search and embedding."""
    parts = [f"Letter from {c['sender'] or 'unknown sender'} to {c['recipient'] or 'unknown recipient'}",
             f"date {c['date'] or 'unknown'}"]
    persons = c.get("persons_checked", []) + c["persons_linked"] + c["persons_mentioned"]
    if persons:
        parts.append("persons: " + ", ".join(persons))
    for f in ("places", "organisations", "artworks", "events"):
        if c[f]:
            parts.append(f"{f}: " + ", ".join(c[f]))
    return " | ".join(parts)


def embed_text(c: dict) -> str:
    return header(c) + "\n" + c["text"]
