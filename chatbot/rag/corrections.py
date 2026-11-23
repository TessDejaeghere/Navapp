"""Fien's manual corrections of the person mentions (OpenRefine export), applied to the chatbot index.

The pipeline writes a mention CSV; Fien cleans it in OpenRefine and exports it again. That export has
one row per PER / PER_NOM mention, with her corrections in these columns:

  surface_v2       her identification: the nodegoat name ("Jean Victor Schnetz (1787-1870)"), another
                   name ("Aurèle Robert", "Wife of Charles-Auguste Van den Berghe"), or the mention as is
  ID_Nodegoat      nodegoat ID, when she filled it in (a nodegoat ID in Notes counts too)
  Notes            DEL / "Geen PER" (not a person), GRP (a group), ALG (generic), ART / LOC (an artwork or
                   place instead of a person), or free text
  Source_v2, Target_v2   corrected sender and recipient of the letter

import_export() turns such a file into normalised rows in data/corrections/person_mentions.jsonl. A new
export replaces the corrections of the letters it contains and keeps those of other letters (batches).
apply() runs at every index build: for a corrected letter, Fien's persons replace the automatic links.
"""
from __future__ import annotations

import collections
import io
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from .stats import GLOSSARY, fold, letter_key, strip_dates

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "data" / "corrections" / "person_mentions.jsonl"
LOG = ROOT / "data" / "corrections" / "imports.json"

REQUIRED = {"letter_id", "label", "surface_v2", "span_start", "span_end"}
NG_ID = re.compile(r"\bng[A-Za-z0-9]{20,}\b")
NOT_A_PERSON = re.compile(r"\bDEL\b|geen per|bijvoeg|werkwoord|niet leesbaar", re.I)
GROUP = re.compile(r"\bGRP\b", re.I)
GENERIC = re.compile(r"\bALG\b", re.I)
AS_ARTWORK = re.compile(r"^\s*ART\s*$", re.I)
AS_PLACE = re.compile(r"^\s*LOC\s*$", re.I)
STATUS_CODES = re.compile(r"\b(DEL|GRP|ALG|ART|LOC|ST)\b", re.I)


def doubt_note(notes: str) -> str:
    """The historian's own words in Notes (doubts, hints), without status codes and nodegoat IDs."""
    rest = STATUS_CODES.sub("", NG_ID.sub("", notes or ""))
    rest = re.sub(r"^[\s/,;:-]+|[\s/,;:-]+$", "", rest)
    return rest if len(re.sub(r"\W", "", rest)) >= 3 else ""


def _glossary() -> tuple[dict, dict, dict]:
    """name by id, id by folded full name, id by folded bare name (only names that are unique)."""
    g = pd.read_csv(GLOSSARY, dtype=str, encoding="utf-8-sig").fillna("")
    name_of, by_full, bare_ids = {}, {}, collections.defaultdict(set)
    for r in g.to_dict("records"):
        pid, name = r["nodegoat ID"].strip(), " ".join(r["Name"].split())
        if not (pid and name):
            continue
        name_of.setdefault(pid, name)
        by_full.setdefault(fold(name), pid)
        bare_ids[fold(strip_dates(name))].add(pid)
    by_bare = {k: next(iter(v)) for k, v in bare_ids.items() if len(v) == 1}
    return name_of, by_full, by_bare


def _looks_like_identification(v1: str, v2: str) -> bool:
    """Did Fien write a name (not just tidy the mention, e.g. 'M.me' -> 'Mme')?"""
    if fold(v1) == fold(v2):
        return False
    return "(" in v2 or (len(v2.split()) >= 2 and bool(re.search(r"[A-ZÀ-Ý][a-zà-ÿ]{2,}", v2)))


def read_export(data: bytes | str | Path) -> pd.DataFrame:
    raw = data if isinstance(data, (str, Path)) else io.BytesIO(data)
    df = pd.read_csv(raw, dtype=str, encoding="utf-8-sig").fillna("")
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"This is not an OpenRefine mention export: missing column(s) {', '.join(sorted(missing))}.")
    return df


def normalise(df: pd.DataFrame) -> tuple[list[dict], dict]:
    """Rows of the export -> one record per mention, plus a summary for the upload screen."""
    name_of, by_full, by_bare = _glossary()
    col = lambda r, c: str(r.get(c, "") or "").strip()  # noqa: E731
    out, tally = [], collections.Counter()
    for r in df.to_dict("records"):
        v1 = col(r, "surface_v1") or col(r, "Surface_OG") or col(r, "surface")
        v2 = " ".join(col(r, "surface_v2").split()) or v1
        notes, label = col(r, "Notes"), col(r, "label")
        pid = col(r, "ID_Nodegoat") if col(r, "ID_Nodegoat").startswith("ng") else ""
        pid = pid or (NG_ID.search(notes).group(0) if NG_ID.search(notes) else "")
        if label == "ART" or AS_ARTWORK.match(notes):
            status = "artwork"
        elif label == "LOC" or AS_PLACE.match(notes):
            status = "place"
        elif NOT_A_PERSON.search(notes):
            status = "deleted"
        elif GROUP.search(notes) or col(r, "ID_Nodegoat") == "GRP":
            status = "group"
        elif GENERIC.search(notes):
            status = "generic"
        else:
            pid = pid or by_full.get(fold(v2), "") or (by_bare.get(fold(strip_dates(v2)), "")
                                                     if len(strip_dates(v2).split()) >= 2 else "")
            status = "linked" if pid in name_of else ("named" if _looks_like_identification(v1, v2) else "open")
        person = name_of.get(pid, "") if status == "linked" else (v2 if status == "named" else "")
        mid = col(r, "suggested_person_id")
        machine = (name_of.get(mid) or col(r, "suggested_person_name")) if mid.startswith("ng") else ""
        tally[status] += 1
        out.append({"letter_key": letter_key(col(r, "letter_id")), "letter_id": col(r, "letter_id"),
                    "start": int(float(col(r, "span_start") or -1)), "end": int(float(col(r, "span_end") or -1)),
                    "label": label, "surface": col(r, "Surface_OG") or v1, "status": status,
                    "person": person, "person_id": pid if status == "linked" else "", "value": v2,
                    "sender": col(r, "Source_v2"), "recipient": col(r, "Target_v2"),
                    "context_before": col(r, "context_before")[-80:],
                    "note": doubt_note(notes), "machine_person": machine})
    summary = {"mentions": len(out), "letters": len({o["letter_key"] for o in out}),
               "persons": len({o["person_id"] for o in out if o["person_id"]}), **dict(tally)}
    return out, summary


def load(include_shared: bool = True) -> list[dict]:
    """Corrections imported here, plus Fien's published ones for letters not corrected here."""
    rows = []
    if STORE.exists():
        with STORE.open(encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    if include_shared:
        from .sharing import shared_person_mentions
        here = {r["letter_key"] for r in rows}
        rows += [r for r in shared_person_mentions() if r["letter_key"] not in here]
    return rows


def imports() -> list[dict]:
    return json.loads(LOG.read_text(encoding="utf-8")) if LOG.exists() else []


def _mention(r: dict) -> tuple:
    return r["letter_key"], r["start"], r["end"]


def _reading(r: dict) -> tuple:
    return r.get("status"), r.get("person"), r.get("note", "")


def with_history(records: list[dict], previous: list[dict], when: str, source: str, by: str) -> list[dict]:
    """Give every mention its history of readings: the machine's suggestion first, then every reading by the
    historian that differs from the one before. A re-import of an unchanged reading adds nothing."""
    prev = {_mention(p): p for p in previous}
    for r in records:
        entry = {"when": when, "by": by, "source": source, "status": r["status"], "person": r["person"],
                 "note": r.get("note", "")}
        old = prev.get(_mention(r))
        if old and old.get("history"):
            hist = list(old["history"])
            if _reading(old) != _reading(r):
                hist.append(entry)
            else:   # same reading: keep when and where it was first made
                r.update({k: old[k] for k in ("imported", "source") if k in old})
        else:
            hist = [{"when": "", "by": "machine", "source": "pipeline (automatic linking)",
                     "status": "linked" if r.get("machine_person") else "open", "person": r.get("machine_person", ""),
                     "note": ""}, entry]
        r["history"] = hist
        r.setdefault("imported", when)
        r.setdefault("source", source)
    return records


def save(records: list[dict], summary: dict, file_name: str, by: str = "historian") -> dict:
    """Replace the corrections of the letters in this export; keep the other letters and every earlier reading."""
    new_letters = {r["letter_key"] for r in records}
    previous = load(include_shared=False)
    previous += [r for r in load() if r["letter_key"] not in {p["letter_key"] for p in previous}]  # published ones
    records = with_history(records, previous, datetime.now().isoformat(timespec="seconds"), file_name, by)
    kept = [r for r in load(include_shared=False) if r["letter_key"] not in new_letters]
    STORE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in kept + records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(STORE)
    log = imports() + [{"file": file_name, "imported": datetime.now().isoformat(timespec="seconds"), **summary,
                        "replaced_letters": len(new_letters & {r["letter_key"] for r in kept}) if kept else 0}]
    LOG.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"letters_total": len({r["letter_key"] for r in kept + records}), **summary}


def import_export(data: bytes | str | Path, file_name: str) -> dict:
    records, summary = normalise(read_export(data))
    return save(records, summary, file_name)


def clear() -> None:
    for p in (STORE, LOG):
        p.unlink(missing_ok=True)


# -- applying ------------------------------------------------------------------------
def _ws(s: str) -> str:
    return " ".join((s or "").split())


def _mode(values: list[str]) -> str:
    values = [v for v in values if v]
    return collections.Counter(values).most_common(1)[0][0] if values else ""


def _in_chunk(m: dict, c: dict, n_chunks_with_surface: int) -> bool:
    """Is this mention inside this chunk? Exact by offset when the chunk knows its start, else by text."""
    if "start" in c:
        s, e = m["start"] - c["start"], m["end"] - c["start"]
        return 0 <= s and e <= len(c["text"]) and _ws(c["text"][s:e]) == _ws(m["surface"])
    t = _ws(c["text"])
    if _ws(m["surface"]) not in t:
        return False
    tail = " ".join(m["context_before"].split()[-3:])
    return n_chunks_with_surface == 1 or (bool(tail) and _ws(tail + " " + m["surface"]) in t)


def apply(chunks: list[dict], corrections: list[dict] | None = None) -> dict:
    """Overwrite the person fields (and sender/recipient) of corrected letters, in place."""
    corrections = load() if corrections is None else corrections
    by_letter = collections.defaultdict(list)
    for m in corrections:
        by_letter[m["letter_key"]].append(m)
    chunks_by_letter = collections.defaultdict(list)
    for c in chunks:
        chunks_by_letter[letter_key(c["letter_id"])].append(c)
    stats = collections.Counter()
    for key, ms in by_letter.items():
        cs = chunks_by_letter.get(key)
        if not cs:
            stats["letters_without_text"] += 1
            continue
        stats["letters"] += 1
        sender, recipient = _mode([m["sender"] for m in ms]), _mode([m["recipient"] for m in ms])
        for c in cs:
            c["persons_linked"], c["persons_mentioned"], c["persons_checked"] = [], [], []
            c["doubts"], c["provenance"] = [], []
            c["sender"], c["recipient"] = sender or c["sender"], recipient or c["recipient"]
        placed = set()
        for i, m in enumerate(ms):
            hits = sum(_ws(m["surface"]) in _ws(c["text"]) for c in cs)
            for c in cs:
                if not _in_chunk(m, c, hits):
                    continue
                placed.add(i)
                c["provenance"].append({
                    "surface": _ws(m["surface"]), "label": m["label"], "status": m["status"], "person": m["person"],
                    "machine": m.get("machine_person", ""), "note": m.get("note", ""),
                    "checked": (m.get("imported") or "")[:10], "source": m.get("source", ""),
                    "history": m.get("history", [])})
                if m.get("note"):
                    who = f"{_ws(m['surface'])} → {m['person']}" if m["person"] else _ws(m["surface"])
                    c["doubts"].append(f"{who}: {m['note']}")
                if m["status"] in ("linked", "named") and m["person"] not in c["persons_checked"]:
                    c["persons_checked"].append(m["person"])
                if m["label"] == "PER" and m["status"] not in ("deleted", "artwork", "place"):
                    if _ws(m["surface"]) not in c["persons_mentioned"]:
                        c["persons_mentioned"].append(_ws(m["surface"]))
                field = {"artwork": "artworks", "place": "places"}.get(m["status"])
                if field and _ws(m["value"]) not in c.get(field, []):
                    c.setdefault(field, []).append(_ws(m["value"]))
        stats["mentions_placed"] += len(placed)
        stats["mentions_not_found"] += len(ms) - len(placed)
    return dict(stats)
