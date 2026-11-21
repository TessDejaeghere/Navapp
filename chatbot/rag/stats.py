"""Exact counts over the letter metadata, for questions the text search cannot answer.

Tables (written into every index version's chunks.db by indexer.py):
  persons         nodegoat person list (CLUSTERING_pipeline/glossary/PERSON_export_2708.csv)
  letters         one row per letter: Fien's nodegoat letters export (all archives), plus letters
                  that only exist in the chatbot (sent from Navapp, or missing from the export)
  letter_persons  persons linked to a letter by the pipeline (automatic suggestions)
  letter_entities places, institutions, artworks, events found by the pipeline

Counts of letters, senders, recipients and dates come from Fien's metadata and are exact.
Counts of mentions come from automatic entity recognition and linking on the transcribed
letters only, so they are estimates; every result says which basis it has.
"""
from __future__ import annotations

import csv
import re
import sqlite3
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent           # chatbot/
REPO = ROOT.parent                                        # Navez research repo / Navapp folder
GLOSSARY = REPO / "CLUSTERING_pipeline" / "glossary" / "PERSON_export_2708.csv"
# Newest first: an export Fien uploads on the "Chatbot data" page, then the repo export.
LETTER_EXPORTS = [REPO / "data" / "nodegoat_letters_export.csv", REPO / "data" / "nodegoat_export_2106.csv"]
NAVEZ = "François-Joseph Navez (1787-1869)"
BASIS_METADATA = "Fien's nodegoat metadata (exact, includes letters without a transcription)"
BASIS_MENTIONS = ("automatic entity recognition and person linking on the transcribed letters only "
                  "(an estimate: missed or wrongly linked names are possible)")
ENTITY_KINDS = ("places", "organisations", "artworks", "events")

csv.field_size_limit(10**9)


def fold(s: str) -> str:
    s = unicodedata.normalize("NFD", (s or "").replace("\n", " ").lower())
    return " ".join("".join(c for c in s if not unicodedata.combining(c)).split())


def strip_dates(s: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", (s or "").replace("\n", " ")).strip()


def year_of(date: str) -> int | None:
    m = re.match(r"^\s*(1[78]\d\d)\b", str(date or ""))
    return int(m.group(1)) if m else None


def letter_key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


# -- building -----------------------------------------------------------------------
def _load_persons() -> list[dict]:
    if not GLOSSARY.exists():
        return []
    with GLOSSARY.open(encoding="utf-8-sig", newline="") as f:
        out = []
        for r in csv.DictReader(f):
            name = " ".join((r.get("Name") or "").split())
            if r.get("nodegoat ID") and name:
                out.append({"person_id": r["nodegoat ID"], "name": name, "bare": strip_dates(name),
                            "born": (r.get("[date of birth] Date Start") or "")[:10],
                            "died": (r.get("[date of death] Date Start") or "")[:10]})
        return out


def _load_letter_export() -> tuple[list[dict], str]:
    for path in LETTER_EXPORTS:
        if path.exists():
            with path.open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.reader(f))
            h = rows[0]
            col = {name: h.index(name) for name in ("Name", "Source", "Target") if name in h}
            idate = next(i for i, c in enumerate(h) if c in ("Date", "catalogue_date"))
            iarch = next((i for i, c in enumerate(h) if "Name Archive" in c), None)
            iidn = next((i for i, c in enumerate(h) if c.endswith("IDN")), None)
            out = []
            for r in rows[1:]:
                if len(r) <= max(col.values()):
                    continue
                out.append({"letter_id": r[col["Name"]].strip(), "sender": r[col["Source"]].strip(),
                            "recipient": r[col["Target"]].strip(), "date": r[idate].strip(),
                            "archive": (r[iarch].strip() if iarch is not None else ""),
                            "idn": (r[iidn].strip() if iidn is not None else "")})
            return out, path.name
    return [], ""


def build_stats_tables(con: sqlite3.Connection, chunks: list[dict]) -> dict:
    persons = _load_persons()
    by_full, by_bare = {}, {}
    for p in persons:
        by_full.setdefault(fold(p["name"]), p["person_id"])
        by_bare.setdefault(fold(p["bare"]), p["person_id"])

    def pid(name: str) -> str:
        n = (name or "").strip()
        if not n:
            return ""
        return by_full.get(fold(n)) or by_bare.get(fold(strip_dates(n))) or f"name:{fold(n)}"

    con.executescript("""
        DROP TABLE IF EXISTS persons; DROP TABLE IF EXISTS letters;
        DROP TABLE IF EXISTS letter_persons; DROP TABLE IF EXISTS letter_entities;
        CREATE TABLE persons (person_id TEXT PRIMARY KEY, name TEXT, bare TEXT, born TEXT, died TEXT);
        CREATE TABLE letters (letter_key TEXT PRIMARY KEY, letter_id TEXT, sender TEXT, recipient TEXT,
            sender_id TEXT, recipient_id TEXT, date TEXT, year INTEGER, archive TEXT, idn TEXT,
            has_text INTEGER, text_source TEXT, origin TEXT);
        CREATE TABLE letter_persons (letter_key TEXT, person_id TEXT, person_name TEXT,
            PRIMARY KEY (letter_key, person_id));
        CREATE TABLE letter_entities (letter_key TEXT, kind TEXT, value TEXT, PRIMARY KEY (letter_key, kind, value));
        CREATE INDEX idx_l_sender ON letters(sender_id); CREATE INDEX idx_l_recipient ON letters(recipient_id);
        CREATE INDEX idx_lp_person ON letter_persons(person_id);
    """)
    con.executemany("INSERT OR IGNORE INTO persons VALUES (:person_id,:name,:bare,:born,:died)", persons)

    text = {}
    for c in chunks:
        t = text.setdefault(letter_key(c["letter_id"]), {"chunk": c, "persons": set(), "ents": set()})
        t["persons"].update(c.get("persons_linked", []))
        for kind in ENTITY_KINDS:
            t["ents"].update((kind, v) for v in c.get(kind, []))

    export, export_name = _load_letter_export()
    rows = {}
    for e in export:
        k = letter_key(e["letter_id"])
        if k and k not in rows:
            rows[k] = {**e, "origin": "nodegoat"}
    for k, t in text.items():   # letters the export does not know (e.g. sent from Navapp)
        c = t["chunk"]
        if k not in rows:
            rows[k] = {"letter_id": c["letter_id"], "sender": c["sender"], "recipient": c["recipient"],
                       "date": c["date"], "archive": c.get("archive", ""), "idn": "",
                       "origin": "navapp" if c.get("added") else "corpus"}
        elif c.get("added"):     # Fien re-processed a known letter: her metadata wins where filled in
            for f in ("sender", "recipient", "date"):
                rows[k][f] = c[f] or rows[k][f]
    con.executemany("INSERT INTO letters VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", [
        (k, r["letter_id"], r["sender"], r["recipient"], pid(r["sender"]), pid(r["recipient"]), r["date"],
         year_of(r["date"]), r["archive"], r["idn"], int(k in text),
         text[k]["chunk"]["text_source"] if k in text else "", r["origin"]) for k, r in rows.items()])
    con.executemany("INSERT OR IGNORE INTO letter_persons VALUES (?,?,?)",
                    [(k, pid(n), n) for k, t in text.items() for n in t["persons"] if pid(n)])
    con.executemany("INSERT OR IGNORE INTO letter_entities VALUES (?,?,?)",
                    [(k, kind, v) for k, t in text.items() for kind, v in t["ents"]])
    con.commit()
    return {"letters": len(rows), "export": export_name, "persons": len(persons)}


# -- querying -----------------------------------------------------------------------
def _person_label(con: sqlite3.Connection, person_id: str) -> str:
    row = con.execute("SELECT name FROM persons WHERE person_id=?", (person_id,)).fetchone()
    if row:
        return row[0]
    row = con.execute("SELECT sender FROM letters WHERE sender_id=? UNION SELECT recipient FROM letters "
                      "WHERE recipient_id=? LIMIT 1", (person_id, person_id)).fetchone()
    return row[0] if row else person_id


def find_person(con: sqlite3.Connection, name: str, limit: int = 6) -> dict:
    """Candidate persons for a name as written in a question, with their letter counts."""
    q = fold(strip_dates(name))
    q_tokens = [t for t in re.findall(r"[\w'-]+", q) if len(t) > 1]
    if not q_tokens:
        return {"query": name, "candidates": []}
    pool = {}
    for pid_, pname in con.execute("SELECT person_id, name FROM persons"):
        pool[pid_] = pname
    for pid_, pname in con.execute("SELECT sender_id, sender FROM letters UNION SELECT recipient_id, recipient FROM letters"):
        if pid_ and pid_ not in pool:
            pool[pid_] = pname
    scored = []
    for pid_, pname in pool.items():
        tokens = re.findall(r"[\w'-]+", fold(strip_dates(pname)))
        if not tokens:
            continue
        if fold(strip_dates(pname)) == q:
            score = 3.0
        elif all(any(t == x or (len(t) >= 4 and x.startswith(t)) for x in tokens) for t in q_tokens):
            score = 2.0 + (0.5 if tokens[-1] == q_tokens[-1] else 0)
        else:
            try:
                from rapidfuzz import fuzz
                s = fuzz.token_set_ratio(q, " ".join(tokens))
                score = 1.0 if s >= 88 else 0
            except ImportError:
                score = 0
        if score:
            scored.append((score, pid_, pname))
    out = []
    for score, pid_, pname in sorted(scored, key=lambda x: -x[0]):
        sent = con.execute("SELECT COUNT(*) FROM letters WHERE sender_id=?", (pid_,)).fetchone()[0]
        received = con.execute("SELECT COUNT(*) FROM letters WHERE recipient_id=?", (pid_,)).fetchone()[0]
        mentioned = con.execute("SELECT COUNT(*) FROM letter_persons WHERE person_id=?", (pid_,)).fetchone()[0]
        out.append({"person_id": pid_, "name": pname, "letters_sent": sent, "letters_received": received,
                    "letters_mentioning_estimate": mentioned, "_score": score})
    out.sort(key=lambda c: (-c["_score"], -(c["letters_sent"] + c["letters_received"] + c["letters_mentioning_estimate"])))
    for c in out:
        c.pop("_score")
    return {"query": name, "candidates": out[:limit]}


def _where(f: dict) -> tuple[str, list, list[str]]:
    """SQL WHERE over letters l from tool filters, plus a readable description."""
    clauses, params, desc = [], [], []
    for key, col, label in (("sender_id", "l.sender_id", "sender"), ("recipient_id", "l.recipient_id", "recipient")):
        if f.get(key):
            clauses.append(f"{col} = ?")
            params.append(f[key])
            desc.append(f"{label} = {f['_labels'].get(f[key], f[key])}")
    if f.get("correspondent_id"):
        clauses.append("(l.sender_id = ? OR l.recipient_id = ?)")
        params += [f["correspondent_id"]] * 2
        desc.append(f"sender or recipient = {f['_labels'].get(f['correspondent_id'])}")
    if f.get("mentions_id"):
        clauses.append("EXISTS (SELECT 1 FROM letter_persons m WHERE m.letter_key = l.letter_key AND m.person_id = ?)")
        params.append(f["mentions_id"])
        desc.append(f"mentions {f['_labels'].get(f['mentions_id'])} (estimate)")
    if f.get("archive"):
        clauses.append("l.archive LIKE ?")
        params.append(f"%{f['archive']}%")
        desc.append(f"archive contains '{f['archive']}'")
    if f.get("only_with_text"):
        clauses.append("l.has_text = 1")
        desc.append("only letters with a transcription")
    return (" AND ".join(clauses) or "1=1"), params, desc


def _years(f: dict) -> tuple[str, list, str]:
    y0, y1 = f.get("year_from"), f.get("year_to")
    if not (y0 or y1):
        return "", [], ""
    y0, y1 = int(y0 or 1700), int(y1 or 1900)
    return " AND l.year BETWEEN ? AND ?", [y0, y1], f"years {y0}–{y1}"


def _prepare(con, f: dict) -> dict:
    f = {k: v for k, v in (f or {}).items() if v not in (None, "", [])}
    f["_labels"] = {f[k]: _person_label(con, f[k]) for k in ("sender_id", "recipient_id", "correspondent_id",
                                                             "mentions_id") if f.get(k)}
    return f


def count_letters(con: sqlite3.Connection, **filters) -> dict:
    f = _prepare(con, filters)
    where, params, desc = _where(f)
    ysql, yparams, ydesc = _years(f)
    total = con.execute(f"SELECT COUNT(*), SUM(has_text) FROM letters l WHERE {where}{ysql}", params + yparams).fetchone()
    res = {"count": total[0], "with_transcription": int(total[1] or 0), "filters": desc + ([ydesc] if ydesc else []),
           "basis": BASIS_MENTIONS if f.get("mentions_id") else BASIS_METADATA}
    if ysql:   # undated letters cannot be placed in a period; say how many are left out
        res["undated_not_counted"] = con.execute(
            f"SELECT COUNT(*) FROM letters l WHERE {where} AND l.year IS NULL", params).fetchone()[0]
    else:
        res["of_which_undated"] = con.execute(
            f"SELECT COUNT(*) FROM letters l WHERE {where} AND l.year IS NULL", params).fetchone()[0]
    return res


def letters_per_year(con: sqlite3.Connection, **filters) -> dict:
    f = _prepare(con, filters)
    where, params, desc = _where(f)
    ysql, yparams, ydesc = _years(f)
    rows = con.execute(f"SELECT l.year, COUNT(*) FROM letters l WHERE {where}{ysql} AND l.year IS NOT NULL "
                       "GROUP BY l.year ORDER BY l.year", params + yparams).fetchall()
    undated = con.execute(f"SELECT COUNT(*) FROM letters l WHERE {where} AND l.year IS NULL", params).fetchone()[0]
    return {"per_year": [{"year": y, "letters": n} for y, n in rows], "undated": undated,
            "filters": desc + ([ydesc] if ydesc else []),
            "basis": BASIS_MENTIONS if f.get("mentions_id") else BASIS_METADATA}


def top_people(con: sqlite3.Connection, role: str = "sender", limit: int = 10, exclude_navez: bool = True,
               **filters) -> dict:
    f = _prepare(con, filters)
    where, params, desc = _where(f)
    ysql, yparams, ydesc = _years(f)
    base = f"FROM letters l WHERE {where}{ysql}"
    if role == "mentioned":
        # persons named in a letter, not counting that letter's own sender and recipient
        sql = (f"SELECT m.person_id, COUNT(*) n FROM letter_persons m JOIN letters l ON l.letter_key = m.letter_key "
               f"WHERE {where}{ysql} AND m.person_id <> COALESCE(l.sender_id,'') "
               f"AND m.person_id <> COALESCE(l.recipient_id,'') GROUP BY m.person_id")
    elif role == "correspondent":
        sql = (f"SELECT pid, COUNT(*) n FROM (SELECT l.sender_id pid {base} UNION ALL "
               f"SELECT l.recipient_id pid {base}) WHERE pid <> '' GROUP BY pid")
        params, yparams = params + yparams + params, yparams
    else:
        col = "l.recipient_id" if role == "recipient" else "l.sender_id"
        sql = f"SELECT {col}, COUNT(*) n {base} AND {col} <> '' GROUP BY {col}"
    navez = con.execute("SELECT person_id FROM persons WHERE name = ?", (NAVEZ,)).fetchone()
    rows = con.execute(sql + " ORDER BY n DESC LIMIT ?", params + yparams + [limit + 2]).fetchall()
    people = [{"person_id": p, "name": _person_label(con, p), "letters": n} for p, n in rows
              if not (exclude_navez and navez and p == navez[0])][:limit]
    return {"role": role, "people": people, "filters": desc + ([ydesc] if ydesc else []),
            "basis": BASIS_MENTIONS if role == "mentioned" or f.get("mentions_id") else BASIS_METADATA}


def top_entities(con: sqlite3.Connection, kind: str = "places", limit: int = 15, **filters) -> dict:
    f = _prepare(con, filters)
    where, params, desc = _where(f)
    ysql, yparams, ydesc = _years(f)
    rows = con.execute(f"SELECT e.value, COUNT(*) n FROM letter_entities e JOIN letters l ON l.letter_key = e.letter_key "
                       f"WHERE e.kind = ? AND {where}{ysql} GROUP BY lower(e.value) ORDER BY n DESC LIMIT ?",
                       [kind] + params + yparams + [limit]).fetchall()
    return {"kind": kind, "values": [{"value": v, "letters": n} for v, n in rows],
            "filters": desc + ([ydesc] if ydesc else []), "basis": BASIS_MENTIONS}


def list_letters(con: sqlite3.Connection, limit: int = 25, **filters) -> dict:
    f = _prepare(con, filters)
    where, params, desc = _where(f)
    ysql, yparams, ydesc = _years(f)
    n = con.execute(f"SELECT COUNT(*) FROM letters l WHERE {where}{ysql}", params + yparams).fetchone()[0]
    rows = con.execute(f"SELECT letter_id, date, sender, recipient, archive, has_text, idn FROM letters l "
                       f"WHERE {where}{ysql} ORDER BY l.year IS NULL, l.date LIMIT ?", params + yparams + [limit]).fetchall()
    letters = [{"letter_id": r[0], "date": r[1], "sender": r[2], "recipient": r[3], "archive": r[4],
                "transcribed": bool(r[5]),
                "kbr_link": f"https://opac.kbr.be/Library/doc/SYRACUSE/{r[6]}" if r[6] and r[4] == "KBR" else ""}
               for r in rows]
    return {"total": n, "shown": len(letters), "letters": letters, "filters": desc + ([ydesc] if ydesc else []),
            "basis": BASIS_MENTIONS if f.get("mentions_id") else BASIS_METADATA}


def overview(con: sqlite3.Connection) -> dict:
    n, t = con.execute("SELECT COUNT(*), SUM(has_text) FROM letters").fetchone()
    arch = con.execute("SELECT COALESCE(NULLIF(archive,''),'unknown'), COUNT(*) FROM letters GROUP BY 1 ORDER BY 2 DESC").fetchall()
    return {"letters": n, "with_transcription": int(t or 0), "archives": dict(arch)}
