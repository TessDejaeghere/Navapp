"""Where every batch of letters stands, between the pipeline, OpenRefine and the chatbot.

A batch is one pipeline run (Step 3 on the Pipeline page). The work on it is spread over days and over
two programs: the mention CSV goes to OpenRefine, the historian checks it there, her export comes back
on the Chatbot data page, and at some point she publishes. This module keeps that state in a small
SQLite file (chatbot/data/batches.sqlite, never touched by updates), so that Navapp can show which batch
waits on whom, and can check an OpenRefine export against the batch it belongs to.

Tables
  batches        one row per run: id (the run stamp), created, letters, mentions
  batch_letters  the letters of a run, with their number of person mentions
  events         dated steps: 'csv_out' (mention CSV downloaded for OpenRefine), 'corrections_in'
                 (her export loaded), 'published' (her corrections published; batch_id is NULL)
Whether a letter is in the chatbot and whether it has checked persons is read live from the chatbot
data, so it can never disagree with what the chatbot actually uses.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "batches.sqlite"
CORPUS = ROOT / "data" / "corpus" / "letters.jsonl"
UPLOADS = ROOT / "data" / "corpus" / "uploads.jsonl"

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (id TEXT PRIMARY KEY, created TEXT NOT NULL,
                                    letters INTEGER NOT NULL, mentions INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS batch_letters (batch_id TEXT NOT NULL, letter_key TEXT NOT NULL,
                                          letter_id TEXT NOT NULL, mentions INTEGER NOT NULL,
                                          PRIMARY KEY (batch_id, letter_key));
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, batch_id TEXT, stage TEXT NOT NULL,
                                   at TEXT NOT NULL, detail TEXT NOT NULL DEFAULT '');
"""


def key(letter_id: str) -> str:
    """Separator/case-insensitive letter id: `Ms. II 70_1_23` == `Ms. II 70/1/23`."""
    return re.sub(r"[\s/_\-]+", " ", (letter_id or "").strip().lower())


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def connect(db: Path | None = None):
    """Open the database, commit on success, and always close it (on Windows an open file stays locked)."""
    path = db or DB
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    try:
        con.row_factory = sqlite3.Row
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


# -- writing -----------------------------------------------------------------------
def register_run(stamp: str, created: str, mentions_per_letter: dict[str, int], db: Path | None = None) -> None:
    """Add a run (or leave it as it is when it is already known). Keys: letter ids as written."""
    with connect(db) as con:
        if con.execute("SELECT 1 FROM batches WHERE id = ?", (stamp,)).fetchone():
            return
        con.execute("INSERT INTO batches VALUES (?, ?, ?, ?)",
                    (stamp, created, len(mentions_per_letter), sum(mentions_per_letter.values())))
        con.executemany("INSERT OR REPLACE INTO batch_letters VALUES (?, ?, ?, ?)",
                        [(stamp, key(lid), lid, int(n)) for lid, n in mentions_per_letter.items()])


def sync_runs(runs_dir: Path, db: Path | None = None) -> int:
    """Register every saved run in data/pipeline_runs that is not in the database yet (e.g. older runs)."""
    import csv

    if not runs_dir.exists():
        return 0
    with connect(db) as con:
        known = {r[0] for r in con.execute("SELECT id FROM batches")}
    added = 0
    for js in sorted(runs_dir.glob("*.json")):
        if js.stem in known:
            continue
        try:
            meta = json.loads(js.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        counts = Counter({L["letter_id"]: 0 for L in meta.get("letters", [])})
        csv_path = js.with_suffix(".csv")
        if csv_path.exists():
            with csv_path.open(encoding="utf-8-sig", newline="") as fh:
                counts.update(r["letter_id"] for r in csv.DictReader(fh) if r.get("letter_id"))
        register_run(js.stem, meta.get("when", js.stem), dict(counts), db)
        added += 1
    return added


def mark(stage: str, batch_ids: list[str] | None = None, detail: str = "", db: Path | None = None) -> None:
    """Record a dated step for some batches, or a global one (batch_ids=None)."""
    with connect(db) as con:
        rows = [(b, stage, _now(), detail) for b in batch_ids] if batch_ids else [(None, stage, _now(), detail)]
        con.executemany("INSERT INTO events (batch_id, stage, at, detail) VALUES (?, ?, ?, ?)", rows)


def batches_with_letters(letter_ids, db: Path | None = None) -> list[str]:
    """The latest batch of each of these letters (a letter run twice belongs to its newest run)."""
    keys = {key(x) for x in letter_ids}
    return sorted({b for k, b in latest_batch_of(db).items() if k in keys})


# -- reading -----------------------------------------------------------------------
def latest_batch_of(db: Path | None = None) -> dict[str, str]:
    """{letter key -> id of the newest batch that contains it}."""
    with connect(db) as con:
        rows = con.execute("SELECT bl.letter_key, bl.batch_id FROM batch_letters bl "
                           "JOIN batches b ON b.id = bl.batch_id ORDER BY b.created").fetchall()
    return {r["letter_key"]: r["batch_id"] for r in rows}


def chatbot_letter_keys() -> set[str]:
    """Letters the chatbot holds text for: the research corpus plus the letters sent from the pipeline."""
    out: set[str] = set()
    for path in (CORPUS, UPLOADS):
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        out.add(key(json.loads(line).get("letter_id", "")))
                    except json.JSONDecodeError:
                        continue
    return out


def overview(corrected_keys: set[str], chatbot_keys: set[str], db: Path | None = None) -> list[dict]:
    """One row per batch, newest first, with what is done and what it waits on."""
    latest = latest_batch_of(db)
    with connect(db) as con:
        batches = con.execute("SELECT * FROM batches ORDER BY created DESC").fetchall()
        letters = con.execute("SELECT batch_id, letter_key, mentions FROM batch_letters").fetchall()
        events = con.execute("SELECT batch_id, stage, MAX(at) AS at FROM events GROUP BY batch_id, stage").fetchall()
    last = {(e["batch_id"], e["stage"]): e["at"] for e in events}
    published = last.get((None, "published"), "")
    by_batch: dict[str, list[str]] = {}
    no_mentions: set[tuple[str, str]] = set()
    for r in letters:
        by_batch.setdefault(r["batch_id"], []).append(r["letter_key"])
        if not r["mentions"]:
            no_mentions.add((r["batch_id"], r["letter_key"]))
    rows = []
    for b in batches:
        keys = by_batch.get(b["id"], [])
        own = [k for k in keys if latest.get(k) == b["id"]]   # letters not run again later
        n = len(own)
        # a letter without person mentions has nothing to check, so it never appears in an export
        n_corr = sum(k in corrected_keys or (b["id"], k) in no_mentions for k in own)
        n_chat = sum(k in chatbot_keys for k in own)
        csv_out = last.get((b["id"], "csv_out"), "")
        corr_in = last.get((b["id"], "corrections_in"), "")
        is_published = bool(published and corr_in and published >= corr_in and n_corr)
        if n == 0:
            status = "all letters run again in a later batch"
        elif n_corr < n and not csv_out:
            status = "next: download the mention CSV for OpenRefine"
        elif n_corr < n:
            status = (f"⏳ with the historian in OpenRefine since {csv_out[:10]}"
                      + (f" ({n_corr} of {n} letters back)" if n_corr else ""))
        elif n_chat < n:
            status = "next: send to the chatbot (Step 4)"
        elif not is_published:
            status = "next: publish the corrections (Chatbot data page)"
        else:
            status = "✅ done"
        rows.append({"batch": b["id"], "run": b["created"].replace("T", " ")[:16], "letters": n,
                     "mentions": b["mentions"], "CSV to OpenRefine": csv_out[:10] or "—",
                     "checked": f"{n_corr}/{n}", "in chatbot": f"{n_chat}/{n}",
                     "published": published[:10] if is_published else "—", "status": status})
    return rows


def check_export(records: list[dict], db: Path | None = None) -> dict:
    """Compare an OpenRefine export with the batches it belongs to.

    records: normalised rows (corrections.normalise), one per mention, with 'letter_id'.
    Returns the batches it covers, the letters with fewer rows than their batch has mentions (a filtered
    export would silently drop the readings of the missing mentions), and the letters outside any batch.
    """
    got = Counter(key(r["letter_id"]) for r in records)
    ids = {key(r["letter_id"]): r["letter_id"] for r in records}
    latest = latest_batch_of(db)
    with connect(db) as con:
        expected = {(r["batch_id"], r["letter_key"]): r["mentions"]
                    for r in con.execute("SELECT batch_id, letter_key, mentions FROM batch_letters")}
        created = {r["id"]: r["created"] for r in con.execute("SELECT id, created FROM batches")}
        size = Counter(r["batch_id"] for r in con.execute("SELECT batch_id FROM batch_letters"))
    covered: Counter = Counter()
    short, outside = [], []
    for k, n in got.items():
        b = latest.get(k)
        if not b:
            outside.append(ids[k])
            continue
        covered[b] += 1
        exp = expected.get((b, k), 0)
        if n < exp:
            short.append({"letter_id": ids[k], "in export": n, "in batch": exp, "batch": b})
    batches = [{"batch": b, "run": created[b].replace("T", " ")[:16], "letters in export": c,
                "letters in batch": size[b]} for b, c in covered.most_common()]
    return {"batches": batches, "short": sorted(short, key=lambda x: x["letter_id"]), "outside": sorted(outside)}
