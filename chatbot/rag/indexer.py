"""(Re)build the chatbot index from the research corpus plus Navapp uploads.

Every build writes a new numbered folder data/index/vNNN/ and then points
data/index/CURRENT at it. A running app keeps reading the old folder until it
reloads, so a build never breaks a session that is open (Windows cannot replace
a database file that is in use). Only the two newest versions are kept.

Embeddings are cached by text in data/index/embed_cache.sqlite, so a rebuild
only pays for chunks whose text or metadata changed.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .corpus import LIST_FIELDS, embed_text, header
from . import corrections
from .ingest import load_uploads
from .stats import build_stats_tables

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus" / "letters.jsonl"
INDEX = ROOT / "data" / "index"
CACHE = INDEX / "embed_cache.sqlite"
MODEL = "openai/text-embedding-3-large"
KEEP_VERSIONS = 2


def fold(s: str) -> str:
    s = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def current_dir() -> Path:
    """Folder of the index in use (falls back to the flat layout of the first build)."""
    ptr = INDEX / "CURRENT"
    if ptr.exists():
        d = INDEX / ptr.read_text(encoding="utf-8").strip()
        if (d / "chunks.db").exists():
            return d
    return INDEX


def all_chunks() -> tuple[list[dict], int]:
    """Research corpus + uploads; an uploaded letter replaces the corpus version."""
    uploads = load_uploads()
    replaced = {u["letter_id"] for u in uploads}
    base = []
    if CORPUS.exists():
        with CORPUS.open(encoding="utf-8") as fh:
            base = [c for c in (json.loads(line) for line in fh) if c["letter_id"] not in replaced]
    chunks = base + uploads
    corrections.apply(chunks)   # Fien's checked persons replace the automatic links of those letters
    return chunks, len(uploads)


# -- embedding cache ------------------------------------------------------------
def _key(text: str) -> str:
    return hashlib.sha1(f"{MODEL}\n{text}".encode("utf-8")).hexdigest()


def _cache() -> sqlite3.Connection:
    INDEX.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(CACHE)
    con.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB NOT NULL)")
    return con


def _seed_cache_from_first_build(con: sqlite3.Connection) -> None:
    """The first build (scripts/build_index.py, flat layout) predates the cache:
    reuse its vectors instead of paying for them again."""
    if con.execute("SELECT COUNT(*) FROM emb").fetchone()[0] or not (INDEX / "embeddings.npy").exists():
        return
    meta = json.loads((INDEX / "embeddings_meta.json").read_text(encoding="utf-8"))
    if meta.get("model") != MODEL or not CORPUS.exists():
        return
    emb = np.load(INDEX / "embeddings.npy")
    with CORPUS.open(encoding="utf-8") as fh:
        chunks = [json.loads(line) for line in fh]
    if len(chunks) != emb.shape[0]:
        return
    con.executemany("INSERT OR IGNORE INTO emb VALUES (?, ?)",
                    ((_key(embed_text(c)), emb[i].astype(np.float32).tobytes()) for i, c in enumerate(chunks)))
    con.commit()


def _embeddings(chunks: list[dict], progress=None) -> tuple[np.ndarray, int, float]:
    from .openrouter import embed
    con = _cache()
    _seed_cache_from_first_build(con)
    keys = [_key(embed_text(c)) for c in chunks]
    have = {}
    for i in range(0, len(keys), 900):
        part = keys[i:i + 900]
        have.update(con.execute(f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(part))})", part))
    todo = [i for i, k in enumerate(keys) if k not in have]
    cost = 0.0
    if todo:
        batches = [todo[i:i + 100] for i in range(0, len(todo), 100)]

        def work(batch: list[int]) -> tuple[list[int], list, float]:
            vecs, c = embed([embed_text(chunks[i]) for i in batch], MODEL)
            return batch, vecs, c

        with ThreadPoolExecutor(4) as ex:
            for n, (batch, vecs, c) in enumerate(ex.map(work, batches), 1):
                cost += c
                for i, v in zip(batch, vecs):
                    have[keys[i]] = np.asarray(v, dtype=np.float32).tobytes()
                con.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)",
                                ((keys[i], have[keys[i]]) for i in batch))
                con.commit()
                if progress:
                    progress(n / len(batches), f"Embedding {min(n * 100, len(todo))}/{len(todo)} new fragments")
    con.close()
    emb = np.stack([np.frombuffer(have[k], dtype=np.float32) for k in keys])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb.astype(np.float32), len(todo), cost


# -- build ------------------------------------------------------------------------
def _write_db(path: Path, chunks: list[dict]) -> None:
    con = sqlite3.connect(path)
    cols = ", ".join(f"{f} TEXT" for f in LIST_FIELDS)
    con.executescript(f"""
        CREATE TABLE chunks (
            rowid INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, letter_id TEXT, chunk INTEGER,
            sender TEXT, recipient TEXT, date TEXT, year INTEGER, archive TEXT,
            permalink TEXT, text_source TEXT, text TEXT, {cols}
        );
        CREATE INDEX idx_year ON chunks(year);
        CREATE VIRTUAL TABLE chunks_fts USING fts5(text_folded, content='', tokenize='trigram');
    """)
    for i, c in enumerate(chunks):
        con.execute(f"INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,{','.join('?' * len(LIST_FIELDS))})",
                    (i, c["id"], c["letter_id"], c["chunk"], c["sender"], c["recipient"], c["date"], c["year"],
                     c["archive"], c["permalink"], c["text_source"], c["text"],
                     *[json.dumps(c.get(f, []), ensure_ascii=False) for f in LIST_FIELDS]))
        con.execute("INSERT INTO chunks_fts(rowid, text_folded) VALUES (?,?)", (i, fold(header(c) + "\n" + c["text"])))
    con.commit()
    build_stats_tables(con, chunks)   # letters / persons / mentions tables for exact counts
    con.close()


def rebuild_index(progress=None) -> dict:
    """Build a new index version and switch to it. progress(fraction, message) is optional."""
    t0 = time.time()
    chunks, n_uploaded = all_chunks()
    if not chunks:
        raise RuntimeError("No letters to index.")
    if progress:
        progress(0.0, f"Preparing {len(chunks)} fragments")
    emb, n_new, cost = _embeddings(chunks, progress)
    versions = sorted(int(m.group(1)) for d in INDEX.glob("v*") if (m := re.fullmatch(r"v(\d+)", d.name)))
    vdir = INDEX / f"v{(versions[-1] + 1) if versions else 1:03d}"
    vdir.mkdir(parents=True)
    _write_db(vdir / "chunks.db", chunks)
    np.save(vdir / "embeddings.npy", emb)
    (vdir / "embeddings_meta.json").write_text(json.dumps(
        {"model": MODEL, "backend": "openrouter", "dim": int(emb.shape[1]), "rows": int(emb.shape[0])}), encoding="utf-8")
    (INDEX / "CURRENT").write_text(vdir.name, encoding="utf-8")
    for old in sorted(INDEX.glob("v*"))[:-KEEP_VERSIONS]:
        shutil.rmtree(old, ignore_errors=True)   # an old version still open elsewhere is left for next time
    stats = {"version": vdir.name, "chunks": len(chunks), "letters": len({c["letter_id"] for c in chunks}),
             "uploaded_chunks": n_uploaded, "newly_embedded": n_new, "cost_usd": round(cost, 4),
             "seconds": round(time.time() - t0, 1)}
    if progress:
        progress(1.0, "Done")
    return stats
