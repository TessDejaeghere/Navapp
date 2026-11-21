"""Hybrid retrieval over the Navez letters (after VaidyaBot):
FTS5 trigram BM25 + dense embeddings, fused with reciprocal rank fusion,
filtered on metadata and optionally reranked with a multilingual cross-encoder.
"""
import json
import re
import sqlite3
import unicodedata
from functools import cached_property
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "data" / "index"
RRF_K = 60
RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MAX_CHUNKS_PER_LETTER = 3
from .corpus import LIST_FIELDS  # noqa: E402
COLUMNS = ["rowid", "id", "letter_id", "chunk", "sender", "recipient", "date", "year", "archive",
           "permalink", "text_source", "text"] + LIST_FIELDS

# Question words in Dutch, English and French that would only add noise to BM25.
STOPWORDS = set("""
wat wie waar wanneer waarom hoe welke welk zijn haar hun het een de der den van voor over met door bij naar
uit aan als dat die dit deze ook nog niet wel maar dan was waren heeft hebben had werd worden wordt schreef
schrijft brief brieven navez letter letters what who where when why how which with about from that this
were wrote write does did the and for his her their there des les une est sont dans pour par avec sur qui que
""".split())


def fold(s: str) -> str:
    s = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(c for c in s if not unicodedata.combining(c))


class Searcher:
    def __init__(self, use_reranker: bool = True):
        from .indexer import current_dir
        d = current_dir()
        self.version = d.name
        self.con = sqlite3.connect(d / "chunks.db", check_same_thread=False)
        have = {r[1] for r in self.con.execute("PRAGMA table_info(chunks)")}
        self.columns = [c for c in COLUMNS if c in have]   # older index versions lack newer fields
        self.emb = np.load(d / "embeddings.npy")
        self.meta = json.loads((d / "embeddings_meta.json").read_text(encoding="utf-8"))
        self.use_reranker = use_reranker

    # -- facets for the settings panel ----------------------------------------
    def correspondents(self, top: int = 40) -> list[str]:
        """Most frequent senders and recipients other than Navez, by letter count."""
        rows = self.con.execute("""
            SELECT name, COUNT(DISTINCT letter_id) n FROM (
                SELECT sender name, letter_id FROM chunks UNION ALL SELECT recipient, letter_id FROM chunks)
            WHERE name <> '' AND name NOT LIKE 'François-Joseph Navez%' GROUP BY name ORDER BY n DESC LIMIT ?""",
                                (top,)).fetchall()
        return [r[0] for r in rows]

    @cached_property
    def _surname_index(self) -> dict[str, set[str]]:
        """Folded surname -> full correspondent names, for spotting names in a question."""
        idx: dict[str, set[str]] = {}
        for (name,) in self.con.execute("SELECT DISTINCT sender FROM chunks UNION SELECT DISTINCT recipient FROM chunks"):
            bare = re.sub(r"\(.*?\)", "", name or "").strip()
            if not bare or bare.startswith("François-Joseph Navez"):
                continue
            surname = fold(bare.split()[-1]).strip("?.,")
            if len(surname) >= 4:
                idx.setdefault(surname, set()).add(name)
        return idx

    def correspondents_in(self, query: str) -> set[str]:
        words = set(re.findall(r"[\w'-]+", fold(query)))
        return {n for w in words for n in self._surname_index.get(w, ())}

    def year_range(self) -> tuple[int, int]:
        lo, hi = self.con.execute("SELECT MIN(year), MAX(year) FROM chunks WHERE year IS NOT NULL").fetchone()
        return int(lo), int(hi)

    # -- channels ---------------------------------------------------------------
    def bm25(self, query: str, limit: int) -> list[int]:
        words = [w for w in re.findall(r"[\w'-]+", fold(query)) if len(w) >= 3 and w not in STOPWORDS]
        if not words:
            return []
        match = " OR ".join('"' + w.replace('"', "") + '"' for w in dict.fromkeys(words))
        rows = self.con.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
                                "ORDER BY bm25(chunks_fts) LIMIT ?", (match, limit)).fetchall()
        return [r[0] for r in rows]

    def _query_vec(self, query: str) -> np.ndarray:
        if self.meta.get("backend") == "openrouter":
            from .openrouter import embed
            v = np.asarray(embed([query], self.meta["model"])[0][0], dtype=np.float32)
        else:
            v = self._embedder.encode([f"query: {query}"], normalize_embeddings=True)[0].astype(np.float32)
        return v / np.linalg.norm(v)

    def dense(self, query: str, limit: int) -> tuple[list[int], np.ndarray]:
        sims = self.emb @ self._query_vec(query)
        top = np.argpartition(-sims, min(limit, len(sims) - 1))[:limit]
        return [int(i) for i in top[np.argsort(-sims[top])]], sims

    def by_correspondent(self, names: set[str], sims: np.ndarray, limit: int) -> list[int]:
        """Chunks of letters from or to the correspondents named in the question, by similarity."""
        if not names:
            return []
        marks = ",".join("?" * len(names))
        rows = [r[0] for r in self.con.execute(
            f"SELECT rowid FROM chunks WHERE sender IN ({marks}) OR recipient IN ({marks})", (*names, *names))]
        return sorted(rows, key=lambda i: -sims[i])[:limit]

    @cached_property
    def _embedder(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(self.meta["model"])

    @cached_property
    def _reranker(self):
        # Local copy first (the online check fails behind some certificate setups),
        # then a normal download; without either, fall back to the fusion order.
        try:
            from huggingface_hub import snapshot_download
            from sentence_transformers import CrossEncoder
        except ImportError:
            return None
        last = None
        try:  # load from the cached folder itself, so no hub call is made at all
            return CrossEncoder(snapshot_download(RERANKER, local_files_only=True))
        except Exception as e:  # noqa: BLE001
            last = e
        try:
            return CrossEncoder(RERANKER)
        except Exception as e:  # noqa: BLE001
            last = e
        print(f"[search] reranker unavailable ({type(last).__name__}); using fusion scores")
        return None

    # -- filters ----------------------------------------------------------------
    @staticmethod
    def _passes(rec: dict, correspondent: str | None, years: tuple[int, int] | None,
                person: str | None, include_undated: bool = True) -> bool:
        if correspondent and correspondent not in (rec["sender"], rec["recipient"]):
            return False
        if years:
            if rec["year"] is None:
                if not include_undated:
                    return False
            elif not years[0] <= rec["year"] <= years[1]:
                return False
        if person:
            p = fold(person)
            names = (rec["persons_checked"] + rec["persons_linked"] + rec["persons_mentioned"]
                     + [rec["sender"], rec["recipient"]])
            if not any(p in fold(n) for n in names):
                return False
        return True

    # -- fusion -----------------------------------------------------------------
    def search(self, query: str, k: int = 8, pool: int = 80, correspondent: str | None = None,
               years: tuple[int, int] | None = None, person: str | None = None,
               include_undated: bool = True) -> list[dict]:
        rrf: dict[int, float] = {}
        filtered = bool(correspondent or years or person)
        dense_ranked, sims = self.dense(query, pool * (8 if filtered else 1))
        named = self.correspondents_in(query)
        for ranked in (self.bm25(query, pool * (4 if filtered else 1)), dense_ranked,
                       self.by_correspondent(named, sims, pool)):
            for rank, rowid in enumerate(ranked):
                rrf[rowid] = rrf.get(rowid, 0.0) + 1.0 / (RRF_K + rank + 1)

        recs, per_letter = [], {}
        for rowid in sorted(rrf, key=rrf.get, reverse=True):
            r = self._record(rowid)
            if not self._passes(r, correspondent, years, person, include_undated):
                continue
            if per_letter.get(r["letter_id"], 0) >= MAX_CHUNKS_PER_LETTER:
                continue
            per_letter[r["letter_id"]] = per_letter.get(r["letter_id"], 0) + 1
            r["fusion"] = rrf[rowid]
            recs.append(r)
            if len(recs) >= max(k * 3, 24):
                break

        reranker = self._reranker if self.use_reranker and recs else None
        if reranker is not None:
            scores = reranker.predict([(query, r["text"][:1500]) for r in recs])
            for r, s in zip(recs, scores):
                r["confidence"] = round(float(1.0 / (1.0 + np.exp(-float(s)))), 3)
            recs.sort(key=lambda r: r["confidence"], reverse=True)
        else:
            for r in recs:
                r["confidence"] = round(r["fusion"] / (2.0 / (RRF_K + 1)), 3)
        return recs[:k]

    def _record(self, rowid: int) -> dict:
        row = self.con.execute(f"SELECT {', '.join(self.columns)} FROM chunks WHERE rowid=?", (rowid,)).fetchone()
        rec = dict(zip(self.columns, row))
        for f in LIST_FIELDS:
            rec[f] = json.loads(rec[f]) if rec.get(f) else []
        return rec
