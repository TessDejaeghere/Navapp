"""Assemble the Navez research corpus into retrieval chunks with metadata and entities.

Output: chatbot/data/corpus/letters.jsonl, one JSON record per chunk
(letters Fien adds from Navapp go to uploads.jsonl instead; see rag/ingest.py).

Sources (all in the Navez research repo):
  - letter texts      NER_pipeline.predict_dalembert_corpus.load_corpus()
                      (corrected HTR > manual transcription > raw Claude HTR)
  - metadata          data/letters_overview.csv (sender, recipient, date, KBR IDN)
  - person mentions   data/ner_corpus/per_mentions_openrefine.csv (with suggested links)
  - other entities    data/ner_corpus/dalembert_corpus_spans.jsonl (LOC, ORG, ART, EVT)

Entity offsets are only used when they still match the letter text; letters whose
text changed after the NER run keep their metadata but lose their entity fields.
"""
from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

CHATBOT = Path(__file__).resolve().parents[1]
REPO = CHATBOT.parent
sys.path.insert(0, str(REPO / "NER_pipeline"))
sys.path.insert(0, str(CHATBOT))
from predict_dalembert_corpus import load_corpus, norm_key  # noqa: E402
from rag.corpus import OTHER_LABELS, make_chunks  # noqa: E402

OVERVIEW = REPO / "data" / "letters_overview.csv"
MENTIONS = REPO / "data" / "ner_corpus" / "per_mentions_openrefine.csv"
SPANS = REPO / "data" / "ner_corpus" / "dalembert_corpus_spans.jsonl"
OUT = CHATBOT / "data" / "corpus" / "letters.jsonl"
PERMALINK = "https://opac.kbr.be/Library/doc/SYRACUSE/{idn}"

csv.field_size_limit(10**9)


def load_metadata() -> dict[str, dict]:
    meta: dict[str, dict] = {}
    with OVERVIEW.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            key = norm_key(r.get("Ms_key", ""))
            if not key or key in meta:
                continue
            idn = (r.get("IDN") or "").strip()
            meta[key] = {"sender": r.get("Source", ""), "recipient": r.get("Target", ""),
                         "date": r.get("Date", ""), "archive": r.get("Archive", ""),
                         "permalink": PERMALINK.format(idn=idn) if idn else ""}
    return meta


def load_person_mentions() -> dict[str, list[dict]]:
    by_letter: dict[str, list[dict]] = collections.defaultdict(list)
    with MENTIONS.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            by_letter[norm_key(r["letter_id"])].append({
                "start": int(r["span_start"]), "end": int(r["span_end"]),
                "surface": r["surface"], "label": r["label"],
                "person": r["suggested_person_name"] if r["suggested_person_id"] not in ("", "NIL") else "",
            })
    return by_letter


def load_other_spans() -> dict[str, list[tuple[int, int, str]]]:
    out: dict[str, list[tuple[int, int, str]]] = {}
    with SPANS.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[norm_key(r["letter_id"])] = [(s, e, lab) for s, e, lab in r["pred_spans"] if lab in OTHER_LABELS]
    return out


def main() -> None:
    letters, meta = load_corpus(), load_metadata()
    persons, others = load_person_mentions(), load_other_spans()
    stats = collections.Counter()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for L in letters:
            key, text = L["letter_id"], L["text"]
            pm = persons.get(key, [])
            if all(text[p["start"]:p["end"]] == p["surface"] for p in pm):
                om = others.get(key, [])
            else:
                stats["letters_entities_dropped"] += 1
                pm, om = [], []
            stats["letters"] += 1
            for rec in make_chunks(key, text, meta.get(key, {}), pm, om, L["text_source"]):
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["chunks"] += 1
    print(dict(stats), "->", OUT)


if __name__ == "__main__":
    main()
