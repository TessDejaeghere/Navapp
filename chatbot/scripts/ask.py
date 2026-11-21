"""Command-line test: python scripts/ask.py "vraag" [--correspondent NAME] [--from 1820 --to 1830] [--no-llm]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.answer import answer  # noqa: E402
from rag.search import Searcher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("query")
ap.add_argument("--correspondent")
ap.add_argument("--person")
ap.add_argument("--from", dest="y0", type=int)
ap.add_argument("--to", dest="y1", type=int)
ap.add_argument("-k", type=int, default=6)
ap.add_argument("--no-llm", action="store_true")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")
filters = {"correspondent": a.correspondent, "person": a.person,
           "years": (a.y0, a.y1) if a.y0 and a.y1 else None}
s = Searcher()
if a.no_llm:
    for i, r in enumerate(s.search(a.query, k=a.k, **filters), 1):
        print(f"#{i} {r['confidence']:.2f} {r['letter_id']} | {r['sender']} -> {r['recipient']} | {r['date']}\n   {r['text'][:160]!r}")
else:
    res = answer(a.query, s, k=a.k, **filters)
    print(res["answer"], "\n\ncost $%.4f" % res["cost"])
    for i, r in enumerate(res["sources"], 1):
        print(f"[#{i}] {'✓' if r.get('cited') else '·'} {r['letter_id']} | {r['sender']} | {r['date']}")
