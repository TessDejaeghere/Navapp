"""Command-line test of the tool-using answerer: python scripts/ask_tools.py "vraag" """
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.agent import answer_with_tools  # noqa: E402
from rag.search import Searcher  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
res = answer_with_tools(sys.argv[1], Searcher())
print(res["answer"])
print("\n--- tools:", [(s["tool"], s["args"]) for s in res["stats"]])
print("--- sources:", len(res["sources"]), "| cost $%.4f" % res["cost"])
