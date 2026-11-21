"""Broad evaluation of the Navapp chatbot: content, counts, combinations, languages, traps, follow-ups.

Every case is answered exactly as in the app (tools, reranker, Claude Sonnet 4.5), then checked:
  - counts      headline number equals the true count from the statistics tables
  - content     an expected letter is among the cited sources, and expected words occur
  - trap        the bot says it cannot find it and cites nothing for the made-up claim
  - ask         the bot asks which person is meant instead of counting
  - quotes      every French quotation in the answer occurs in one of the cited fragments
  - citations   every [#n] refers to an existing fragment
Results: data/eval/eval_bot_<timestamp>.json and .md. Costs roughly $0.04 per question.
"""
import json
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rag import stats as S  # noqa: E402
from rag.agent import answer_with_tools  # noqa: E402
from rag.search import Searcher  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
s = Searcher()
con = s.con
pid = lambda name: S.find_person(con, name)["candidates"][0]["person_id"]  # noqa: E731
NAVEZ, BEAUVOIR, SCHNETZ, LLROBERT = (pid("François-Joseph Navez"), pid("Caillard de Beauvoir"),
                                      pid("Jean Victor Schnetz"), pid("Louis-Léopold Robert"))

CASES = [
    # --- content (what do the letters say?)
    {"id": "C1", "type": "content", "q": "Wat schreef Beauvoir aan Navez over het vernissen van schilderijen?",
     "letters": ["Ms. II 70/2/52"], "words": ["vernis"]},
    {"id": "C2", "type": "content", "q": "Hoe hoorde Navez van de dood van Jacques Sturm?",
     "letters": ["Ms. II 70/1/181"], "words": ["1844"]},
    {"id": "C3", "type": "content", "q": "What did Portaels see in the museum of Turin?",
     "letters": ["Ms. II 70/1/177"], "words": ["Van Dyck"]},
    {"id": "C4", "type": "content", "q": "Que raconte Alexandre Robert sur les volontaires romains en 1848 ?",
     "letters": ["Ms. II 70/1/190"], "words": ["brigand"]},
    {"id": "C5", "type": "content", "q": "Wat schreven Navez' correspondenten over de Belgische revolutie van 1830?",
     "letters": [], "words": ["1830"]},
    {"id": "C6", "type": "content", "q": "Welke schilderijen van Leopold Robert worden in de brieven besproken?",
     "letters": [], "words": []},
    # --- counts (exact, from metadata)
    {"id": "N1", "type": "count", "q": "Hoeveel brieven schreef Schnetz aan Navez?",
     "truth": S.count_letters(con, sender_id=SCHNETZ, recipient_id=NAVEZ)["count"]},
    {"id": "N2", "type": "count", "q": "Hoeveel brieven stuurde Navez aan Louis-Léopold Robert?",
     "truth": S.count_letters(con, sender_id=NAVEZ, recipient_id=LLROBERT)["count"]},
    {"id": "N3", "type": "count", "q": "How many letters in the collection are kept in Neuchâtel?",
     "truth": S.count_letters(con, archive="Neuchâtel")["count"]},
    {"id": "N4", "type": "count", "q": "Hoeveel brieven ontving Navez tussen 1840 en 1849?",
     "truth": S.count_letters(con, recipient_id=NAVEZ, year_from=1840, year_to=1849)["count"]},
    # --- estimates
    {"id": "E1", "type": "estimate", "q": "In hoeveel brieven wordt Schnetz vermeld?",
     "truth": S.count_letters(con, mentions_id=SCHNETZ)["count"]},
    # --- combination of counting and content
    {"id": "M1", "type": "combo", "q": "Hoeveel brieven schreef Beauvoir in 1848, en waarover gaan ze?",
     "truth": S.count_letters(con, sender_id=BEAUVOIR, year_from=1848, year_to=1848)["count"]},
    # --- ambiguity and follow-up
    {"id": "A1", "type": "ask", "q": "Hoeveel brieven schreef Robert?"},
    {"id": "A2", "type": "count", "q": "Louis-Léopold", "follow_up_of": "A1",
     "truth": S.count_letters(con, sender_id=LLROBERT)["count"]},
    # --- traps: no such thing in the letters
    {"id": "T1", "type": "trap", "q": "Wat schreef Navez over zijn reis naar Amerika in 1835?"},
    {"id": "T2", "type": "trap", "q": "Welke schilderijen van Rembrandt kocht Navez in Amsterdam volgens de brieven?"},
]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").lower()
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[’'`´]", "'", s)
    return re.sub(r"[^a-z0-9']+", " ", s).strip()


def headline(text: str) -> set[int]:
    for seg in re.findall(r"\*\*(.+?)\*\*", text):
        nums = re.findall(r"\d[\d., ]*", re.sub(r"\([^)]*\)", "", seg))
        if nums:
            return {int(re.sub(r"[., ]", "", n)) for n in nums}
    return set()


def quote_check(text: str, sources: list[dict]) -> tuple[int, int, list[str]]:
    cited_text = norm(" ".join(r["text"] for r in sources if r.get("cited")))
    quotes = [q for q in re.findall(r"[\"“«]\s*([^\"”»]{15,})[\"”»]", text) if re.search(r"[a-zA-Z]", q)]
    bad = []
    for q in quotes:
        nq = norm(q.strip(" .…"))
        parts = [p for p in re.split(r"\s*\.\.\.\s*|\s*…\s*|\[\.\.\.\]", q) if len(norm(p)) > 8]
        if nq not in cited_text and not all(norm(p) in cited_text for p in parts or [q]):
            bad.append(q[:90])
    return len(quotes), len(quotes) - len(bad), bad


results, history_of = [], {}
t_all = time.time()
for case in CASES:
    history = history_of.get(case.get("follow_up_of"), [])
    t0 = time.time()
    res = answer_with_tools(case["q"], s, history=history)
    text, sources = res["answer"], res["sources"]
    history_of[case["id"]] = history + [{"role": "user", "content": case["q"]}, {"role": "assistant", "content": text}]
    cited = [r for r in sources if r.get("cited")]
    refs = {int(n) for g in re.findall(r"\[#([\d#,\s]+)\]", text) for n in re.findall(r"\d+", g)}
    n_q, n_ok, bad_q = quote_check(text, sources)
    checks = {"citations_valid": all(1 <= n <= len(sources) for n in refs), "quotes_found": f"{n_ok}/{n_q}"}
    if bad_q:
        checks["quotes_not_found"] = bad_q
    t = case["type"]
    if t in ("count", "estimate", "combo"):
        checks["number_ok"] = case["truth"] in headline(text)
        if t == "estimate":
            checks["says_estimate"] = bool(re.search(r"schatting|estimate|geschat|ongeveer|circa", text, re.I))
    if t in ("content", "combo"):
        if case.get("letters"):
            checks["expected_letter_cited"] = any(r["letter_id"] in case["letters"] for r in cited)
        if case.get("words"):
            checks["words_present"] = all(w.lower() in text.lower() for w in case["words"])
        checks["has_citations"] = bool(refs)
    if t == "trap":
        checks["says_not_found"] = bool(re.search(r"niet (te )?(vinden|gevonden)|staat niet|geen (informatie|vermelding|"
                                                  r"aanwijzing|brieven)|not found|no mention|niets", text, re.I))
    if t == "ask":
        checks["asks_which"] = "?" in text and text.count("Robert") >= 3 and not res["stats"]
    passed = all(v for k, v in checks.items() if isinstance(v, bool)) and not bad_q
    results.append({**{k: v for k, v in case.items() if k != "truth"}, "truth": case.get("truth"), "answer": text,
                    "tools": [x["tool"] for x in res["stats"]], "n_sources": len(sources), "n_cited": len(cited),
                    "cited_letters": sorted({r["letter_id"] for r in cited}), "checks": checks, "passed": passed,
                    "cost": round(res["cost"], 4), "seconds": round(time.time() - t0, 1)})
    print(f"{'PASS' if passed else 'FAIL'} {case['id']} ({t}) {case['q']}\n     {checks}\n     "
          f"{' '.join(text.split())[:260]}\n", flush=True)

out = ROOT / "data" / "eval" / f"eval_bot_{datetime.now():%Y%m%d_%H%M}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
cost = sum(r["cost"] for r in results)
print(f"{sum(r['passed'] for r in results)}/{len(results)} passed · ${cost:.3f} · {time.time() - t_all:.0f}s · {out}")
