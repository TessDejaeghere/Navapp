"""Test set for counting questions: does the chatbot report the right numbers?

Each case is a question plus the exact answer computed directly from the statistics tables
(Fien's nodegoat metadata), or an expectation that the bot asks which person is meant.
A case passes when the expected number appears in the answer (or, for 'ask', when the bot
names at least two candidates and gives no count). Costs roughly $0.03 per question.

Run:  python scripts/eval_counts.py            (all cases)
      python scripts/eval_counts.py 1 4 7       (selected cases)
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag import stats as S  # noqa: E402
from rag.agent import answer_with_tools  # noqa: E402
from rag.search import Searcher  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
s = Searcher(use_reranker=False)
con = s.con
pid = lambda name: S.find_person(con, name)["candidates"][0]["person_id"]  # noqa: E731
NAVEZ, BEAUVOIR, SCHNETZ = pid("François-Joseph Navez"), pid("Caillard de Beauvoir"), pid("Jean Victor Schnetz")
PORTAELS = pid("Jean François Portaels")
top_sender = S.top_people(con, "sender", recipient_id=NAVEZ, limit=1)["people"][0]
per_year = S.letters_per_year(con, recipient_id=NAVEZ)["per_year"]
peak = max(per_year, key=lambda x: x["letters"])

CASES = [
    ("Hoeveel brieven schreef Beauvoir?", S.count_letters(con, sender_id=BEAUVOIR)["count"]),
    ("How many letters did Beauvoir write to Navez?", S.count_letters(con, sender_id=BEAUVOIR, recipient_id=NAVEZ)["count"]),
    ("Hoeveel brieven schreef Schnetz aan Navez?", S.count_letters(con, sender_id=SCHNETZ, recipient_id=NAVEZ)["count"]),
    ("Hoeveel brieven ontving Navez in 1830?", S.count_letters(con, recipient_id=NAVEZ, year_from=1830, year_to=1830)["count"]),
    ("Wie schreef het vaakst brieven aan Navez, en hoeveel?", top_sender["letters"]),
    ("Hoeveel brieven van Navez zelf zijn er bewaard?", S.count_letters(con, sender_id=NAVEZ)["count"]),
    ("In welk jaar ontving Navez de meeste brieven?", peak["year"]),
    ("Hoeveel brieven schreef Portaels tussen 1843 en 1845?",
     S.count_letters(con, sender_id=PORTAELS, year_from=1843, year_to=1845)["count"]),
    ("Hoeveel brieven schreef Robert?", "ask"),
    ("Hoeveel brieven in de collectie zijn er in totaal, en hoeveel zijn getranscribeerd?",
     S.overview(con)["letters"]),
]

picked = [int(a) for a in sys.argv[1:]] or range(1, len(CASES) + 1)
passed, total_cost = 0, 0.0
for i in picked:
    q, expected = CASES[i - 1]
    res = answer_with_tools(q, s)
    total_cost += res["cost"]
    text = res["answer"]
    # the headline: numbers in the first bold phrase that has any outside parentheses (life dates)
    headline = set()
    for seg in re.findall(r"\*\*(.+?)\*\*", text):
        nums = re.findall(r"\d[\d., ]*", re.sub(r"\([^)]*\)", "", seg))
        if nums:
            headline = {int(re.sub(r"[., ]", "", n)) for n in nums}
            break
    if expected == "ask":
        ok = text.count("Robert") >= 3 and "?" in text and not res["stats"]
    else:
        ok = expected in headline
    passed += ok
    print(f"{'PASS' if ok else 'FAIL'} #{i} {q}\n     expected {expected} | tools {[t['tool'] for t in res['stats']]}"
          f"\n     {' '.join(text.split())[:220]}\n")
print(f"{passed}/{len(list(picked))} passed · cost ${total_cost:.3f}")
