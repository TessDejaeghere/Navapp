"""Answering with tools: text search for content questions, exact metadata queries for counts.

The model decides per question which tools to call (several if needed), then writes the
answer. Fragments from text search are numbered [#n] for citation, as in answer.py;
statistics results are returned separately so the app can show them as tables and charts.
"""
from __future__ import annotations

import json
import re

from .answer import SOURCE_LABEL, _context_block
from .openrouter import chat_with_tools
from .search import Searcher
from . import stats

MAX_ROUNDS = 6

SYSTEM = """Je bent een onderzoeksassistent voor de briefwisseling van de Belgische schilder \
François-Joseph Navez (1787-1869). Je hebt twee soorten gereedschap:

A. search_letters: zoekt brieffragmenten op inhoud. Gebruik dit voor vragen over WAT er in de \
brieven staat. Citeer elke bewering over de inhoud als [#n].
B. Statistiek (find_person, count_letters, letters_per_year, top_people, top_entities, \
list_letters, collection_overview): exacte tellingen op Fiens nodegoat-metadata. Gebruik dit \
voor ELKE vraag naar aantallen, lijsten, wie het meest, wanneer, trends of verhoudingen.

STRIKTE REGELS:
1. Tel NOOIT zelf op basis van brieffragmenten: gebruik altijd de statistiek-tools.
2. Zet namen eerst om met find_person en gebruik de person_id in de andere tools. Zijn er \
meerdere plausibele kandidaten (bv. "Robert": verschillende personen met brieven) en maakt de \
vraag niet duidelijk wie bedoeld is, tel dan NIET maar vraag welke persoon bedoeld wordt, met de \
kandidaten (naam, jaartallen, aantal brieven). Is er één duidelijke kandidaat, ga dan verder.
3. Vermeld bij elk getal welke filters je gebruikte en waarop het steunt: tellingen van brieven, \
afzenders, ontvangers en datums zijn exact (metadata); tellingen van vermeldingen zijn \
SCHATTINGEN (automatische herkenning, alleen getranscribeerde brieven) en moet je zo noemen.
4. Bij een periode: meld ook hoeveel ongedateerde brieven buiten de telling vallen \
(undated_not_counted), zeker als dat er veel zijn.
5. "Brieven van X" = X is afzender; "brieven aan X" = X is ontvanger; "correspondentie met X" = \
beide richtingen. Zonder andere aanwijzing gaat het over de hele collectie (alle archieven); \
vermeld eventueel apart hoeveel ervan getranscribeerd zijn.
6. Voor inhoudsvragen gelden de gewone regels: alleen op basis van fragmenten, citeer [#n], \
korte Franse citaten met vertaling, ongecorrigeerde HTR en gekoppelde personen voorzichtig.
7. Gebruik voor elk getal in je antwoord een tabel-tool (count_letters, letters_per_year, \
top_people, top_entities of list_letters), ook als find_person al aantallen toont: de \
onderzoeker krijgt die tabellen te zien als bewijs.
8. Gebruik search_letters hoogstens twee keer per vraag.
9. Begin meteen met het antwoord, zonder zinnen over je werkwijze ("Nu heb ik…").
10. Antwoord in de taal van de vraag, helder en beknopt."""

MAX_SEARCHES = 2

_FILTERS = {
    "sender_id": {"type": "string", "description": "person_id of the sender (from find_person)"},
    "recipient_id": {"type": "string", "description": "person_id of the recipient"},
    "correspondent_id": {"type": "string", "description": "person_id as sender OR recipient"},
    "mentions_id": {"type": "string", "description": "person_id mentioned in the letter (estimate)"},
    "year_from": {"type": "integer"}, "year_to": {"type": "integer"},
    "archive": {"type": "string", "description": "part of the archive name, e.g. KBR, Neuchâtel"},
    "only_with_text": {"type": "boolean", "description": "only letters that have a transcription"},
}


def _tool(name: str, description: str, props: dict, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": required or []}}}


TOOLS = [
    _tool("search_letters", "Search letter fragments by content (any language). Returns numbered fragments to cite.",
          {"query": {"type": "string"}, "k": {"type": "integer", "description": "fragments, default 8"},
           "sender_id": _FILTERS["sender_id"], "year_from": _FILTERS["year_from"], "year_to": _FILTERS["year_to"]},
          ["query"]),
    _tool("find_person", "Resolve a name as written in the question to persons in the nodegoat person list, with "
          "their letter counts. Call this before using a person in other tools.", {"name": {"type": "string"}}, ["name"]),
    _tool("count_letters", "Exact number of letters matching the filters.", _FILTERS),
    _tool("letters_per_year", "Number of letters per year matching the filters (plus undated).", _FILTERS),
    _tool("top_people", "Ranking of persons by number of letters, as sender, recipient, correspondent (either) or "
          "mentioned (estimate; excludes each letter's own sender and recipient). Navez is excluded unless asked.",
          {"role": {"type": "string", "enum": ["sender", "recipient", "correspondent", "mentioned"]},
           "limit": {"type": "integer"}, "exclude_navez": {"type": "boolean"}, **_FILTERS}, ["role"]),
    _tool("top_entities", "Most frequent places, institutions, artworks or events in the letters (estimate).",
          {"kind": {"type": "string", "enum": ["places", "organisations", "artworks", "events"]},
           "limit": {"type": "integer"}, **_FILTERS}, ["kind"]),
    _tool("list_letters", "List the letters matching the filters (date, sender, recipient, archive).",
          {"limit": {"type": "integer"}, **_FILTERS}),
    _tool("collection_overview", "Total number of letters, how many are transcribed, and per archive.", {}),
]


def _run_tool(name: str, args: dict, searcher: Searcher, sources: list[dict], ui_filters: dict) -> dict:
    con = searcher.con
    if name == "search_letters":
        if sum(1 for r in sources if r.get("_first")) >= MAX_SEARCHES:
            return {"fragments": "Zoeklimiet bereikt: beantwoord de vraag met de fragmenten die je al hebt."}
        flt = dict(ui_filters)
        if args.get("sender_id"):
            flt["correspondent"] = stats._person_label(con, args["sender_id"])
        if args.get("year_from") or args.get("year_to"):
            flt["years"] = (int(args.get("year_from") or 1700), int(args.get("year_to") or 1900))
        found = searcher.search(args["query"], k=min(int(args.get("k") or 8), 10), **flt)
        if found:
            found[0]["_first"] = True   # marks the start of one search, for the search limit
        start = len(sources)
        sources.extend(found)
        return {"fragments": "\n\n".join(_context_block(start + i, r) for i, r in enumerate(found, 1))
                or "Geen fragmenten gevonden."}
    fn = {"find_person": stats.find_person, "count_letters": stats.count_letters,
          "letters_per_year": stats.letters_per_year, "top_people": stats.top_people,
          "top_entities": stats.top_entities, "list_letters": stats.list_letters,
          "collection_overview": stats.overview}[name]
    return fn(con, **args)


def stats_markdown(results: list[dict]) -> str:
    """Plain-markdown rendering of statistics results (for the Chainlit app and exports)."""
    out = []
    for s in results:
        r = s["result"]
        head = f"**{s['tool']}** — {', '.join(r.get('filters') or ['whole collection'])}"
        if s["tool"] == "count_letters":
            body = f"{r['count']} letters ({r['with_transcription']} transcribed)"
            if r.get("undated_not_counted"):
                body += f"; {r['undated_not_counted']} undated letters not counted"
        elif s["tool"] == "letters_per_year":
            body = ", ".join(f"{x['year']}: {x['letters']}" for x in r["per_year"]) + f"; undated: {r['undated']}"
        elif s["tool"] == "top_people":
            body = "; ".join(f"{p['name']} ({p['letters']})" for p in r["people"])
        elif s["tool"] == "top_entities":
            body = "; ".join(f"{v['value']} ({v['letters']})" for v in r["values"])
        elif s["tool"] == "list_letters":
            body = "; ".join(f"{x['letter_id']} ({x['date'] or 's.d.'})" for x in r["letters"])
            body += f" — {r['shown']} of {r['total']}"
        else:
            body = json.dumps(r, ensure_ascii=False)[:500]
        out.append(f"{head}\n{body}\n*Basis: {r.get('basis', '')}*")
    return "\n\n".join(out)


def answer_with_tools(query: str, searcher: Searcher, model: str = "anthropic/claude-sonnet-4.5",
                      history: list[dict] | None = None, k: int = 8, **ui_filters) -> dict:
    """Returns {answer, sources (for [#n]), stats [{tool, args, result}], cost}."""
    ui_filters = {key: v for key, v in ui_filters.items() if v is not None}
    navez = searcher.con.execute("SELECT person_id FROM persons WHERE name = ?", (stats.NAVEZ,)).fetchone()
    system = SYSTEM + (
        f"\n\nNavez zelf heeft person_id {navez[0]}. \"Brieven van X aan Navez\" = sender_id X EN "
        f"recipient_id {navez[0]}; \"brieven van Navez aan X\" = sender_id {navez[0]} EN recipient_id X. "
        "Laat die tweede filter nooit weg: niet alle brieven in de collectie zijn aan Navez gericht."
        if navez else "")
    messages = [{"role": "system", "content": system}] + (history or []) + [{"role": "user", "content": query}]
    sources, results, cost = [], [], 0.0
    for _ in range(MAX_ROUNDS):
        msg, c = chat_with_tools(messages, TOOLS, model=model)
        cost += c
        calls = msg.get("tool_calls") or []
        if not calls:
            text = msg.get("content") or ""
            cited = {int(n) for grp in re.findall(r"\[#([\d#,\s]+)\]", text) for n in re.findall(r"\d+", grp)}
            for i, r in enumerate(sources, 1):
                r["cited"] = i in cited
            return {"answer": text, "sources": sources, "stats": results, "cost": cost}
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for call in calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
                out = _run_tool(name, args, searcher, sources, ui_filters)
            except Exception as e:  # noqa: BLE001 - report tool errors back to the model
                args, out = {}, {"error": f"{type(e).__name__}: {e}"}
            if name not in ("search_letters", "find_person") and "error" not in out:
                results.append({"tool": name, "args": args, "result": out})
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": json.dumps(out, ensure_ascii=False, default=str)[:20000]})
    return {"answer": "Ik kon geen antwoord samenstellen binnen het maximale aantal stappen.",
            "sources": sources, "stats": results, "cost": cost}
