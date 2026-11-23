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
from . import memory, stats

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
8. Gebruik search_letters hoogstens twee keer per vraag. Gaat de vraag over één brief-ID, lees die brief met search_letters(letter_id=...) en haal de metadata op met list_letters(letter_id=...).
9. Begin meteen met het antwoord, zonder zinnen over je werkwijze ("Nu heb ik…").
10. Antwoord in de taal van de vraag (een Engelse vraag krijgt een Engels antwoord), helder en beknopt.
11. Corrigeert of vult de onderzoeker een FEIT aan over de brieven, personen of metadata (bv. "die brief \
is van Alexandre Robert, niet van Louis-Léopold"), roep dan remember_correction aan: de correctie als \
één zelfstandige, volledige zin (wie, wat, welke brief), met de betrokken brief-ID's en personen. \
Bevestig daarna kort dat je het onthouden hebt. Niet bij vragen, twijfels of meningen, en niet als de \
onderzoeker iets bevestigt wat al klopt. Is de correctie onduidelijk, vraag dan eerst wat bedoeld wordt.
12. Correcties van het onderzoeksteam gaan voor op metadata, automatische koppelingen en transcripties. \
Ze veranderen de statistiek-tabellen niet: zeg het als een correctie een getal zou beïnvloeden.
13. Noteerde de historicus twijfel of een opmerking bij een persoon in een fragment ("Twijfel of opmerking \
van de historicus"), vermeld die twijfel als je over die persoon spreekt, in plaats van de identificatie als \
zeker voor te stellen.
14. Vraagt de onderzoeker hoe een persoon geïdentificeerd werd, door wie, of of dat veranderde, gebruik dan \
identification_history voor die brief."""

MAX_SEARCHES = 2

_FILTERS = {
    "sender_id": {"type": "string", "description": "person_id of the sender (from find_person)"},
    "recipient_id": {"type": "string", "description": "person_id of the recipient"},
    "correspondent_id": {"type": "string", "description": "person_id as sender OR recipient"},
    "mentions_id": {"type": "string", "description": "person_id mentioned in the letter (estimate)"},
    "year_from": {"type": "integer"}, "year_to": {"type": "integer"},
    "archive": {"type": "string", "description": "part of the archive name, e.g. KBR, Neuchâtel"},
    "only_with_text": {"type": "boolean", "description": "only letters that have a transcription"},
    "letter_id": {"type": "string", "description": "one specific letter, e.g. Ms. II 70/1/184"},
}


def _tool(name: str, description: str, props: dict, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": required or []}}}


TOOLS = [
    _tool("search_letters", "Search letter fragments by content (any language). Returns numbered fragments to cite.",
          {"query": {"type": "string"}, "k": {"type": "integer", "description": "fragments, default 8"},
           "letter_id": {"type": "string", "description": "read this one letter (all its fragments) instead of searching"},
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
    _tool("identification_history", "How the persons in one letter were identified: the machine's suggestion, "
          "every reading by the historian (when, from which source) and her doubts.",
          {"letter_id": {"type": "string"}, "person": {"type": "string", "description": "optional: only this name"}},
          ["letter_id"]),
    _tool("remember_correction", "Store a correction of a fact, given by the researcher, so that it is used in all "
          "later conversations.",
          {"fact": {"type": "string", "description": "the corrected fact as one self-contained sentence"},
           "letter_ids": {"type": "array", "items": {"type": "string"},
                          "description": "letters concerned, e.g. Ms. II 70/1/181"},
           "persons": {"type": "array", "items": {"type": "string"}, "description": "persons concerned (full names)"}},
          ["fact"]),
]


def _fragment(i: int, r: dict) -> str:
    """A fragment for the model, with any correction the research team made about that letter."""
    notes = memory.notes_for_letter(r["letter_id"])
    return _context_block(i, r) + "".join(f"\nCorrectie onderzoeksteam: {n['fact']}" for n in notes)


def _run_tool(name: str, args: dict, searcher: Searcher, sources: list[dict], ui_filters: dict,
              ctx: dict | None = None) -> dict:
    con = searcher.con
    if name == "identification_history":
        rows = con.execute("SELECT provenance, persons_linked FROM chunks WHERE lower(letter_id) = lower(?) "
                           "ORDER BY chunk", (args["letter_id"].strip(),)).fetchall()             if "provenance" in searcher.columns else []
        seen, out = set(), []
        for prov, linked in rows:
            for p in json.loads(prov or "[]"):
                key = (p["surface"], p["person"], p["status"])
                want = (args.get("person") or "").lower()
                if key in seen or (want and want not in (p["surface"] + " " + p["person"] + " " + p["machine"]).lower()):
                    continue
                seen.add(key)
                out.append(p)
        if not rows:
            return {"error": "Deze brief heeft geen transcriptie in de chatbot."}
        if not out:
            auto = sorted({n for _, linked in rows for n in json.loads(linked or "[]")})
            return {"checked_by_historian": False,
                    "automatic_links": auto, "note": "Deze brief is nog niet nagekeken: alleen automatische koppelingen."}
        return {"checked_by_historian": True, "mentions": out[:40]}
    if name == "remember_correction":
        ctx = {} if ctx is None else ctx
        note = memory.add_note(args.get("fact", ""), args.get("letter_ids"), args.get("persons"),
                               author=ctx.get("author", ""), question=ctx.get("question", ""),
                               conversation_id=ctx.get("conversation_id", ""))
        ctx.setdefault("remembered", []).append(note)
        return {"saved": True, "id": note["id"]}
    if name == "search_letters":
        if sum(1 for r in sources if r.get("_first")) >= MAX_SEARCHES:
            return {"fragments": "Zoeklimiet bereikt: beantwoord de vraag met de fragmenten die je al hebt."}
        flt = dict(ui_filters)
        if args.get("sender_id"):
            flt["correspondent"] = stats._person_label(con, args["sender_id"])
        if args.get("year_from") or args.get("year_to"):
            flt["years"] = (int(args.get("year_from") or 1700), int(args.get("year_to") or 1900))
        if args.get("letter_id"):   # read one letter: all its fragments, in order
            rows = con.execute("SELECT rowid FROM chunks WHERE lower(letter_id) = lower(?) ORDER BY chunk",
                               (args["letter_id"].strip(),)).fetchall()
            found = [searcher._record(r[0]) for r in rows[:6]]
            if not found:
                return {"fragments": "Deze brief heeft geen transcriptie in de chatbot (gebruik list_letters "
                                     "met letter_id voor de metadata)."}
        else:
            found = searcher.search(args["query"], k=min(int(args.get("k") or 8), 10), **flt)
        if found:
            found[0]["_first"] = True   # marks the start of one search, for the search limit
        start = len(sources)
        sources.extend(found)
        return {"fragments": "\n\n".join(_fragment(start + i, r) for i, r in enumerate(found, 1))
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
                      history: list[dict] | None = None, k: int = 8, author: str = "",
                      conversation_id: str = "", **ui_filters) -> dict:
    """Returns {answer, sources (for [#n]), stats [{tool, args, result}], remembered [notes], cost}."""
    ui_filters = {key: v for key, v in ui_filters.items() if v is not None}
    navez = searcher.con.execute("SELECT person_id FROM persons WHERE name = ?", (stats.NAVEZ,)).fetchone()
    system = SYSTEM + (
        f"\n\nNavez zelf heeft person_id {navez[0]}. \"Brieven van X aan Navez\" = sender_id X EN "
        f"recipient_id {navez[0]}; \"brieven van Navez aan X\" = sender_id {navez[0]} EN recipient_id X. "
        "Laat die tweede filter nooit weg: niet alle brieven in de collectie zijn aan Navez gericht."
        if navez else "")
    recent = " ".join([query] + [h["content"] for h in (history or [])[-2:]])
    system += memory.notes_prompt(memory.relevant_notes(recent))
    ctx = {"author": author, "question": query, "conversation_id": conversation_id}
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
            return {"answer": text, "sources": sources, "stats": results, "cost": cost,
                    "remembered": ctx.get("remembered", [])}
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for call in calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
                out = _run_tool(name, args, searcher, sources, ui_filters, ctx)
            except Exception as e:  # noqa: BLE001 - report tool errors back to the model
                args, out = {}, {"error": f"{type(e).__name__}: {e}"}
            if name not in ("search_letters", "find_person", "remember_correction", "identification_history") and "error" not in out:
                results.append({"tool": name, "args": args, "result": out})
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": json.dumps(out, ensure_ascii=False, default=str)[:20000]})
    return {"answer": "Ik kon geen antwoord samenstellen binnen het maximale aantal stappen.",
            "sources": sources, "stats": results, "cost": cost, "remembered": ctx.get("remembered", [])}
