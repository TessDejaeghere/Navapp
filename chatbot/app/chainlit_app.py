"""Navapp chatbot: Chainlit UI over the Navez letters (after VaidyaBot).

Start:  chainlit run app/chainlit_app.py --port 8766      (or double-click Start_NavappBot.bat)
"""
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import chainlit as cl
import pandas as pd
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rag.agent import answer_with_tools, stats_markdown  # noqa: E402
from rag.answer import SOURCE_LABEL, export_rows  # noqa: E402
from rag.search import Searcher  # noqa: E402

EXPORT_DIR = ROOT / "data" / "exports"
HISTORY_DB = ROOT / "data" / "history.db"
ALL = "Alle correspondenten"
MODELS = {"Claude Sonnet 4.5": "anthropic/claude-sonnet-4.5", "Claude Opus 4.8": "anthropic/claude-opus-4.8"}

# Chainlit data-layer schema (same as VaidyaBot), so conversations are kept.
HISTORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS users ("id" TEXT PRIMARY KEY, "identifier" TEXT NOT NULL UNIQUE, "metadata" TEXT NOT NULL, "createdAt" TEXT);
CREATE TABLE IF NOT EXISTS threads ("id" TEXT PRIMARY KEY, "createdAt" TEXT, "name" TEXT, "userId" TEXT, "userIdentifier" TEXT,
    "tags" TEXT, "metadata" TEXT, FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS steps ("id" TEXT PRIMARY KEY, "name" TEXT NOT NULL, "type" TEXT NOT NULL, "threadId" TEXT NOT NULL,
    "parentId" TEXT, "streaming" INTEGER NOT NULL, "waitForAnswer" INTEGER, "isError" INTEGER, "metadata" TEXT, "tags" TEXT,
    "input" TEXT, "output" TEXT, "createdAt" TEXT, "command" TEXT, "start" TEXT, "end" TEXT, "generation" TEXT,
    "showInput" TEXT, "language" TEXT, "indent" INTEGER, "defaultOpen" INTEGER, "autoCollapse" INTEGER,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS elements ("id" TEXT PRIMARY KEY, "threadId" TEXT, "type" TEXT, "url" TEXT, "chainlitKey" TEXT,
    "name" TEXT NOT NULL, "display" TEXT, "objectKey" TEXT, "size" TEXT, "page" INTEGER, "language" TEXT, "forId" TEXT,
    "mime" TEXT, "props" TEXT, FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS feedbacks ("id" TEXT PRIMARY KEY, "forId" TEXT NOT NULL, "threadId" TEXT NOT NULL,
    "value" INTEGER NOT NULL, "comment" TEXT, FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE);
"""


@cl.data_layer
def _data_layer() -> SQLAlchemyDataLayer:
    HISTORY_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(HISTORY_DB) as con:
        con.executescript(HISTORY_SCHEMA)
    return SQLAlchemyDataLayer(conninfo=f"sqlite+aiosqlite:///{HISTORY_DB.as_posix()}")


@cl.header_auth_callback
def _auto_login(headers) -> cl.User | None:
    # Local single-user app: log in silently so that conversation history works.
    return cl.User(identifier="onderzoeker")


_searcher: Searcher | None = None


def get_searcher() -> Searcher:
    global _searcher
    if _searcher is None:
        _searcher = Searcher()
    return _searcher


async def _send_settings() -> None:
    s = get_searcher()
    lo, hi = s.year_range()
    settings = await cl.ChatSettings([
        cl.input_widget.Select(id="correspondent", label="Correspondent", values=[ALL] + s.correspondents(), initial_index=0),
        cl.input_widget.TextInput(id="person", label="Alleen brieven die deze persoon vermelden", initial="",
                                  placeholder="bv. Schnetz"),
        cl.input_widget.Slider(id="year_from", label="Vanaf jaar", initial=lo, min=lo, max=hi, step=1),
        cl.input_widget.Slider(id="year_to", label="Tot en met jaar", initial=hi, min=lo, max=hi, step=1),
        cl.input_widget.Switch(id="undated", label="Ongedateerde brieven meenemen", initial=True),
        cl.input_widget.Select(id="model", label="Taalmodel", values=list(MODELS), initial_index=0),
        cl.input_widget.Slider(id="k", label="Aantal bronfragmenten", initial=8, min=3, max=15, step=1),
    ]).send()
    cl.user_session.set("settings", settings)


@cl.on_chat_start
async def start() -> None:
    await _send_settings()
    await cl.Message(
        "Welkom bij de **Navapp-chatbot**. Stel een vraag over de briefwisseling van "
        "**François-Joseph Navez** (KBR, Ms. II 70), in het Nederlands, Engels of Frans.\n\n"
        "Onder elk antwoord staan de brieffragmenten waarop het steunt. Via ⚙️ kunt u filteren op "
        "correspondent, vermelde persoon en periode. Typ **exporteer** om de laatste bronnen als Excel op te slaan.\n\n"
        "*Let op: een deel van de transcripties is ongecorrigeerde HTR en de gekoppelde personen zijn "
        "automatische suggesties.*").send()


@cl.on_chat_resume
async def on_resume(thread) -> None:
    await _send_settings()


@cl.on_settings_update
async def on_settings(settings: dict) -> None:
    cl.user_session.set("settings", settings)


def _filters(settings: dict) -> dict:
    lo, hi = get_searcher().year_range()
    y0, y1 = int(settings.get("year_from", lo)), int(settings.get("year_to", hi))
    undated = bool(settings.get("undated", True))
    years = None if (y0, y1) == (lo, hi) and undated else (min(y0, y1), max(y0, y1))
    corr = settings.get("correspondent", ALL)
    return {"correspondent": None if corr == ALL else corr,
            "person": (settings.get("person") or "").strip() or None,
            "years": years, "include_undated": undated}


def _source_text(i: int, r: dict) -> str:
    lines = [f"### #{i} — {r['letter_id']}",
             f"**{r['sender'] or 'onbekend'}** aan **{r['recipient'] or 'onbekend'}** · {r['date'] or 'datum onbekend'}",
             f"*Transcriptie: {SOURCE_LABEL.get(r['text_source'], r['text_source'])} · relevantie {r.get('confidence', 0):.0%}*",
             "", r["text"], ""]
    if r.get("persons_checked"):
        lines.append("**Personen (nagekeken):** " + ", ".join(r["persons_checked"]))
    if r["persons_linked"]:
        lines.append("**Gekoppelde personen (automatische suggestie):** " + ", ".join(r["persons_linked"]))
    for label, f in (("Plaatsen", "places"), ("Instellingen", "organisations"), ("Kunstwerken", "artworks"), ("Gebeurtenissen", "events")):
        if r[f]:
            lines.append(f"**{label}:** " + ", ".join(r[f]))
    if r["permalink"]:
        lines.append(f"\n[📜 Bekijk de brief in de KBR-catalogus]({r['permalink']})")
    return "\n".join(lines)


async def do_export() -> None:
    sources = cl.user_session.get("last_sources")
    if not sources:
        await cl.Message("Er is nog geen zoekresultaat om te exporteren.").send()
        return
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPORT_DIR / f"navez_bronnen_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
    pd.DataFrame(export_rows(sources)).to_excel(path, index=False)
    await cl.Message(f"Geëxporteerd naar {path.name}", elements=[cl.File(name=path.name, path=str(path))]).send()


@cl.action_callback("export_excel")
async def on_export(action: cl.Action) -> None:
    await do_export()


@cl.action_callback("open_source")
async def on_open_source(action: cl.Action) -> None:
    sources = cl.user_session.get("last_sources") or []
    i = action.payload["i"]
    if 1 <= i <= len(sources):
        r = sources[i - 1]
        el = cl.Text(name=f"bron-{i}", content=_source_text(i, r), display="side")
        await cl.ElementSidebar.set_elements([el], key=f"bron-{r['id']}")
        await cl.ElementSidebar.set_title(f"Bron #{i} — {r['letter_id']}")


@cl.on_message
async def main(message: cl.Message) -> None:
    query = message.content.strip()
    if query.lower() in ("exporteer", "export", "exporteren"):
        await do_export()
        return
    settings = cl.user_session.get("settings") or {}
    model_label = settings.get("model", "Claude Sonnet 4.5")
    msg = cl.Message("Zoeken in de brieven…")
    await msg.send()
    try:
        result = await cl.make_async(answer_with_tools)(query, get_searcher(),
                                                        model=MODELS.get(model_label, MODELS["Claude Sonnet 4.5"]),
                                                        **_filters(settings))
    except RuntimeError as e:
        msg.content = f"⚠️ Er ging iets mis bij het ophalen van het antwoord: {e}"
        await msg.update()
        return
    sources = result["sources"]
    cl.user_session.set("last_sources", sources)
    actions = []
    for i, r in enumerate(sources, 1):
        who = re.sub(r"\s*\(.*?\)", "", r["sender"] or "?")
        actions.append(cl.Action(name="open_source", payload={"i": i},
                                 label=f"[#{i}] {'✓' if r.get('cited') else '·'} {who}, {r['date'] or 's.d.'}"))
    if sources:
        actions.append(cl.Action(name="export_excel", payload={}, label="⬇ Exporteer naar Excel"))
    footer = "\n\n---\n**Bronnen** — klik om het fragment te lezen (✓ = geciteerd):" if sources else ""
    data = ("\n\n---\n**📊 Gegevens achter dit antwoord**\n\n" + stats_markdown(result["stats"])) if result["stats"] else ""
    msg.content = result["answer"] + data + footer
    msg.actions = actions
    await msg.update()
