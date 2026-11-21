"""Navapp — Chatbot page: questions about the Navez letters.

Content questions are answered from letter fragments (cited as [#n]); questions about counts,
rankings and dates are answered from Fien's nodegoat metadata, with the numbers shown as tables.
"""
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "chatbot"))
from rag import memory  # noqa: E402
from rag.agent import answer_with_tools  # noqa: E402
from rag.answer import SOURCE_LABEL, export_rows  # noqa: E402
from rag.indexer import current_dir  # noqa: E402
from rag.search import Searcher  # noqa: E402

ASSETS = REPO / "assets"
AVATAR = str(ASSETS / "navapp_icon.png")
MODELS = {"Claude Sonnet 4.5": "anthropic/claude-sonnet-4.5", "Claude Opus 4.8": "anthropic/claude-opus-4.8"}
ALL = "All correspondents"
TOOL_TITLES = {"count_letters": "Count", "letters_per_year": "Letters per year", "top_people": "Ranking of persons",
               "top_entities": "Most frequent", "list_letters": "Letters", "collection_overview": "Collection"}


@st.cache_resource(show_spinner="Loading the chatbot index …", max_entries=1)
def get_searcher(version: str) -> Searcher:
    """One searcher per index version: after 'send to chatbot' the new version loads by itself."""
    return Searcher()


def source_markdown(r: dict) -> str:
    lines = [f"**{r['sender'] or 'unknown'}** to **{r['recipient'] or 'unknown'}** · {r['date'] or 'undated'}  ",
             f"*{r['letter_id']} · transcription: {SOURCE_LABEL.get(r['text_source'], r['text_source'])} · "
             f"relevance {r.get('confidence', 0):.0%}*", "", r["text"].replace("\n", "  \n"), ""]
    if r.get("persons_checked"):
        lines.append("**Persons (checked by the research team):** " + ", ".join(r["persons_checked"]))
    if r["persons_linked"]:
        lines.append("**Linked persons (automatic suggestion):** " + ", ".join(r["persons_linked"]))
    for label, f in (("Places", "places"), ("Institutions", "organisations"), ("Artworks", "artworks"),
                     ("Events", "events")):
        if r[f]:
            lines.append(f"**{label}:** " + ", ".join(r[f]))
    if r["permalink"]:
        lines.append(f"\n[📜 Open the letter in the KBR catalogue]({r['permalink']})")
    return "\n".join(lines)


def show_sources(sources: list[dict]) -> None:
    for i, r in enumerate(sources, 1):
        if not r.get("cited"):
            continue
        who = re.sub(r"\s*\(.*?\)", "", r["sender"] or "?")
        with st.expander(f"[#{i}] {who}, {r['date'] or 's.d.'} — {r['letter_id']}"):
            st.markdown(source_markdown(r))
    others = sum(1 for r in sources if not r.get("cited"))
    if others:
        st.caption(f"{others} more fragment(s) were read but not cited.")


def show_remembered(notes: list[dict]) -> None:
    for n in notes:
        st.info(f"🧠 **Remembered for future conversations:** {n['fact']}\n\n"
                "You can edit or delete it on the *Chatbot data* page.")


def show_stats(results: list[dict], key: str) -> None:
    """The data behind every number in the answer."""
    for j, s in enumerate(results):
        r = s["result"]
        filters = ", ".join(r.get("filters") or ["whole collection"])
        with st.expander(f"📊 {TOOL_TITLES.get(s['tool'], s['tool'])} — {filters}", expanded=False):
            if s["tool"] == "count_letters":
                c1, c2 = st.columns(2)
                c1.metric("Letters", f"{r['count']:,}")
                c2.metric("Of which transcribed", f"{r['with_transcription']:,}")
                if r.get("undated_not_counted"):
                    st.caption(f"⚠️ {r['undated_not_counted']:,} matching letters have no date and are not counted.")
            elif s["tool"] == "letters_per_year" and r["per_year"]:
                df = pd.DataFrame(r["per_year"]).set_index("year")
                st.bar_chart(df, y="letters")
                st.caption(f"Undated (not in the chart): {r['undated']:,}")
            elif s["tool"] == "top_people":
                st.dataframe(pd.DataFrame(r["people"])[["name", "letters"]], hide_index=True, width="stretch")
            elif s["tool"] == "top_entities":
                st.dataframe(pd.DataFrame(r["values"]), hide_index=True, width="stretch")
            elif s["tool"] == "list_letters" and r["letters"]:
                st.dataframe(pd.DataFrame(r["letters"]), hide_index=True, width="stretch",
                             column_config={"kbr_link": st.column_config.LinkColumn("KBR", display_text="open")})
                if r["total"] > r["shown"]:
                    st.caption(f"Showing {r['shown']} of {r['total']:,} letters.")
            elif s["tool"] == "collection_overview":
                st.json(r)
            st.caption(f"Basis: {r.get('basis', '')}")


searcher = get_searcher(current_dir().name)
lo, hi = searcher.year_range()

with st.sidebar:
    st.header("Search in")
    corr = st.selectbox("Correspondent", [ALL] + searcher.correspondents())
    person = st.text_input("Only letters mentioning", placeholder="e.g. Schnetz")
    years = st.slider("Period", lo, hi, (lo, hi))
    undated = st.toggle("Include undated letters", value=True)
    st.caption("These filters apply to the text search. Counting questions use the whole collection "
               "unless your question says otherwise.")
    st.header("Answer")
    model = st.selectbox("Language model", list(MODELS))
    st.caption(f"Index {searcher.version} · ≈ $0.02–0.10 per answer with Sonnet")
    if st.button("➕ New conversation"):
        st.session_state.chat = []
        st.session_state.conv_id = memory.new_conversation_id()
        st.rerun()
    st.header("Conversations")
    author = st.text_input("Your name", key="author", placeholder="e.g. Fien",
                           help="Stored with the corrections the chatbot remembers from you.")
    past = memory.list_conversations()
    if past:
        labels = {c["id"]: f"{c['updated'][:16].replace('T', ' ')} · {c['title'][:45]}" for c in past}
        pick = st.selectbox("Reopen a conversation", [""] + list(labels), format_func=lambda i: labels.get(i, "—"))
        if pick and pick != st.session_state.get("conv_id"):
            st.session_state.chat = memory.load_conversation(pick)
            st.session_state.conv_id = pick
            st.rerun()
    st.caption("Conversations are saved automatically. When you correct the chatbot on a fact, it remembers "
               "the correction in all later conversations.")

chat = st.session_state.setdefault("chat", [])
conv_id = st.session_state.setdefault("conv_id", memory.new_conversation_id())
if not chat:   # empty conversation: Navez himself welcomes you
    _, mid, _ = st.columns([2, 1, 2])
    mid.image(str(ASSETS / "navapp_logo.png"), width="stretch")
    st.markdown("<p style='text-align:center'><i>Ask Navez — or rather, his letters.</i></p>",
                unsafe_allow_html=True)
for n, turn in enumerate(chat):
    with st.chat_message(turn["role"], avatar=AVATAR if turn["role"] == "assistant" else None):
        st.markdown(turn["content"])
        if turn.get("remembered"):
            show_remembered(turn["remembered"])
        if turn.get("stats"):
            show_stats(turn["stats"], key=f"h{n}")
        if turn.get("sources"):
            show_sources(turn["sources"])

question = st.chat_input("Ask a question about the letters …")
if question:
    history = [{"role": t["role"], "content": t["content"]} for t in chat[-6:]]
    chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    filters = {"correspondent": None if corr == ALL else corr, "person": person.strip() or None,
               "years": None if years == (lo, hi) and undated else years, "include_undated": undated}
    with st.chat_message("assistant", avatar=AVATAR):
        with st.spinner("Searching and counting …"):
            try:
                res = answer_with_tools(question, searcher, model=MODELS[model], history=history,
                                        author=author.strip(), conversation_id=conv_id, **filters)
            except RuntimeError as e:
                res = {"answer": f"⚠️ No answer: {e}. Check the OpenRouter key on the Pipeline page (Settings).",
                       "sources": [], "stats": []}
        st.markdown(res["answer"])
        show_remembered(res.get("remembered", []))
        show_stats(res["stats"], key="new")
        show_sources(res["sources"])
    chat.append({"role": "assistant", "content": res["answer"], "sources": res["sources"], "stats": res["stats"],
                 "remembered": res.get("remembered", [])})
    memory.save_conversation(conv_id, chat)

last = next((t["sources"] for t in reversed(chat) if t.get("sources")), None)
if last:
    st.download_button("⬇️ Download the last sources (CSV)", pd.DataFrame(export_rows(last)).to_csv(index=False)
                       .encode("utf-8-sig"), file_name="navez_sources.csv", mime="text/csv")
