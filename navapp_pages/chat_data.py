"""Navapp — Chatbot data page: which letters the chatbot knows, and removing uploads."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "chatbot"))
from rag import indexer, ingest  # noqa: E402

st.title("Chatbot data")
st.caption("The chatbot searches the research corpus plus the letters you sent from the Pipeline page "
           "(Step 4). A letter you sent replaces the research-corpus version with the same ID.")

chunks, _ = indexer.all_chunks()
uploads = ingest.uploaded_letters()
c1, c2, c3 = st.columns(3)
c1.metric("Letters in the chatbot", f"{len({c['letter_id'] for c in chunks}):,}")
c2.metric("Sent from Navapp", f"{len(uploads):,}")
c3.metric("Index version", indexer.current_dir().name)

st.subheader("Letters you sent")
if not uploads:
    st.info("None yet. Process letters on the Pipeline page and use Step 4 to send them here.")
else:
    st.dataframe(pd.DataFrame(uploads), width="stretch", hide_index=True)
    drop = st.multiselect("Remove letters from the chatbot", [u["letter_id"] for u in uploads])
    if st.button(f"🗑 Remove {len(drop)} letter(s)", disabled=not drop):
        n = ingest.remove_letters(drop)
        with st.spinner("Updating the index …"):
            stats = indexer.rebuild_index()
        st.session_state.removed_msg = (f"Removed {n} letter(s); index {stats['version']} now has "
                                        f"{stats['letters']:,} letters.")
        st.rerun()
if msg := st.session_state.pop("removed_msg", None):
    st.success(msg)

st.subheader("Metadata for counting questions")
from rag import stats  # noqa: E402

current = next((p for p in stats.LETTER_EXPORTS if p.exists()), None)
st.caption("Questions like 'how many letters did Beauvoir write?' are answered from your nodegoat letters "
           f"export (all archives). In use: **{current.name if current else 'none'}**. After you update "
           "nodegoat, upload a fresh export of the letters here.")
up = st.file_uploader("nodegoat letters export (CSV)", type=["csv"], key="letters_export")
if up is not None and st.button("Use this export"):
    head = up.getvalue()[:4000].decode("utf-8-sig", errors="replace").splitlines()[0]
    needed = ["Name", "Source", "Target"]
    if not all(n in head for n in needed) or not ("Date" in head or "catalogue_date" in head):
        st.error("This does not look like a nodegoat letters export (needs Name, Source, Target and a date column).")
    else:
        stats.LETTER_EXPORTS[0].write_bytes(up.getvalue())
        with st.spinner("Updating the index …"):
            s2 = indexer.rebuild_index()
        st.success(f"Export saved; index {s2['version']} rebuilt.")

st.subheader("Maintenance")
st.caption("Rebuilding is only needed after the research corpus itself changed (scripts/build_corpus.py). "
           "Fragments that were embedded before are reused, so it costs nothing unless texts changed.")
if st.button("🔄 Rebuild the index"):
    bar = st.progress(0.0, "Rebuilding …")
    try:
        stats = indexer.rebuild_index(progress=lambda f, m: bar.progress(min(max(f, 0.0), 1.0), m))
        st.success(f"Index {stats['version']}: {stats['letters']:,} letters, {stats['chunks']:,} fragments, "
                   f"{stats['newly_embedded']} newly embedded (${stats['cost_usd']:.4f}).")
    except RuntimeError as e:
        st.error(f"Rebuild failed: {e}")
