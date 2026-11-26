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
c1, c2, c3, c4 = st.columns(4)
c1.metric("Letters in the chatbot", f"{len({c['letter_id'] for c in chunks}):,}")
c2.metric("Sent from Navapp", f"{len(uploads):,}")
c3.metric("With checked persons", f"{len({c['letter_id'] for c in chunks if 'persons_checked' in c}):,}")
c4.metric("Index version", indexer.current_dir().name)

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


@st.cache_data(show_spinner=False)
def _transcriptions(which: str, version: str, n_uploads: int) -> tuple[bytes, int]:
    """ZIP of full transcriptions; cached per index version and number of uploads."""
    source = ingest.load_uploads() if which == "sent" else indexer.all_chunks()[0]
    letters = ingest.letter_texts(source)
    return ingest.transcriptions_zip(letters), len(letters)


st.subheader("Export transcriptions")
st.caption("Download the full text of the letters as they are in the chatbot: one .txt file per letter, "
           "plus a CSV with sender, recipient, date, the kind of transcription and the text.")
which = st.radio("Which letters", ["sent", "all"], horizontal=True, key="export_which",
                 format_func=lambda w: (f"Letters I sent ({len(uploads):,})" if w == "sent"
                                        else "All letters in the chatbot"))
if which == "sent" and not uploads:
    st.info("You have not sent any letters yet.")
else:
    with st.spinner("Preparing the transcriptions …"):
        data, n_letters = _transcriptions(which, indexer.current_dir().name, len(uploads))
    st.download_button(f"⬇️ Download transcriptions ({n_letters:,} letters, ZIP)", data,
                       file_name=f"navapp_transcriptions_{which}.zip", mime="application/zip",
                       key="export_transcriptions")

st.subheader("Checked persons (OpenRefine export)")
from rag import corrections  # noqa: E402

st.caption("Upload the mention CSV you cleaned in OpenRefine (with **surface_v2**, ID_Nodegoat, Notes, Source_v2, "
           "Target_v2). For the letters in that file, your persons replace the automatic links in the chatbot: "
           "a nodegoat name or ID links the mention to that person, another name in surface_v2 is kept as a name, "
           "DEL / Geen PER removes it, GRP and ALG keep it without a person, ART and LOC move it to artworks and "
           "places. Your corrected sender and recipient are used as well. A newer export of the same letters "
           "replaces the older one; letters from other batches are kept.")
done = corrections.imports()
if done:
    st.dataframe(pd.DataFrame(done)[["file", "imported", "letters", "mentions", "linked", "named", "deleted"]],
                 width="stretch", hide_index=True)
corr_up = st.file_uploader("OpenRefine export (CSV)", type=["csv"], key="openrefine_export")
if corr_up is not None:
    try:
        records, summary = corrections.normalise(corrections.read_export(corr_up.getvalue()))
    except ValueError as e:
        st.error(str(e))
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Letters", f"{summary['letters']:,}")
        c2.metric("Linked to a nodegoat person", f"{summary.get('linked', 0):,}",
                  help=f"{summary['persons']:,} different persons")
        c3.metric("Named, without nodegoat ID", f"{summary.get('named', 0):,}")
        c4.metric("Removed (DEL / not a person)", f"{summary.get('deleted', 0):,}")
        st.caption(f"Also: {summary.get('group', 0)} groups, {summary.get('generic', 0)} generic, "
                   f"{summary.get('artwork', 0)} artworks, {summary.get('place', 0)} places, "
                   f"{summary.get('open', 0):,} mentions left as they were (kept as a name in the text, "
                   "without a person).")
        if st.button(f"✅ Use these corrections for {summary['letters']:,} letters", type="primary"):
            corrections.save(records, summary, corr_up.name)
            bar = st.progress(0.0, "Updating the index …")
            try:
                s3 = indexer.rebuild_index(progress=lambda f, m: bar.progress(min(max(f, 0.0), 1.0), m))
                st.session_state.corr_msg = (f"Corrections saved; index {s3['version']} rebuilt "
                                             f"({s3['newly_embedded']} fragments re-embedded, ${s3['cost_usd']:.3f}).")
            except RuntimeError as e:
                st.session_state.corr_msg = f"Corrections saved, but the index could not be rebuilt: {e}"
            st.rerun()
if msg := st.session_state.pop("corr_msg", None):
    st.success(msg)
if done and st.button("Remove all corrections (back to the automatic links)"):
    corrections.clear()
    with st.spinner("Updating the index …"):
        indexer.rebuild_index()
    st.rerun()

st.subheader("History of the readings")
st.caption("Every checked mention keeps its history: what the machine suggested, and every reading by the historian "
           "that differed from the one before (when, and from which export). Her doubts are kept with them.")
rows = []
for r in corrections.load():
    hist = r.get("history") or []
    first = next((h for h in hist if h["by"] == "machine"), {})
    last = hist[-1] if hist else {}
    rows.append({"letter": r["letter_id"], "mention": r["surface"],
                 "machine": first.get("person") or "—", "historian": r.get("person") or r["status"],
                 "changed": (first.get("person") or "") != (r.get("person") or ""), "doubt": r.get("note", ""),
                 "readings": len(hist), "since": (last.get("when") or "")[:10], "source": last.get("source", "")})
if rows:
    hist_df = pd.DataFrame(rows)
    c1, c2, c3 = st.columns(3)
    c1.metric("Checked mentions", f"{len(hist_df):,}")
    c2.metric("Reading changed by the historian", f"{int(hist_df['changed'].sum()):,}")
    c3.metric("With a doubt or remark", f"{int((hist_df['doubt'] != '').sum()):,}")
    view = st.radio("Show", ["Most frequent changes", "Doubts and remarks", "Search"], horizontal=True)
    if view == "Most frequent changes":
        top = (hist_df[hist_df["changed"]].groupby(["machine", "historian"]).size()
               .reset_index(name="mentions").sort_values("mentions", ascending=False).head(25))
        st.dataframe(top, hide_index=True, width="stretch")
    elif view == "Doubts and remarks":
        st.dataframe(hist_df[hist_df["doubt"] != ""][["letter", "mention", "historian", "doubt"]],
                     hide_index=True, width="stretch")
    else:
        q = st.text_input("Name, mention or letter", placeholder="e.g. Robert, Granet, Ms. II 70/1/181")
        if q:
            hit = hist_df[hist_df[["letter", "mention", "machine", "historian"]].apply(
                lambda col: col.str.contains(q, case=False, regex=False)).any(axis=1)]
            st.dataframe(hit, hide_index=True, width="stretch")
    st.download_button("⬇️ Download all readings (CSV)", hist_df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="navapp_readings.csv", mime="text/csv")
else:
    st.info("No checked mentions yet: upload an OpenRefine export above.")

st.subheader("What the chatbot remembers")
from rag import memory  # noqa: E402

notes = memory.load_notes()
st.caption("Facts you corrected in the chat. Every later answer uses them, and they take precedence over the "
           "metadata, the automatic links and the transcriptions. They do not change the counts: correct "
           "those in nodegoat and upload a new export below.")
if not notes:
    st.info("No corrections yet. Correct the chatbot in the chat (e.g. \"That letter was written by Alexandre "
            "Robert, not Louis-Léopold\") or add one here.")
for n in sorted(notes, key=lambda x: x["created"], reverse=True):
    if n.get("shared"):   # published by Fien: read-only here, she corrects it at the source
        with st.expander(f"📢 {n['created'][:10]} · {n['author']} · {n['fact'][:90]}"):
            st.markdown(n["fact"])
            if n["letter_ids"]:
                st.caption("Letters: " + ", ".join(n["letter_ids"]))
            for h in n.get("history") or []:
                st.caption(f"Earlier ({h['when'][:10]}): ~~{h['fact']}~~")
            st.caption(f"Published by {n.get('published_by') or n['author']}: read-only here.")
        continue
    with st.expander(f"{n['created'][:10]} · {n['author']} · {n['fact'][:90]}"):
        fact = st.text_area("Correction", n["fact"], key=f"fact_{n['id']}")
        letters = st.text_input("Letters (comma-separated)", ", ".join(n["letter_ids"]), key=f"let_{n['id']}")
        if n.get("question"):
            st.caption(f"Saved while answering: *{n['question']}*")
        for h in n.get("history") or []:
            st.caption(f"Earlier version ({h['when'][:10]}): ~~{h['fact']}~~")
        b1, b2 = st.columns(2)
        if b1.button("💾 Save", key=f"save_{n['id']}"):
            memory.update_note(n["id"], fact=" ".join(fact.split()),
                               letter_ids=[x.strip() for x in letters.split(",") if x.strip()])
            st.rerun()
        if b2.button("🗑 Delete", key=f"del_{n['id']}"):
            memory.delete_note(n["id"])
            st.rerun()
with st.form("add_note", clear_on_submit=True):
    st.markdown("**Add a correction**")
    fact = st.text_area("Correction (one full sentence)", placeholder="Ms. II 70/1/184 was written by Alexandre "
                        "Robert, not by Jean François Portaels.")
    letters = st.text_input("Letters (comma-separated, optional)")
    who = st.text_input("Your name", value=st.session_state.get("author", ""))
    if st.form_submit_button("➕ Remember"):
        try:
            memory.add_note(fact, [x for x in letters.split(",")], author=who.strip())
            st.rerun()
        except ValueError as e:
            st.error(str(e))

st.subheader("Fien's published corrections")
from rag import sharing  # noqa: E402

st.caption("Fien is the source for all corrections. When she publishes hers, every Navapp can use them: the facts "
           "she corrected in the chat and the persons she checked in OpenRefine. Your own corrections stay on "
           "this computer; for letters you corrected yourself, yours are used.")
have, remote = sharing.local_manifest(), sharing.check()
if have:
    st.markdown(f"In use: published by **{have['by']}** on {have['published'][:16].replace('T', ' ')} "
                f"({have['notes']} chat corrections, {have['person_letters']:,} letters with checked persons).")
if remote.get("error"):
    st.caption(f"Could not look for published corrections ({remote['error']}).")
elif remote.get("new"):
    st.info(f"📢 **New corrections from {remote['by']}**, published {remote['published'][:16].replace('T', ' ')}: "
            f"{remote['notes']} chat corrections, {remote['person_letters']:,} letters with checked persons.")
    if st.button("⬇️ Use these corrections", type="primary"):
        bar = st.progress(0.0, "Downloading …")
        try:
            sharing.fetch()
            st.cache_data.clear()   # drop the sidebar's 'new corrections' notice
            s4 = indexer.rebuild_index(progress=lambda f, m: bar.progress(min(max(f, 0.0), 1.0), m))
            st.session_state.shared_msg = (f"Fien's corrections are in use (index {s4['version']}, "
                                           f"{s4['newly_embedded']} fragments re-embedded, ${s4['cost_usd']:.3f}).")
        except Exception as e:  # noqa: BLE001
            st.session_state.shared_msg = f"⚠️ Could not load the corrections: {type(e).__name__}: {e}"
        st.rerun()
elif have:
    st.caption("Up to date.")
if msg := st.session_state.pop("shared_msg", None):
    st.success(msg)

with st.expander("⬆️ Publish my corrections (for Fien)"):
    st.markdown(f"Publishes your chat corrections and your OpenRefine corrections to the public GitHub repo "
                f"[{sharing.DATA_REPO}](https://github.com/{sharing.DATA_REPO}), so that other Navapps can use them. "
                "Conversations and letters are never published. **Everything you publish is public.**")
    if not sharing.token():
        st.caption("Needs a GitHub key once: on github.com → Settings → Developer settings → Fine-grained tokens, "
                   f"create a token for the repository {sharing.DATA_REPO} only, with *Contents: Read and write*.")
        tok = st.text_input("GitHub key", type="password", key="gh_token")
        if st.button("Save key") and tok.strip():
            sharing.save_token(tok)
            st.rerun()
    else:
        mine_notes = len(memory.load_notes(include_shared=False))
        mine_letters = len({r["letter_key"] for r in corrections.load(include_shared=False)})
        by = st.text_input("Published by", value=st.session_state.get("author", "") or "Fien")
        confirm = st.checkbox(f"Publish {mine_notes} chat corrections and the checked persons of "
                              f"{mine_letters:,} letters, publicly, replacing what was published before.")
        c1, c2 = st.columns(2)
        if c1.button("⬆️ Publish", type="primary", disabled=not confirm):
            with st.spinner("Publishing …"):
                try:
                    m = sharing.publish(by.strip())
                    st.cache_data.clear()
                    st.success(f"Published on {m['published'][:16].replace('T', ' ')}. Other Navapps will see a notice.")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Publishing failed: {type(e).__name__}: {e}")
        if c2.button("Remove the GitHub key from this computer"):
            sharing.TOKEN_FILE.unlink(missing_ok=True)
            st.rerun()

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
