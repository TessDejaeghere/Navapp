"""Navapp — one app for the Navez letters: the pipeline (HTR → entities → clusters → links)
and the chatbot that answers questions from the letters.

Run:  streamlit run navapp.py        (or double-click start_navapp.bat)
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import navapp_update  # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets"

st.set_page_config(page_title="Navapp", page_icon=str(ASSETS / "navapp_icon.png"), layout="wide")
st.logo(str(ASSETS / "navapp_mark.png"), size="large", icon_image=str(ASSETS / "navapp_frame.png"))


@st.cache_data(ttl=3600, show_spinner=False)
def update_status() -> dict:
    """Asks GitHub at most once an hour whether a newer Navapp exists."""
    return navapp_update.check()


def update_box() -> None:
    u = update_status()
    with st.sidebar:
        if u.get("available"):
            st.info(f"🆕 **A new Navapp is available** ({u['latest']}; you have {u['current']}).\n\n{u['notes']}")
            if st.button("⬇️ Install the update", type="primary"):
                with st.status("Updating Navapp …", expanded=True) as box:
                    try:
                        res = navapp_update.install(progress=box.write)
                    except Exception as e:  # noqa: BLE001 - show any download/install problem
                        box.update(label="The update failed", state="error")
                        st.error(f"Nothing was lost; Navapp keeps working as before. ({type(e).__name__}: {e})")
                        return
                    box.update(label=f"Updated to {res['to']}", state="complete")
                update_status.clear()
                if res["packages"].startswith("failed"):
                    st.warning(f"New Python packages could not be installed: {res['packages']}")
                st.success("Done! Close Navapp (the black window) and start it again with start_navapp.bat. "
                           "Your letters, corrections and conversations are untouched.")
                st.stop()   # do not run old and new code side by side
            if not st.session_state.get("update_toast"):
                st.session_state.update_toast = True
                st.toast(f"A new Navapp ({u['latest']}) is available: see the sidebar.", icon="🆕")
        cap = f"Navapp {u['current']}"
        if u.get("error"):
            cap += " · could not check for updates"
        elif not u.get("available"):
            cap += " · up to date"
        st.caption(cap)


update_box()

page = st.navigation([
    st.Page("pipeline_app.py", title="Pipeline", icon="🛠️", default=True),
    st.Page("navapp_pages/chat.py", title="Chatbot", icon="💬"),
    st.Page("navapp_pages/chat_data.py", title="Chatbot data", icon="🗂️"),
])
page.run()
