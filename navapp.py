"""Navapp — one app for the Navez letters: the pipeline (HTR → entities → clusters → links)
and the chatbot that answers questions from the letters.

Run:  streamlit run navapp.py        (or double-click start_navapp.bat)
"""
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets"

st.set_page_config(page_title="Navapp", page_icon=str(ASSETS / "navapp_icon.png"), layout="wide")
st.logo(str(ASSETS / "navapp_mark.png"), size="large", icon_image=str(ASSETS / "navapp_frame.png"))

page = st.navigation([
    st.Page("pipeline_app.py", title="Pipeline", icon="🛠️", default=True),
    st.Page("navapp_pages/chat.py", title="Chatbot", icon="💬"),
    st.Page("navapp_pages/chat_data.py", title="Chatbot data", icon="🗂️"),
])
page.run()
