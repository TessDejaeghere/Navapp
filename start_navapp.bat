@echo off
REM Navapp — Fien's letter pipeline. Double-click to start.
REM One-time setup (in this folder):
REM     python -m venv .venv
REM     .venv\Scripts\pip install streamlit pandas requests openai pillow ^
REM         rapidfuzz jellyfish transformers safetensors
REM     .venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe -m streamlit run navapp.py
) else (
    python -m streamlit run navapp.py
)
pause
