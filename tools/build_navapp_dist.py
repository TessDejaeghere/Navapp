#!/usr/bin/env python3
"""
Assemble a self-contained desktop distribution of Navapp (pipeline_app.py).

Collects only what the app imports and reads at runtime:
  - the app + launcher + Streamlit config
  - the pipeline modules it wraps (HTR / NER / clustering / linking / LLM check)
  - the d'AlemBERT checkpoint (~420 MB) + the gold file its label order is
    rebuilt from
  - the person glossary and the corpus files used by the duplicate check
  - the chatbot (chatbot/rag), its corpus, current index version and embedding cache
  - setup.bat, requirements.txt and a README for a non-technical user

The OpenRouter key is deliberately NOT included: the recipient adds their own
key in the app's Settings sidebar.

Usage (from repo root):
    python tools/build_navapp_dist.py            # -> dist/navapp/
    python tools/build_navapp_dist.py --zip      # -> dist/navapp.zip as well
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIST = REPO / "dist" / "navapp"

# (source, keep-relative-path) — module REPO resolution relies on the layout
# staying identical to the research repo.
FILES = [
    "navapp.py",
    "navapp_update.py",                             # update check and installer (GitHub)
    "navapp_version.json",                                    # entry: Pipeline, Chatbot, Chatbot data pages
    "pipeline_app.py",
    "navapp_pages/chat.py",
    "navapp_pages/chat_data.py",
    "chatbot/rag/__init__.py",
    "chatbot/rag/agent.py",
    "chatbot/rag/stats.py",
    "chatbot/rag/corrections.py",
    "chatbot/rag/memory.py",
    "data/nodegoat_export_2106.csv",                # letter metadata for counting questions
    "chatbot/rag/answer.py",
    "chatbot/rag/corpus.py",
    "chatbot/rag/indexer.py",
    "chatbot/rag/ingest.py",
    "chatbot/rag/openrouter.py",
    "chatbot/rag/search.py",
    "chatbot/data/corpus/letters.jsonl",            # research corpus for the chatbot
    "chatbot/data/index/CURRENT",
    "chatbot/data/index/embed_cache.sqlite",        # so rebuilds on Fien's machine only embed new letters
    "start_navapp.bat",
    "assets/navapp_logo.png",
    "assets/navapp_frame.png",
    "assets/navapp_icon.png",
    "assets/navapp_mark.png",
    "assets/navapp_wordmark.png",
    ".streamlit/config.toml",
    "HTR_pipeline/run_htr_opus.py",
    "HTR_pipeline/download_kbr_iiif_batch2.py",
    "NER_pipeline/predict_dalembert_corpus.py",
    "NER_pipeline/eval/gold_561_merged.jsonl",
    "CLUSTERING_pipeline/link_persons.py",
    "CLUSTERING_pipeline/eval_clustering_linking.py",
    "CLUSTERING_pipeline/llm_adjudicate_shared_surnames.py",
    "CLUSTERING_pipeline/glossary/PERSON_export_2708.csv",
    "CLUSTERING_pipeline/pilot50_gold.csv",
    "data/letters_overview.csv",                    # corpus prior + duplicate check
    "data/ner_corpus/per_mentions_openrefine.csv",  # duplicate check (mention counts)
]
DIRS = [
    "NER_pipeline/ckpts/dalembert_fold2/checkpoint-450",
]
# Copied when present: letters already sent to the chatbot from this machine.
OPTIONAL_FILES = ["chatbot/data/corpus/uploads.jsonl", "data/nodegoat_letters_export.csv",
                  "chatbot/data/corrections/person_mentions.jsonl", "chatbot/data/corrections/imports.json"]


def chatbot_index_dir() -> str:
    """The chatbot index version currently in use (data/index/CURRENT)."""
    ptr = REPO / "chatbot" / "data" / "index" / "CURRENT"
    if not ptr.exists():
        raise SystemExit("No chatbot index: run chatbot/scripts/build_index.py first.")
    return f"chatbot/data/index/{ptr.read_text(encoding='utf-8').strip()}"

REQUIREMENTS = """\
streamlit
pandas
requests
openai
pillow
rapidfuzz
jellyfish
transformers
safetensors
sentence-transformers
--extra-index-url https://download.pytorch.org/whl/cpu
torch
"""

SETUP_BAT = """\
@echo off
REM One-time setup for Navapp. Needs Python 3.11+ installed
REM (python.org -> download -> tick "Add python.exe to PATH").
cd /d "%~dp0"
python --version >nul 2>&1
if errorlevel 1 (
    echo Python was not found. Install it from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" during installation.
    pause
    exit /b 1
)
echo Creating a private Python environment (one time, a few minutes) ...
python -m venv .venv
.venv\\Scripts\\python.exe -m pip install --upgrade pip
.venv\\Scripts\\python.exe -m pip install -r requirements.txt
echo.
echo Done! From now on, double-click start_navapp.bat to use the app.
pause
"""

README = """\
# Navapp — the Navez letter pipeline on your own computer

Turn letter scans into a spreadsheet of person mentions, linked to the
nodegoat person database. Everything runs on this computer; only the
handwriting transcription (HTR) and the optional LLM check call an online
service (OpenRouter), which costs a few cents per letter.

## One-time setup (±15 minutes)

1. Install Python 3.11 or newer from https://www.python.org/downloads/
   — IMPORTANT: tick "Add python.exe to PATH" in the installer.
2. Double-click `setup.bat` in this folder and wait for it to finish.
3. Double-click `start_navapp.bat`. Your browser opens with the app.
4. In the sidebar, open Settings -> "OpenRouter API key" and paste your key
   (create one at https://openrouter.ai -> Keys; set a monthly spend limit
   there too).

## Daily use

- Double-click `start_navapp.bat`; close the black window to stop the app.
- Choose how to load letters: images, a zip, a folder on this computer
  (best for big batches), an IIIF manifest URL, or paste a transcription.
- Toggle pipeline steps on/off in the sidebar; prices are shown before
  anything runs. The app warns if a letter was already processed before.
- Check/fix the transcription, run the pipeline, download the CSV.

## The chatbot

Navapp has three pages (left sidebar):

- **Pipeline**: from scans to the person CSV. At the end, *Step 4 — send to the Navapp
  chatbot* adds the letters you just processed (text, sender, recipient, year and the
  entities found) to the chatbot.
- **Chatbot**: ask questions about the letters in Dutch, English or French. Every answer
  cites its letters as [#n]; open a source to read the fragment and follow the KBR link.
  About $0.02 per answer.
  Conversations are saved (reopen them in the sidebar). When you correct the chatbot on a
  fact ("that letter is by Alexandre Robert, not Portaels"), it remembers the correction in
  all later conversations.
- **Chatbot data**: see which letters you sent and remove them again; upload your cleaned
  OpenRefine export (surface_v2, ID_Nodegoat, Notes, Source_v2, Target_v2) so the chatbot uses
  your checked persons; see, edit or delete the corrections the chatbot remembers.

## Updating Navapp
When you receive a new version, keep your own `chatbot/data` folder: it holds the letters you
sent, your OpenRefine corrections, saved conversations and remembered corrections. Copy the
new files over the old ones, but do not replace `chatbot/data`.

## Updating the person list

Export the PERSON list from nodegoat as CSV, then in the app:
Settings -> "Person list (entity linking)" -> upload -> Replace.
The old list is kept in CLUSTERING_pipeline/glossary/_archive/.

## Requirements

64-bit Windows, ±3 GB free disk space, 4 GB free RAM, internet access.
First start takes ±15 s (the language model loads); after that, entity
recognition takes a second per letter and transcription 10-30 s per letter.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", action="store_true", help="also write dist/navapp.zip")
    args = ap.parse_args()

    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)

    missing = []
    for rel in FILES:
        src = REPO / rel
        if not src.exists():
            missing.append(rel)
            continue
        dst = DIST / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for rel in OPTIONAL_FILES:
        if (REPO / rel).exists():
            (DIST / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / rel, DIST / rel)
    for rel in DIRS + [chatbot_index_dir()]:
        src = REPO / rel
        if not src.exists():
            missing.append(rel)
            continue
        shutil.copytree(src, DIST / rel)
    if missing:
        raise SystemExit(f"Missing sources, dist incomplete: {missing}")

    # the same file as in the GitHub repo, so an update only reinstalls packages when they really changed
    shutil.copy2(REPO / "tools" / "navapp_requirements.txt", DIST / "requirements.txt")
    (DIST / "setup.bat").write_text(SETUP_BAT, encoding="utf-8")
    (DIST / "README.txt").write_text(README, encoding="utf-8")
    (DIST / "config").mkdir(exist_ok=True)  # the app saves the API key here

    n_files = sum(1 for p in DIST.rglob("*") if p.is_file())
    size_mb = sum(p.stat().st_size for p in DIST.rglob("*") if p.is_file()) / 1e6
    print(f"dist/navapp: {n_files} files, {size_mb:.0f} MB")

    if args.zip:
        zpath = REPO / "dist" / "navapp.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(DIST.rglob("*")):
                if p.is_file():
                    zf.write(p, Path("navapp") / p.relative_to(DIST))
        print(f"dist/navapp.zip: {zpath.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
