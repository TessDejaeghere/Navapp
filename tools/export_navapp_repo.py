#!/usr/bin/env python3
"""
Copy the Navapp code (no data, no model, no keys) into the separate Navapp git repo.

The Navapp repo keeps the research-repo layout, because the modules find each other
through paths relative to the repo root. Run this after changing Navapp here, then
commit and push in the Navapp repo:

    python tools/export_navapp_repo.py                  # -> C:/Users/tessd/navapp
    python tools/export_navapp_repo.py --target D:/x    # elsewhere

Files that exist only in the Navapp repo (README.md, .gitignore) are never touched.

To ship the change to Fien, add --release with a short note; her Navapp then offers the update:

    python tools/export_navapp_repo.py --release "Funny messages while the chatbot works" Files listed here but deleted from the research repo are
reported, not removed.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CODE = [
    # the app
    "navapp.py", "navapp_update.py", "navapp_version.json", "pipeline_app.py", "start_navapp.bat", ".streamlit/config.toml",
    "navapp_pages/chat.py", "navapp_pages/chat_data.py",
    "assets/navapp_logo.png", "assets/navapp_frame.png", "assets/navapp_icon.png",
    "assets/navapp_mark.png", "assets/navapp_wordmark.png",
    # pipeline modules the app wraps
    "HTR_pipeline/run_htr_opus.py", "HTR_pipeline/download_kbr_iiif_batch2.py",
    "HTR_pipeline/orient_pages.py", "HTR_pipeline/train_orientation_classifier.py", "HTR_pipeline/ORIENTATION.md",
    "HTR_pipeline/models/orientation_cnn.pt", "HTR_pipeline/models/orientation_cnn.json",   # 1.7 MB, needed
    "NER_pipeline/predict_dalembert_corpus.py",
    "CLUSTERING_pipeline/link_persons.py", "CLUSTERING_pipeline/eval_clustering_linking.py",
    "CLUSTERING_pipeline/llm_adjudicate_shared_surnames.py",
    # chatbot
    "chatbot/rag/__init__.py", "chatbot/rag/agent.py", "chatbot/rag/answer.py", "chatbot/rag/corpus.py",
    "chatbot/rag/indexer.py", "chatbot/rag/ingest.py", "chatbot/rag/openrouter.py", "chatbot/rag/search.py",
    "chatbot/rag/stats.py", "chatbot/rag/corrections.py", "chatbot/rag/memory.py", "chatbot/rag/sharing.py",
    "chatbot/scripts/ask.py", "chatbot/scripts/ask_tools.py", "chatbot/scripts/build_corpus.py",
    "chatbot/scripts/build_index.py", "chatbot/scripts/eval_bot.py", "chatbot/scripts/eval_counts.py",
    "chatbot/app/chainlit_app.py", "chatbot/Start_NavappBot.bat", "chatbot/chainlit.md",
    "chatbot/.chainlit/config.toml", "chatbot/.gitignore", "chatbot/README.md",
    # tooling and documentation
    "tools/build_navapp_dist.py", "tools/export_navapp_repo.py",
]
RENAMED = {"chatbot/RAPPORT_pipeline_en_database.md": "docs/pipeline_en_database.md",
           "tools/navapp_requirements.txt": "requirements.txt"}
VERSION_FILE = REPO / "navapp_version.json"


def release(notes: str) -> str:
    """New version number (today's date, .2 .3 … for more releases on one day) with release notes."""
    today = date.today().strftime("%Y.%m.%d")
    try:
        old = json.loads(VERSION_FILE.read_text(encoding="utf-8"))["version"]
    except (OSError, ValueError, KeyError):
        old = ""
    if old == today or old.startswith(today + "."):
        n = int(old.split(".")[3]) + 1 if old.count(".") == 3 else 2
        version = f"{today}.{n}"
    else:
        version = today
    VERSION_FILE.write_text(json.dumps({"version": version, "notes": notes}, ensure_ascii=False, indent=1) + "\n",
                            encoding="utf-8")
    return version


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default=str(Path.home() / "navapp"))
    ap.add_argument("--release", metavar="NOTES",
                    help="publish a new version for Fien: bumps navapp_version.json with these release notes, "
                         "so her Navapp shows an update notice after you push")
    args = ap.parse_args()
    target = Path(args.target)
    if args.release:
        print("New version:", release(args.release))
    missing, copied = [], 0
    for src_rel, dst_rel in [(f, f) for f in CODE] + list(RENAMED.items()):
        src = REPO / src_rel
        if not src.exists():
            missing.append(src_rel)
            continue
        dst = target / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    print(f"Copied {copied} files to {target}")
    if missing:
        print("Not found (left as they are in the Navapp repo):", *missing, sep="\n  ")


if __name__ == "__main__":
    main()
