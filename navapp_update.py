"""Navapp updates from GitHub: check for a newer version, tell the user, install it on request.

The public repo github.com/TessDejaeghere/Navapp holds the code and navapp_version.json
({"version": "2026.11.23", "notes": "..."}). The app compares that with its own
navapp_version.json. Installing downloads the repo as a zip and copies its files over this
installation, except the folders that hold the user's own data, the model and the key (those
are not in the repo anyway; the check is a second safety net). Replaced files are kept under
_backup/<old version>/. Afterwards pip installs whatever requirements.txt lists that is missing.

Downloads go through urllib with the Windows certificate store, which also works behind virus
scanners and proxies that break the certificate bundle shipped with `requests`.
For testing, NAVAPP_UPDATE_SOURCE can point to a local folder with navapp_version.json and main.zip.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import ssl
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = "TessDejaeghere/Navapp"
BRANCH = "main"
VERSION_FILE = ROOT / "navapp_version.json"
REMOTE_VERSION = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/navapp_version.json"
REMOTE_ZIP = f"https://codeload.github.com/{REPO}/zip/refs/heads/{BRANCH}"
PROTECTED = ("chatbot/data/", "data/", "config/", "NER_pipeline/ckpts/", "NER_pipeline/eval/",
             "CLUSTERING_pipeline/glossary/", ".venv/", "_backup/", ".git/")


def _get(url: str, timeout: int = 15) -> bytes:
    local = os.environ.get("NAVAPP_UPDATE_SOURCE")
    if local:   # test mode: a folder instead of GitHub
        return (Path(local) / ("main.zip" if url == REMOTE_ZIP else "navapp_version.json")).read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": "Navapp-updater", "Cache-Control": "no-cache"})
    with urllib.request.urlopen(req, timeout=timeout, context=ssl.create_default_context()) as r:
        return r.read()


def _key(version: str) -> tuple[int, ...]:
    return tuple(int(p) for p in version.split(".") if p.isdigit())


def local_version() -> dict:
    try:
        return json.loads(VERSION_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"version": "0", "notes": ""}


def check() -> dict:
    """{'current', 'latest', 'notes', 'available'} or {'error'} when GitHub cannot be reached."""
    current = local_version()["version"]
    try:
        remote = json.loads(_get(REMOTE_VERSION, timeout=8))
    except Exception as e:  # noqa: BLE001 - offline, blocked, repo private …: never break the app
        return {"current": current, "error": f"{type(e).__name__}: {e}"}
    return {"current": current, "latest": remote.get("version", "0"), "notes": remote.get("notes", ""),
            "available": _key(remote.get("version", "0")) > _key(current)}


def _protected(rel: str) -> bool:
    return rel.startswith(PROTECTED) or (rel.startswith("CLUSTERING_pipeline/") and rel.endswith(".csv"))


def install_packages(*packages: str) -> str:
    """pip install the given packages, or requirements.txt when none are given. Returns 'ok' or
    'failed: <end of pip's error>'. pip is told to use the Windows certificate store (truststore),
    like the downloads above: with its own certificate bundle, pip fails behind virus scanners and
    proxies that inspect HTTPS ('CERTIFICATE_VERIFY_FAILED'). A pip too old for that option
    (before 22.2) is simply run without it."""
    what = list(packages) or ["-r", str(ROOT / "requirements.txt")]
    base = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
    r = subprocess.run(base + ["--use-feature=truststore"] + what, capture_output=True, text=True)
    err = r.stderr.lower()
    if r.returncode != 0 and "truststore" in err and ("invalid choice" in err or "no such option" in err):
        r = subprocess.run(base + what, capture_output=True, text=True)
    return "ok" if r.returncode == 0 else f"failed: {r.stderr[-500:]}"


def install(progress=None) -> dict:
    """Download the newest version and copy it over this installation. Returns a summary."""
    say = progress or (lambda msg: None)
    old = local_version()["version"]
    say("Downloading the new version …")
    archive = zipfile.ZipFile(io.BytesIO(_get(REMOTE_ZIP, timeout=120)))
    names = [n for n in archive.namelist() if not n.endswith("/")]
    prefix = names[0].split("/", 1)[0] + "/"                        # GitHub puts everything in Navapp-main/
    files = {n[len(prefix):]: n for n in names if n.startswith(prefix)}
    if "navapp.py" not in files or "navapp_version.json" not in files:
        raise RuntimeError("The download does not look like Navapp.")
    old_requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8") if (ROOT / "requirements.txt").exists() else ""
    backup = ROOT / "_backup" / old
    changed, added, skipped = [], [], []
    say(f"Installing {len(files)} files …")
    for rel, member in sorted(files.items()):
        if _protected(rel) or ".." in Path(rel).parts:
            skipped.append(rel)
            continue
        data = archive.read(member)
        target = ROOT / rel
        if target.exists():
            if target.read_bytes() == data:
                continue
            (backup / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup / rel)
            changed.append(rel)
        else:
            added.append(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    # Always run pip: when everything is installed it only checks (a few seconds, no download), and
    # a package that failed to install during an earlier update gets a new chance.
    new_requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    say("Installing new Python packages (this can take a few minutes) …" if new_requirements != old_requirements
        else "Checking the Python packages …")
    packages = install_packages()
    return {"from": old, "to": local_version()["version"], "changed": changed, "added": added,
            "skipped": skipped, "backup": str(backup) if changed else "", "packages": packages}
