"""Sharing Fien's corrections through GitHub.

Fien is the authority: she decides when her corrections are published. Publishing writes them to a
separate public repo (DATA_REPO), not to the code repo, so her GitHub key can only change corrections,
never the code that every Navapp installs automatically.

    manifest.json              when, by whom, how many
    notes.jsonl                facts corrected in the chat (rag/memory.py), without the conversations
    person_mentions.jsonl.gz   checked persons from her OpenRefine exports (rag/corrections.py)

Other installations check manifest.json, show a notice when something new was published, and on request
download the files into data/shared/. Those corrections are read-only there and are combined with the
local ones (memory.load_notes, corrections.load). Conversations and letters are never published.

For tests: NAVAPP_SHARED_SOURCE (folder to fetch from) and NAVAPP_SHARED_TARGET (folder to publish to).
"""
from __future__ import annotations

import base64
import gzip
import json
import os
import ssl
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHARED = ROOT / "data" / "shared"
TOKEN_FILE = ROOT.parent / "config" / "github_token.txt"
DATA_REPO = "TessDejaeghere/Navapp-corrections"
BRANCH = "main"
RAW = f"https://raw.githubusercontent.com/{DATA_REPO}/{BRANCH}/"
API = f"https://api.github.com/repos/{DATA_REPO}/contents/"
FILES = ("notes.jsonl", "person_mentions.jsonl.gz", "manifest.json")   # manifest last: it marks completion
NOTE_FIELDS = ("id", "created", "updated", "author", "fact", "letter_ids", "persons", "history")


def _open(req: urllib.request.Request, timeout: int = 60) -> bytes:
    with urllib.request.urlopen(req, timeout=timeout, context=ssl.create_default_context()) as r:
        return r.read()


# -- reading what Fien published ------------------------------------------------------
def _fetch(name: str, timeout: int = 60) -> bytes:
    local = os.environ.get("NAVAPP_SHARED_SOURCE")
    if local:
        return (Path(local) / name).read_bytes()
    return _open(urllib.request.Request(RAW + name, headers={"User-Agent": "Navapp", "Cache-Control": "no-cache"}),
                 timeout=timeout)


def local_manifest() -> dict:
    try:
        return json.loads((SHARED / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def check() -> dict:
    """{'published', 'by', ..., 'new': bool} or {'error'}; never raises."""
    try:
        try:
            remote = json.loads(_fetch("manifest.json", timeout=8))
        except urllib.error.HTTPError as e:
            if e.code != 404 or os.environ.get("NAVAPP_SHARED_SOURCE"):
                raise
            # raw.githubusercontent.com keeps a 'not found' for a few minutes after the first publication;
            # the API is not cached (60 unauthenticated calls per hour, plenty for an hourly check)
            remote = json.loads(_open(urllib.request.Request(
                API + "manifest.json", headers={"User-Agent": "Navapp", "Accept": "application/vnd.github.raw+json"}),
                timeout=8))
    except urllib.error.HTTPError as e:
        return {"error": "nothing published yet" if e.code == 404 else f"HTTP {e.code}"}
    except Exception as e:  # noqa: BLE001 - offline etc.: the app keeps working
        return {"error": f"{type(e).__name__}: {e}"}
    return {**remote, "new": remote.get("published", "") > local_manifest().get("published", "")}


def fetch() -> dict:
    """Download Fien's latest published corrections into data/shared/ (all files, then the manifest)."""
    data = {name: _fetch(name) for name in FILES}
    json.loads(data["manifest.json"])                      # refuse a broken download
    gzip.decompress(data["person_mentions.jsonl.gz"])
    SHARED.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        tmp = SHARED / (name + ".tmp")
        tmp.write_bytes(data[name])
        tmp.replace(SHARED / name)
    return json.loads(data["manifest.json"])


def shared_notes() -> list[dict]:
    p = SHARED / "notes.jsonl"
    if not p.exists():
        return []
    by = local_manifest().get("by", "")
    with p.open(encoding="utf-8") as fh:
        return [{**json.loads(line), "shared": True, "published_by": by} for line in fh if line.strip()]


def shared_person_mentions() -> list[dict]:
    p = SHARED / "person_mentions.jsonl.gz"
    if not p.exists():
        return []
    return [json.loads(line) for line in gzip.decompress(p.read_bytes()).decode("utf-8").splitlines() if line]


# -- publishing (Fien) -------------------------------------------------------------------
def token() -> str:
    try:
        return TOKEN_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def save_token(value: str) -> None:
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(value.strip(), encoding="utf-8")


def _api(method: str, name: str, tok: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(API + name, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json",
                                          "User-Agent": "Navapp", "X-GitHub-Api-Version": "2022-11-28"})
    return json.loads(_open(req) or b"{}")


def _put(name: str, content: bytes, tok: str, message: str) -> None:
    target = os.environ.get("NAVAPP_SHARED_TARGET")
    if target:   # test mode
        Path(target).mkdir(parents=True, exist_ok=True)
        (Path(target) / name).write_bytes(content)
        return
    try:
        sha = _api("GET", name, tok).get("sha")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        sha = None
    body = {"message": message, "content": base64.b64encode(content).decode(), "branch": BRANCH}
    if sha:
        body["sha"] = sha
    _api("PUT", name, tok, body)


def publish(by: str, tok: str | None = None) -> dict:
    """Publish this installation's corrections. Returns the manifest."""
    from . import corrections, memory   # here, to avoid an import cycle
    tok = tok or token()
    if not tok and not os.environ.get("NAVAPP_SHARED_TARGET"):
        raise RuntimeError("No GitHub key saved yet.")
    notes = [{k: n.get(k) for k in NOTE_FIELDS} for n in memory.load_notes(include_shared=False)]
    mentions = corrections.load(include_shared=False)
    manifest = {"published": datetime.now().isoformat(timespec="seconds"), "by": by or "research team",
                "notes": len(notes), "person_letters": len({m["letter_key"] for m in mentions}),
                "person_mentions": len(mentions)}
    files = {
        "notes.jsonl": "".join(json.dumps(n, ensure_ascii=False) + "\n" for n in notes).encode("utf-8"),
        "person_mentions.jsonl.gz": gzip.compress(
            "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in mentions).encode("utf-8"), mtime=0),
        "manifest.json": json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8"),
    }
    message = f"Corrections published by {manifest['by']} ({manifest['notes']} chat corrections, " \
              f"{manifest['person_letters']} letters with checked persons)"
    for name in FILES:
        _put(name, files[name], tok, message)
    SHARED.mkdir(parents=True, exist_ok=True)   # so the publisher's own app shows no 'new corrections' notice
    (SHARED / "manifest.json").write_bytes(files["manifest.json"])
    return manifest
