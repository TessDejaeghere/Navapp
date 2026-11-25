"""
Fien's letter pipeline — image(s) in, per-mention CSV out.

One Streamlit app wrapping the production pipeline, stage by stage, each
stage toggleable:

    HTR (Claude Opus via OpenRouter)
      -> NER (fine-tuned d'AlemBERT, CPU)
        -> mention clustering (honorific-strip + gender-split rule)
          -> optional LLM adjudication of shared-surname clusters
            -> person linking against the nodegoat glossary

The app is STATELESS with respect to the research corpus: it never writes to
data/ — every upload produces a standalone CSV download in the same format as
data/ner_corpus/per_mentions_openrefine.csv.

Run locally:
    streamlit run pipeline_app.py        (or double-click start_navapp.bat)

Desktop use (no server needed) — requirements on Fien's machine:
  - 64-bit Windows/macOS, Python 3.11+, ~3 GB free disk
    (checkpoint 0.4 GB + CPU torch + transformers), internet access
    (OpenRouter for HTR/LLM; Hugging Face once, for the tokenizer).
  - RAM: 4 GB free is enough with the 'Folder on this computer' input
    (images stay on disk and are loaded one letter at a time); browser
    zip-uploads of 1-3 GB additionally need the zip ~twice in RAM.
  - Speed on a normal laptop: first start ~15 s (model load); NER ~1-3 s
    per letter on CPU; HTR is network-bound at ~10-30 s per letter, so a
    100-letter batch is ~30-60 min and is dominated by the HTR API calls.

Deploy on navapp.ugent.be instead: see comments at the bottom of this file.
"""
from __future__ import annotations

import importlib.util
import io
import json
import re
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "NER_pipeline"))
sys.path.insert(0, str(REPO / "CLUSTERING_pipeline"))
sys.path.insert(0, str(REPO / "HTR_pipeline"))

# production pipeline pieces (no logic duplicated here)
from run_htr_opus import (  # noqa: E402
    MODEL_SLUG, PRICE_IN_PER_M, PRICE_OUT_PER_M, SYSTEM_PROMPT as HTR_PROMPT,
    image_to_data_uri, user_prompt as htr_user_prompt,
)
from predict_dalembert_corpus import (  # noqa: E402
    BASE_MODEL, encode_letter, decode_windows_to_spans, load_model,
    resolve_checkpoint,
)
from link_persons import (  # noqa: E402
    GLOSSARY_CSV, NIL, gender_of, link, load_corpus_prior, load_glossary, norm,
    split_surface, strip_honorifics,
)
import llm_adjudicate_shared_surnames as adj  # noqa: E402

KEY_FILE = REPO / "config" / "key_openrouter.txt"
CKPT_ROOT = REPO / "NER_pipeline" / "ckpts" / "dalembert_fold2" / "checkpoint-450"
HISTORY_CSV = REPO / "data" / "navapp_history.csv"
RUNS_DIR = REPO / "data" / "pipeline_runs"   # every run: its mention CSV and its letters, to reopen later
CONTEXT_SENTENCES = 2       # whole sentences before and after the mention's own sentence
CONTEXT_MAX_WORDS = 200     # cap per side, as in the Batch 1 export
# words that end with a full stop without ending the sentence (lower case, without the stop);
# a single capital letter (an initial, as in J.-L. David) never ends one either
ABBREVIATIONS = {"m", "mr", "mm", "mme", "mmes", "mlle", "mlles", "mgr", "st", "ste", "dr", "messrs",
                 "cie", "cte", "ctesse", "bon", "bne", "vve", "vte", "sr", "pp", "cf", "vol"}
SENTENCE_END = re.compile(r"[.!?…]+[»\"')\]]*\s+(?=[«\"'(\[]?[A-ZÀ-ÖØ-Þ])|\n\s*\n")

LINK_METHOD = "token_set"
LINK_THRESHOLD = 90.0
LINK_TIEBREAK = {"dates", "prior", "context"}

# Measured on the 2026 corpus runs: $129.21 for 3,929 pages of Opus HTR.
HTR_COST_PER_PAGE = 0.035
ADJ_COST_PER_CLUSTER = 0.03

GLOSSARY_REQUIRED_COLS = {"Name", "nodegoat ID", "family name", "maiden name",
                          "[date of birth] Date Start", "[date of death] Date Start"}

# Pillow in this stack decodes all of these (JP2 = JPEG-2000, common on IIIF servers).
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".jp2", ".webp", ".bmp")
IMAGE_TYPES = [e.lstrip(".") for e in IMAGE_EXTS]

CSV_COLUMNS = ["mention_id", "letter_id", "htr_file_id", "permalink", "label",
               "surface", "span_start", "span_end", "context_before",
               "context_after", "year", "sender", "receiver",
               "suggested_cluster", "cluster_size", "suggested_person_id",
               "suggested_person_name", "link_stage", "link_score",
               "person_candidates", "text_source"]

try:  # inside navapp.py the entry page has already set this; standalone it is set here
    st.set_page_config(page_title="Navapp — letter pipeline",
                       page_icon=str(REPO / "assets" / "navapp_icon.png"), layout="wide")
except st.errors.StreamlitAPIException:
    pass

CHATBOT_DIR = REPO / "chatbot"
PERMALINK = "https://opac.kbr.be/Library/doc/SYRACUSE/{idn}"


# ---------------------------------------------------------------------------
# Cached heavy resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the d'AlemBERT NER model (first start only) …")
def ner_resources():
    import torch  # noqa: F401
    from transformers import AutoTokenizer

    # resolve_checkpoint and load_model come from the batch script and stop with SystemExit when
    # something is missing; inside Streamlit that ends the run silently, so turn it into an error.
    try:
        log(f"loading the NER model from {CKPT_ROOT} …")
        ckpt = resolve_checkpoint(CKPT_ROOT)
        model = load_model(ckpt, "cpu")
        log("NER model loaded; loading the tokenizer …")
        id2label = {int(k): v for k, v in model.config.id2label.items()}
        tok_src = ckpt if (ckpt / "tokenizer_config.json").exists() else BASE_MODEL
        tokenizer = AutoTokenizer.from_pretrained(tok_src, add_prefix_space=True)
    except SystemExit as e:
        raise RuntimeError(str(e)) from None
    return tokenizer, model, id2label


def log(msg: str) -> None:
    """A line in the terminal window, so that a crash there shows how far the pipeline got."""
    print(f"[Navapp pipeline] {msg}", flush=True)


def ner_problem() -> str | None:
    """Why the NER model cannot run on this computer, or None when everything is in place."""
    weights = CKPT_ROOT / "model.safetensors"
    labels = REPO / "NER_pipeline" / "eval" / "gold_561_merged.jsonl"
    if not (weights.exists() or (CKPT_ROOT / "config.json").exists()):
        return (f"The NER model is missing: there is no model file in `{CKPT_ROOT.relative_to(REPO).as_posix()}`. "
                "It is not on GitHub (it is too large); it comes with the Navapp zip. Copy the folder "
                "`NER_pipeline/ckpts` from the zip into this Navapp folder and restart Navapp.")
    if not (CKPT_ROOT / "config.json").exists() and not labels.exists():
        return (f"The NER model's label list is missing (`{labels.relative_to(REPO).as_posix()}`). Copy the folder "
                "`NER_pipeline/eval` from the Navapp zip into this Navapp folder and restart Navapp.")
    return None


@st.cache_resource(show_spinner="Loading the person glossary …")
def linking_resources():
    gloss = load_glossary()
    gloss["surname"] = gloss["bare"].map(lambda s: split_surface(s)[2])
    prior = load_corpus_prior()
    name_of = dict(zip(gloss["nodegoat ID"], gloss["bare"]))
    return gloss, prior, name_of


def openrouter_key() -> str | None:
    k = st.session_state.get("api_key_override", "")
    if isinstance(k, str) and k.startswith("sk-or-"):
        return k
    if KEY_FILE.exists():
        k = KEY_FILE.read_text(encoding="utf-8").strip()
        if k.startswith("sk-or-"):
            return k
    return None


# ---------------------------------------------------------------------------
# "Already processed?" check
# ---------------------------------------------------------------------------
def _dup_key(s: str) -> str:
    """Separator/case-insensitive letter id: `Letter_100` == `Letter 100`,
    `Ms. II 70_1_23` == `Ms. II 70/1/23`."""
    return re.sub(r"[\s/_\-]+", " ", (s or "").strip().lower())


@st.cache_data(ttl=600)
def known_letters() -> dict[str, str]:
    """{normalised letter id -> human-readable provenance} from the corpus
    overview, the corpus mention CSV, and this app's own run history."""
    info: dict[str, str] = {}

    def add(key: str, note: str) -> None:
        k = _dup_key(key)
        if k:
            info[k] = f"{info[k]}; {note}" if k in info else note

    ov = REPO / "data" / "letters_overview.csv"
    if ov.exists():
        df = pd.read_csv(ov, encoding="utf-8-sig", dtype=str).fillna("")
        for r in df.itertuples():
            bits = []
            if r.manually_transcribed == "TRUE":
                bits.append("manual transcription")
            if r.HTR_corrected == "TRUE":
                bits.append("HTR corrected")
            elif r.re_HTR == "TRUE":
                bits.append("HTR'd by Opus")
            if r.Ms_key:
                add(r.Ms_key, "in the corpus overview (" + ", ".join(bits or ["status unknown"]) + ")")

    mcsv = REPO / "data" / "ner_corpus" / "per_mentions_openrefine.csv"
    if mcsv.exists():
        counts = pd.read_csv(mcsv, usecols=["letter_id"], dtype=str)["letter_id"].value_counts()
        for lid, n in counts.items():
            add(str(lid), f"{n} mentions in the corpus CSV")

    if HISTORY_CSV.exists():
        hist = pd.read_csv(HISTORY_CSV, dtype=str).fillna("")
        for r in hist.itertuples():
            add(r.letter_id, f"processed in this app on {r.date}")
    return info


def save_run(df: pd.DataFrame, letters: list[dict], ner_spans: dict) -> str:
    """Keep a run on disk (mention CSV + letters with their text), so that it survives a reload."""
    import datetime

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RUNS_DIR / f"{stamp}.csv", index=False, encoding="utf-8-sig")
    ids = {L["letter_id"] for L in letters}
    (RUNS_DIR / f"{stamp}.json").write_text(json.dumps({
        "when": datetime.datetime.now().isoformat(timespec="seconds"),
        "letters": letters, "mentions": len(df),
        "ner_spans": {k: [list(x) for x in v] for k, v in ner_spans.items() if k in ids},
    }, ensure_ascii=False), encoding="utf-8")
    return stamp


def list_runs() -> list[dict]:
    """Earlier runs, newest first: {stamp, when, letters, mentions}."""
    out = []
    for js in sorted(RUNS_DIR.glob("*.json"), reverse=True) if RUNS_DIR.exists() else []:
        try:
            meta = json.loads(js.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append({"stamp": js.stem, "when": meta.get("when", js.stem),
                    "letter_ids": [L["letter_id"] for L in meta.get("letters", [])],
                    "mentions": meta.get("mentions", 0)})
    return out


def reopen_run(stamp: str) -> None:
    meta = json.loads((RUNS_DIR / f"{stamp}.json").read_text(encoding="utf-8"))
    letters = [{**L, "images": []} for L in meta["letters"]]
    st.session_state.letters = letters
    st.session_state.processed = [{k: L.get(k, "") for k in ("letter_id", "sender", "receiver", "year",
                                                              "text", "text_source")} for L in letters]
    st.session_state.result = pd.read_csv(RUNS_DIR / f"{stamp}.csv", dtype=str,
                                          encoding="utf-8-sig").fillna("")[CSV_COLUMNS]
    st.session_state.ner_spans = {k: [tuple(x) for x in v] for k, v in meta.get("ner_spans", {}).items()}
    st.session_state.pop("chatbot_sent", None)
    for k in list(st.session_state.keys()):
        if k.startswith(("txt_", "snd_", "rcv_", "yr_")):
            del st.session_state[k]


def letters_in_chatbot() -> set[str]:
    """Letters this app has sent to the chatbot (Step 4)."""
    up = CHATBOT_DIR / "data" / "corpus" / "uploads.jsonl"
    if not up.exists():
        return set()
    ids = set()
    with up.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                ids.add(_dup_key(json.loads(line).get("letter_id", "")))
            except json.JSONDecodeError:
                continue
    return ids


def record_processed(df: pd.DataFrame) -> None:
    """Append this run to the app history so later sessions can warn."""
    import datetime

    rows = (df.groupby("letter_id").size().rename("n_mentions").reset_index())
    rows["date"] = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    header = not HISTORY_CSV.exists()
    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows[["letter_id", "date", "n_mentions"]].to_csv(
        HISTORY_CSV, mode="a", header=header, index=False)
    known_letters.clear()


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------
def straighten_pages(paths: list[Path], out_dir: Path) -> tuple[list[Path], list[str]]:
    """Turn sideways / upside-down / skewed pages upright before HTR (HTR_pipeline/orient_pages.py,
    cascade mode). Returns the paths to send and a short note per page that was changed. Any
    problem (missing package, odd image) leaves that page as it was: never blocks the HTR."""
    try:
        from orient_pages import straighten_file
    except Exception as e:  # noqa: BLE001 - e.g. opencv not installed yet
        log(f"page orientation skipped: {type(e).__name__}: {e}")
        return paths, []
    out, notes = [], []
    for i, p in enumerate(paths, start=1):
        try:
            q, o = straighten_file(p, out_dir, mode="cascade", api_key=openrouter_key())
        except Exception as e:  # noqa: BLE001
            log(f"page {i}: orientation check failed ({type(e).__name__}: {e}); sent as is")
            q, o = p, None
        if o is not None and (o.rotation_cw or o.skew_deg):
            bits = ([f"turned {o.rotation_cw}° clockwise"] if o.rotation_cw else []) + \
                   ([f"straightened {o.skew_deg:+.1f}°"] if o.skew_deg else [])
            notes.append(f"page {i}: " + ", ".join(bits))
            log(notes[-1])
        out.append(q)
    return out, notes


def run_htr(image_bytes: list[tuple[str, bytes]], straighten: bool = True) -> tuple[str, float, list[str]]:
    """Transcribe one letter (all its pages) with the production prompt.
    Returns the text, the cost, and notes on pages that were turned upright first."""
    from openai import OpenAI

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key(),
                    default_headers={"HTTP-Referer": "https://github.com/local/navez-htr",
                                     "X-Title": "Navez pipeline app"})
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i, (name, data) in enumerate(image_bytes, start=1):
            if isinstance(data, Path):   # folder mode stores paths, loads lazily
                data = data.read_bytes()
            suffix = Path(name).suffix or ".jpg"
            p = Path(td) / f"page_{i:02d}{suffix}"
            p.write_bytes(data)
            paths.append(p)
        notes: list[str] = []
        if straighten:
            paths, notes = straighten_pages(paths, Path(td) / "upright")
        content = [{"type": "text", "text": htr_user_prompt(len(paths))}]
        for i, p in enumerate(paths, start=1):
            if len(paths) > 1:
                content.append({"type": "text", "text": f"--- Page {i} of {len(paths)} ---"})
            content.append({"type": "image_url", "image_url": {"url": image_to_data_uri(p)}})
        resp = client.chat.completions.create(
            model=MODEL_SLUG,
            messages=[{"role": "system", "content": HTR_PROMPT},
                      {"role": "user", "content": content}],
            temperature=0.0, max_tokens=8000,
        )
    text = (resp.choices[0].message.content or "").strip()
    u = resp.usage
    cost = ((u.prompt_tokens or 0) * PRICE_IN_PER_M
            + (u.completion_tokens or 0) * PRICE_OUT_PER_M) / 1e6 if u else 0.0
    return text, cost, notes


def run_ner(text: str) -> list[tuple[int, int, str]]:
    import torch

    tokenizer, model, id2label = ner_resources()
    windows = encode_letter(tokenizer, text)
    log(f"recognising entities: {len(text)} characters, {len(windows)} window(s)")
    provenance = [{"letter_id": "L", "offset_mapping": w["offset_mapping"]} for w in windows]
    preds = []
    with torch.no_grad():
        for i in range(0, len(windows), 8):
            chunk = windows[i:i + 8]
            batch = tokenizer.pad(
                {"input_ids": [w["input_ids"] for w in chunk],
                 "attention_mask": [w["attention_mask"] for w in chunk]},
                return_tensors="pt")
            logits = model(**batch).logits.argmax(-1).tolist()
            for w, row in zip(chunk, logits):
                preds.append(row[:len(w["input_ids"])])
    spans = decode_windows_to_spans(preds, provenance, id2label)
    return sorted(spans.get("L", set()))


def sentence_starts(text: str) -> list[int]:
    """Offsets where sentences begin. A full stop after an abbreviation (M., Mme., St.) or an initial
    (J.-L.) does not end a sentence; a blank line always does."""
    starts = [0]
    for m in SENTENCE_END.finditer(text):
        if m.group(0)[0] == ".":
            word = text[:m.start()].split()[-1:] or [""]
            word = word[0].strip("«\"'([").split("-")[-1]
            if word.lower() in ABBREVIATIONS or (len(word) == 1 and word.isupper()):
                continue
        starts.append(m.end())
    return starts


def context_window(text: str, start: int, end: int, n_sentences: int = CONTEXT_SENTENCES,
                   starts: list[int] | None = None) -> tuple[str, str]:
    """The mention's own sentence plus n whole sentences before and after it, at most
    CONTEXT_MAX_WORDS words per side."""
    starts = starts if starts is not None else sentence_starts(text)
    first = max(i for i, b in enumerate(starts) if b <= start)
    last = max(i for i, b in enumerate(starts) if b < max(end, start + 1))
    lo = starts[max(0, first - n_sentences)]
    hi_idx = last + 1 + n_sentences
    hi = starts[hi_idx] if hi_idx < len(starts) else len(text)
    before = " ".join(text[lo:start].split()[-CONTEXT_MAX_WORDS:])
    after = " ".join(text[end:hi].split()[:CONTEXT_MAX_WORDS])
    return before, after


def build_mentions(letters: list[dict]) -> pd.DataFrame:
    """NER over every letter -> one row per PER/PER_NOM mention."""
    import hashlib

    rows = []
    all_spans = st.session_state.setdefault("ner_spans", {})
    n_ctx = st.session_state.get("context_sentences", CONTEXT_SENTENCES)
    for L in letters:
        spans = run_ner(L["text"])
        starts = sentence_starts(L["text"])
        all_spans[L["letter_id"]] = spans   # every label, also LOC/ORG/ART/EVT, for the chatbot
        for s, e, lab in spans:
            if lab not in ("PER", "PER_NOM"):
                continue
            surface = L["text"][s:e]
            before, after = context_window(L["text"], s, e, n_ctx, starts)
            mid = hashlib.md5(f"{L['letter_id']}::{s}-{e}".encode()).hexdigest()[:10]
            rows.append({
                "mention_id": mid, "letter_id": L["letter_id"],
                "htr_file_id": "", "permalink": "",
                "label": lab, "surface": surface,
                "span_start": s, "span_end": e,
                "context_before": before, "context_after": after,
                "year": L.get("year", ""), "sender": L.get("sender", ""),
                "receiver": L.get("receiver", ""),
                "text_source": L.get("text_source", "app_upload"),
            })
    return pd.DataFrame(rows)


def cluster_mentions(df: pd.DataFrame) -> pd.Series:
    """Shipped fallback rule: honorific-strip corpus-wide + gender split."""
    base = df["surface"].map(strip_honorifics)
    gender = df["surface"].map(gender_of)
    is_per = df["label"] == "PER"
    key = pd.Series("", index=df.index)
    glob = is_per & (base != "")
    conflicted = {k for k, grp in gender[glob].groupby(base[glob].values)
                  if {"M", "F"} <= set(grp)}
    key[glob] = "gs::" + base[glob] + gender[glob].where(
        base[glob].isin(conflicted) & (gender[glob] == "F"), "").replace("F", "::F")
    local = ~glob
    key[local] = ("lt::" + df.loc[local, "letter_id"].astype(str)
                  + "::" + df.loc[local, "surface"].map(norm))
    return key


def adjudicate(df: pd.DataFrame, clusters: pd.Series, max_cluster: int = 50) -> tuple[pd.Series, list[str]]:
    """LLM pass over risky shared-surname clusters (experiment D, capped)."""
    gloss, _prior, _ = linking_resources()
    surname_counts = gloss.groupby("surname").size()
    out = clusters.copy()
    notes = []
    api_key = openrouter_key()
    per = df[df["label"] == "PER"].copy()
    per["context_pre"] = per["context_before"]
    per["context_post"] = per["context_after"]
    per["recipient"] = per["receiver"]
    base = per["surface"].map(strip_honorifics)
    for key_name, grp in per.groupby(base):
        if (not key_name or " " in key_name or len(grp) < 2
                or len(grp) > max_cluster
                or int(surname_counts.get(key_name, 0)) < 2):
            continue
        prompt, id_map = adj.cluster_prompt(key_name, grp, gloss)
        try:
            resp = adj.call_openrouter(api_key, adj.DEFAULT_MODEL, adj.SYSTEM_PROMPT, prompt)
            raw = resp["choices"][0]["message"]["content"]
            assign = adj.parse_partition(raw, id_map)
        except Exception as e:  # noqa: BLE001 — surface any API/parse problem in the UI
            notes.append(f"'{key_name}': adjudication failed ({type(e).__name__}), kept rule clusters")
            continue
        n_groups = len(set(assign.values()))
        for mid, gi in assign.items():
            out[df["mention_id"] == mid] = f"llm::{key_name}::{gi}"
        notes.append(f"'{key_name}': {len(grp)} mentions -> {n_groups} person(s)")
    return out, notes


def link_mentions(df: pd.DataFrame, clusters: pd.Series) -> pd.DataFrame:
    gloss, prior, name_of = linking_resources()
    df = df.copy()
    df["suggested_cluster"] = clusters.values
    df["cluster_size"] = df.groupby("suggested_cluster")["mention_id"].transform("size")
    for col in ("suggested_person_id", "suggested_person_name", "link_stage",
                "link_score", "person_candidates"):
        # object, not pandas 3's default 'str' dtype: link_score receives numbers
        df[col] = pd.Series("", index=df.index, dtype=object)
    per = df[df["label"] == "PER"].copy()
    if per.empty:
        return df
    per["recipient"] = per["receiver"]
    from link_persons import cluster_candidates, apply_nil_rule
    cands = cluster_candidates(per, gloss, per["suggested_cluster"], LINK_METHOD,
                               LINK_TIEBREAK, prior, gender_guard=True,
                               phonetic_floor=92.0)
    linked = apply_nil_rule(per, per["suggested_cluster"], cands, LINK_THRESHOLD)
    cand_by_cluster = cands.set_index("cluster")
    idx = df.index[df["label"] == "PER"]
    df.loc[idx, "suggested_person_id"] = linked["pred_nodegoat_id"].values
    df.loc[idx, "link_stage"] = linked["stage"].values
    df.loc[idx, "suggested_person_name"] = [name_of.get(i, "") for i in linked["pred_nodegoat_id"]]
    df.loc[idx, "link_score"] = [cand_by_cluster["top1"].get(c, "") for c in linked["cluster"]]
    df.loc[idx, "person_candidates"] = [
        " || ".join(f"{name_of.get(i, i)}|{i}|{s:.0f}"
                    for i, s in (cand_by_cluster["alts"].get(c) or []))
        for c in linked["cluster"]]
    return df


# ---------------------------------------------------------------------------
# Hand-over to the Navapp chatbot
# ---------------------------------------------------------------------------
def chatbot_modules():
    """The chatbot's corpus, indexer and ingest modules (chatbot/rag)."""
    if str(CHATBOT_DIR) not in sys.path:
        sys.path.insert(0, str(CHATBOT_DIR))
    from rag import corpus, indexer, ingest
    return corpus, indexer, ingest


@st.cache_data(ttl=600)
def overview_metadata() -> dict[str, dict]:
    """Fien's nodegoat metadata per letter, to fill gaps (KBR link, archive, date)."""
    ov = REPO / "data" / "letters_overview.csv"
    out: dict[str, dict] = {}
    if ov.exists():
        df = pd.read_csv(ov, encoding="utf-8-sig", dtype=str).fillna("")
        for r in df.itertuples():
            k = _dup_key(r.Ms_key)
            if k and k not in out:
                out[k] = {"letter_id": r.Ms_key, "sender": r.Source, "recipient": r.Target, "date": r.Date,
                          "archive": r.Archive,
                          "permalink": PERMALINK.format(idn=r.IDN) if r.IDN else ""}
    return out


def nodegoat_letter_ids() -> dict[str, str]:
    """{separator-insensitive key -> letter ID as spelled in nodegoat}, from the chatbot index (all letters,
    also those without a transcription) and the letters overview."""
    import sqlite3

    out: dict[str, str] = {}
    try:
        _, indexer, _ = chatbot_modules()
        con = sqlite3.connect(indexer.current_dir() / "chunks.db")
        out.update({_dup_key(r[0]): r[0] for r in con.execute("SELECT letter_id FROM letters") if r[0]})
        con.close()
    except Exception:  # noqa: BLE001 - no index yet: fall back on the overview
        pass
    for k, m in overview_metadata().items():
        out.setdefault(k, m.get("letter_id", ""))
    return {k: v for k, v in out.items() if v}


def chatbot_chunks(processed: list[dict], result: pd.DataFrame | None, ner_spans: dict,
                   checked: bool) -> list[dict]:
    """Chunk records for the chatbot, in the same format as the research corpus."""
    corpus, _, _ = chatbot_modules()
    ov = overview_metadata()
    canonical = nodegoat_letter_ids()
    out = []
    for L in processed:
        lid = L["letter_id"]
        rows = result[result["letter_id"] == lid] if result is not None else pd.DataFrame()
        persons = [{"start": int(r.span_start), "end": int(r.span_end), "surface": r.surface, "label": r.label,
                    "person": r.suggested_person_name if r.suggested_person_id not in ("", NIL) else ""}
                   for r in rows.itertuples()]
        others = [(s, e, lab) for s, e, lab in ner_spans.get(lid, []) if lab in corpus.OTHER_LABELS]
        known = ov.get(_dup_key(lid), {})
        meta = {"sender": L.get("sender") or known.get("sender", ""),
                "recipient": L.get("receiver") or known.get("recipient", ""),
                "date": L.get("year") or known.get("date", ""),
                "archive": known.get("archive", ""), "permalink": known.get("permalink", "")}
        source = "corrected_htr" if checked and L["text_source"] == "opus_htr" else L["text_source"]
        # a zip folder cannot hold '/', so "Ms. II 70_1_23" arrives for "Ms. II 70/1/23": use nodegoat's spelling,
        # or the chatbot would count the letter twice (metadata without text, text without metadata)
        out += corpus.make_chunks(canonical.get(_dup_key(lid), lid), L["text"], meta, persons, others, source)
    return out


def letters_from_zip(data: bytes) -> dict[str, list[tuple[str, bytes]]]:
    """Group zip images into letters: by subfolder, else by `<letter>_<page>` stem."""
    out: dict[str, list[tuple[str, bytes]]] = {}
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for n in sorted(zf.namelist()):
            if n.endswith("/") or not n.lower().endswith(IMAGE_EXTS):
                continue
            parts = Path(n)
            if len(parts.parts) > 1:
                letter = parts.parts[-2]
            else:
                m = re.match(r"(.+?)[_-]\d+$", parts.stem)
                letter = m.group(1) if m else parts.stem
            out.setdefault(letter, []).append((parts.name, zf.read(n)))
    return out


TEXT_EXTS = (".txt", ".md")
TABLE_EXTS = (".csv", ".tsv")
COLUMN_NAMES = {   # accepted column names in a transcription table (lower case)
    "letter_id": ("letter_id", "letter id", "letter", "id", "brief", "brief_id", "brief-id", "briefid"),
    "text": ("text", "transcription", "transcriptie", "tekst", "transcript"),
    "sender": ("sender", "afzender", "from"),
    "receiver": ("receiver", "recipient", "ontvanger", "to", "addressee"),
    "year": ("year", "jaar", "date", "datum"),
}


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _transcription_table(name: str, data: bytes) -> list[dict]:
    """A CSV/TSV with one letter per row: letter ID and text, optionally sender, recipient, year."""
    sep = "\t" if name.lower().endswith(".tsv") else None
    df = pd.read_csv(io.StringIO(_decode(data)), sep=sep, engine="python", dtype=str).fillna("")
    cols = {c.strip().lower(): c for c in df.columns}
    found = {key: next((cols[a] for a in aliases if a in cols), None) for key, aliases in COLUMN_NAMES.items()}
    if not (found["letter_id"] and found["text"]):
        raise ValueError(f"{name}: needs a column for the letter ID (e.g. 'letter_id') and one for the "
                         f"text (e.g. 'text' or 'transcription'); found: {', '.join(df.columns)}")
    out = []
    for _, r in df.iterrows():
        lid, text = r[found["letter_id"]].strip(), r[found["text"]].strip()
        if lid and text:
            out.append({"letter_id": lid, "text": text,
                        **{k: (r[found[k]].strip() if found[k] else "") for k in ("sender", "receiver", "year")}})
    return out


def letters_from_transcriptions(files: list[tuple[str, bytes]]) -> list[dict]:
    """Existing transcriptions in bulk: .txt files (the file name is the letter ID), zips of them,
    or a CSV/TSV table with one letter per row. Returns letters ready for entity extraction."""
    found: list[dict] = []
    for name, data in files:
        low = name.lower()
        if low.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                inner = [(n, zf.read(n)) for n in sorted(zf.namelist())
                         if not n.endswith("/") and not Path(n).name.startswith((".", "__"))
                         and n.lower().endswith(TEXT_EXTS + TABLE_EXTS)]
            found += letters_from_transcriptions(inner)
        elif low.endswith(TABLE_EXTS):
            found += _transcription_table(Path(name).name, data)
        elif low.endswith(TEXT_EXTS):
            text = _decode(data).strip()
            if text:
                found.append({"letter_id": Path(name).stem.strip(), "text": text,
                              "sender": "", "receiver": "", "year": ""})
    letters, seen = [], {}
    for L in found:   # a letter ID that occurs twice keeps its last text
        seen[L["letter_id"]] = {**L, "images": [], "text_source": "manual"}
    letters = list(seen.values())
    return letters


def letters_from_folder(root: Path) -> dict[str, list[tuple[str, Path]]]:
    """Same grouping as the zip, but images stay on disk and load lazily —
    the right mode for multi-GB batches on Fien's own machine."""
    out: dict[str, list[tuple[str, Path]]] = {}
    for p in sorted(root.rglob("*")):
        if not (p.is_file() and p.suffix.lower() in IMAGE_EXTS):
            continue
        if p.parent != root:
            letter = p.parent.name
        else:
            m = re.match(r"(.+?)[_-]\d+$", p.stem)
            letter = m.group(1) if m else p.stem
        out.setdefault(letter, []).append((p.name, p))
    return out


def letters_from_iiif(url_field: str) -> list[tuple[str, bytes]]:
    """Download the pages of one letter from IIIF manifest URL(s).

    Accepts several manifests separated by `|` (multi-part letters, as in the
    nodegoat CSV). Handles Presentation API v3 and falls back to v2.
    """
    from download_kbr_iiif_batch2 import fetch_json, manifest_image_urls
    import requests

    page_urls: list[str] = []
    for murl in [u.strip() for u in url_field.split("|") if u.strip()]:
        manifest = fetch_json(murl)
        urls = manifest_image_urls(manifest)  # v3: items -> AnnotationPage
        if not urls:  # v2: sequences -> canvases -> images -> resource
            for seq in manifest.get("sequences", []):
                for canvas in seq.get("canvases", []):
                    for img in canvas.get("images", []):
                        rid = (img.get("resource") or {}).get("@id")
                        if rid:
                            urls.append(rid)
        page_urls.extend(urls)
    pages = []
    for i, u in enumerate(page_urls, start=1):
        r = requests.get(u, timeout=120,
                         headers={"User-Agent": "Navez-correspondence-research (UGent GhentCDH)"})
        r.raise_for_status()
        pages.append((f"page_{i:04d}.jpg", r.content))
    return pages


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("Navapp — Navez letter pipeline")
st.caption("Image → transcription → entities → clusters → person links → CSV. "
           "Nothing is written to the research corpus; every run produces a standalone download.")

with st.sidebar:
    st.header("Stages")
    do_htr = st.toggle("1 · HTR (Claude Opus)", value=True,
                       help="Transcribe uploaded images via OpenRouter.")
    st.caption(f"≈ ${HTR_COST_PER_PAGE:.3f} per page")
    do_straighten = st.toggle("Turn pages upright first", value=True, disabled=not do_htr,
                              help="Pages scanned sideways, upside down or crooked are turned upright "
                                   "before HTR. A small local model decides; only for doubtful pages "
                                   "is a cheap Claude Haiku check asked (≈ $0.0002 a page). The "
                                   "original scans are not changed.")
    if do_straighten and do_htr and importlib.util.find_spec("cv2") is None:
        st.warning("Turning pages upright needs a package that is not installed yet (OpenCV). "
                   "Until then, pages are sent as they are.")
        if st.button("Install it now", key="install_cv2"):
            with st.spinner("Installing OpenCV (about a minute) …"):
                from navapp_update import install_packages
                res = install_packages("opencv-python-headless")
            if res == "ok":
                importlib.invalidate_caches()
                st.rerun()
            st.error(f"Installing did not work: {res}")
    do_ner = st.toggle("2 · Entity recognition (d'AlemBERT)", value=True)
    st.caption("free — runs locally on this server")
    do_cluster = st.toggle("3 · Mention clustering", value=True,
                           help="Honorific-stripped surface rule with gender split.")
    st.caption("free — rule-based")
    do_adjudicate = st.toggle("3b · LLM check on shared surnames", value=False,
                              help="Sends ambiguous surname clusters (Robert, David …) to Claude "
                                   "with letter context.")
    st.caption(f"≈ ${ADJ_COST_PER_CLUSTER:.2f} per ambiguous surname "
               "(typically a handful per batch)")
    do_link = st.toggle("4 · Person linking (nodegoat glossary)", value=True)
    st.caption("free — matches against the person list")
    st.slider("Context in the CSV: sentences before and after", 0, 6, CONTEXT_SENTENCES,
              key="context_sentences",
              help="The columns context_before and context_after hold the mention's own sentence plus this "
                   f"many whole sentences on either side (at most {CONTEXT_MAX_WORDS} words per side), "
                   "to make the review in OpenRefine easier.")

    st.divider()
    st.header("Settings")

    key_ok = openrouter_key() is not None
    with st.expander("🔑 OpenRouter API key", expanded=not key_ok):
        st.write("Status:", "✅ key available" if key_ok else "❌ no key configured")
        new_key = st.text_input("Paste your key (starts with sk-or-)", type="password",
                                key="new_api_key")
        if st.button("Save key"):
            nk = (new_key or "").strip()
            if nk.startswith("sk-or-"):
                KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
                KEY_FILE.write_text(nk, encoding="utf-8")
                st.session_state["api_key_override"] = nk
                st.success("Key saved — it will be used for HTR and the LLM check.")
                st.rerun()
            else:
                st.error("That does not look like an OpenRouter key (should start with sk-or-).")
        st.caption("Keys are created at openrouter.ai → Keys. Tip: set a monthly "
                   "spend limit on the key there.")

    with st.expander("👤 Person list (entity linking)"):
        if GLOSSARY_CSV.exists():
            import datetime
            n_persons = len(linking_resources()[0])
            mtime = datetime.datetime.fromtimestamp(GLOSSARY_CSV.stat().st_mtime)
            st.write(f"Current list: **{n_persons} persons** "
                     f"({GLOSSARY_CSV.name}, updated {mtime:%d/%m/%Y})")
        new_gloss = st.file_uploader("Upload a new nodegoat PERSON export (CSV)",
                                     type=["csv"], key="gloss_upload")
        if new_gloss is not None and st.button("Replace person list"):
            try:
                df_new = pd.read_csv(io.BytesIO(new_gloss.getvalue()), dtype=str)
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read the CSV: {e}")
                df_new = None
            if df_new is not None:
                missing = GLOSSARY_REQUIRED_COLS - set(df_new.columns)
                if missing:
                    st.error("This does not look like a nodegoat PERSON export — "
                             f"missing columns: {', '.join(sorted(missing))}")
                else:
                    import datetime
                    backup_dir = GLOSSARY_CSV.parent / "_archive"
                    backup_dir.mkdir(exist_ok=True)
                    if GLOSSARY_CSV.exists():
                        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                        (backup_dir / f"{GLOSSARY_CSV.stem}_{stamp}.csv").write_bytes(
                            GLOSSARY_CSV.read_bytes())
                    GLOSSARY_CSV.write_bytes(new_gloss.getvalue())
                    linking_resources.clear()
                    st.success(f"Person list replaced ({len(df_new)} rows). "
                               "The old list is backed up in glossary/_archive/.")
                    st.rerun()

    if (do_htr or do_adjudicate) and not key_ok:
        st.error("HTR and the LLM check need an OpenRouter key — add one under Settings.")

if "letters" not in st.session_state:
    st.session_state.letters = []

runs = list_runs()
if runs:
    with st.expander(f"📚 Letters already through the pipeline ({len(runs)} earlier run(s))",
                     expanded=not st.session_state.letters):
        st.caption("Every run is kept on this computer (data/pipeline_runs), also after closing Navapp or "
                   "an update. Reopen a run to see its entities again or to send it to the chatbot.")
        in_chat = letters_in_chatbot()
        seen = {}
        for r in reversed(runs):   # oldest first, so the newest run of a letter wins
            for lid in r["letter_ids"]:
                seen[lid] = r
        overview = pd.DataFrame([{"letter": lid, "last run": r["when"].replace("T", " ")[:16],
                                  "in the chatbot": "yes" if _dup_key(lid) in in_chat else "no"}
                                 for lid, r in sorted(seen.items(), key=lambda x: x[1]["when"], reverse=True)])
        st.dataframe(overview, hide_index=True, width="stretch", height=min(38 + 35 * len(overview), 300))
        labels = {r["stamp"]: f"{r['when'].replace('T', ' ')[:16]} — {len(r['letter_ids'])} letter(s), "
                              f"{r['mentions']} mentions: {', '.join(r['letter_ids'][:3])}"
                              f"{' …' if len(r['letter_ids']) > 3 else ''}" for r in runs}
        c1, c2, c3 = st.columns([3, 1, 1])
        pick = c1.selectbox("Run", list(labels), format_func=labels.get, label_visibility="collapsed")
        if c2.button("Reopen this run"):
            reopen_run(pick)
            st.rerun()
        c3.download_button("⬇️ Its CSV", (RUNS_DIR / f"{pick}.csv").read_bytes(),
                           file_name=f"mentions_{pick}.csv", mime="text/csv")
        frames = [pd.read_csv(RUNS_DIR / f"{r['stamp']}.csv", dtype=str, encoding="utf-8-sig").fillna("")
                  .assign(_run=r["stamp"]) for r in runs if (RUNS_DIR / f"{r['stamp']}.csv").exists()]
        if frames:   # a letter run more than once keeps only the mentions of its latest run
            allm = pd.concat(frames, ignore_index=True)
            latest = {lid: r["stamp"] for r in reversed(runs) for lid in r["letter_ids"]}
            allm = allm[allm["_run"] == allm["letter_id"].map(latest)].drop(columns="_run")
            st.download_button(f"⬇️ All mentions of all runs ({allm['letter_id'].nunique()} letters, "
                               f"{len(allm)} mentions) as one CSV",
                               allm.to_csv(index=False).encode("utf-8-sig"),
                               file_name="mentions_all_runs.csv", mime="text/csv")

st.subheader("Step 1 — letter input")
mode = st.radio("Input type",
                ["Images of one letter", "Zip with several letters",
                 "Folder on this computer", "IIIF manifest URL",
                 "Paste an existing transcription", "Transcriptions in bulk"],
                horizontal=True, label_visibility="collapsed")

with st.form("input_form"):
    letters_input: list[dict] = []
    if mode == "Images of one letter":
        files = st.file_uploader("Page images, in reading order",
                                 type=IMAGE_TYPES, accept_multiple_files=True)
        c1, c2, c3, c4 = st.columns(4)
        lid = c1.text_input("Letter ID *", placeholder="e.g. Letter 144 or Ms. II 70/1/23")
        sender = c2.text_input("Sender", placeholder="as in nodegoat, incl. dates")
        receiver = c3.text_input("Recipient", placeholder="e.g. François-Joseph Navez (1787-1869)")
        year = c4.text_input("Year", placeholder="e.g. 1824")
    elif mode == "Zip with several letters":
        zfile = st.file_uploader("Zip — one folder per letter, or files named `<letter>_<page>.jpg`",
                                 type=["zip"])
        st.caption("Sender/recipient/year can be filled in per letter after transcription. "
                   "For very large batches (1 GB+), prefer 'Folder on this computer'.")
    elif mode == "Folder on this computer":
        folder = st.text_input("Folder path",
                               placeholder=r"e.g. C:\Users\fien\Pictures\new_batch")
        st.caption("Best for large batches: images are read straight from disk, nothing is "
                   "uploaded. One subfolder per letter, or files named `<letter>_<page>.jpg`. "
                   f"Accepted formats: {', '.join(IMAGE_TYPES)}.")
    elif mode == "IIIF manifest URL":
        c1, c2, c3, c4 = st.columns(4)
        lid = c1.text_input("Letter ID *", placeholder="e.g. Ms. II 70/4/330")
        sender = c2.text_input("Sender")
        receiver = c3.text_input("Recipient")
        year = c4.text_input("Year")
        iiif_url = st.text_input("Manifest URL(s)",
                                 placeholder="https://iiif.kbr.be/manifest/v3/846164 "
                                             "(several manifests: separate with | )")
        st.caption("Pages are downloaded from the IIIF server (Presentation API v3 or v2).")
    elif mode == "Paste an existing transcription":
        c1, c2, c3, c4 = st.columns(4)
        lid = c1.text_input("Letter ID *")
        sender = c2.text_input("Sender")
        receiver = c3.text_input("Recipient")
        year = c4.text_input("Year")
        pasted = st.text_area("Transcription", height=200)
    else:
        tfiles = st.file_uploader("Text files, a zip of text files, or a table (CSV/TSV)",
                                  type=["txt", "md", "zip", "csv", "tsv"], accept_multiple_files=True)
        st.caption("**Text files:** one letter per file; the file name is the letter ID "
                   "(e.g. `Letter 144.txt`). **Table:** one letter per row, with a column `letter_id` "
                   "and a column `text` (or `transcription`), and optionally `sender`, `recipient` and "
                   "`year`. Missing senders, recipients and years can be filled in below, or are taken "
                   "from the nodegoat overview when the letters go to the chatbot.")
    add_to_batch = False
    if st.session_state.letters:
        add_to_batch = st.checkbox(
            f"Add to the {len(st.session_state.letters)} letter(s) already loaded, instead of "
            "replacing them", value=True,
            help="Lets you combine sources in one batch, e.g. a zip of scans and a set of existing "
                 "transcriptions, and export all their entities in one CSV. A letter loaded again "
                 "replaces its earlier version.")
    submitted = st.form_submit_button("Load letter(s)", type="primary")

if submitted:
    previous = list(st.session_state.letters) if add_to_batch else []
    st.session_state.letters = []
    st.session_state.pop("result", None)
    # clear stale per-letter widget state from a previous load, otherwise old
    # (possibly empty) text areas shadow freshly loaded/transcribed content;
    # letters kept in the batch carry their edits in the letter itself
    for k in list(st.session_state.keys()):
        if k.startswith(("txt_", "snd_", "rcv_", "yr_")):
            del st.session_state[k]
    if mode == "Images of one letter":
        if not files or not lid:
            st.error("Upload at least one image and give the letter an ID.")
        else:
            st.session_state.letters = [{
                "letter_id": lid, "sender": sender, "receiver": receiver,
                "year": year, "images": [(f.name, f.getvalue()) for f in files],
                "text": "", "text_source": "opus_htr"}]
    elif mode == "Zip with several letters":
        if not zfile:
            st.error("Upload a zip.")
        else:
            groups = letters_from_zip(zfile.getvalue())
            st.session_state.letters = [
                {"letter_id": k, "sender": "", "receiver": "", "year": "",
                 "images": v, "text": "", "text_source": "opus_htr"}
                for k, v in groups.items()]
            st.success(f"Found {len(groups)} letters in the zip.")
    elif mode == "Folder on this computer":
        root = Path((folder or "").strip().strip('"'))
        if not root.is_dir():
            st.error("That folder does not exist.")
        else:
            groups = letters_from_folder(root)
            if not groups:
                st.error(f"No images found under {root} (accepted: {', '.join(IMAGE_TYPES)}).")
            else:
                st.session_state.letters = [
                    {"letter_id": k, "sender": "", "receiver": "", "year": "",
                     "images": v, "text": "", "text_source": "opus_htr"}
                    for k, v in groups.items()]
                n_pages = sum(len(v) for v in groups.values())
                st.success(f"Found {len(groups)} letters ({n_pages} pages) in the folder.")
    elif mode == "IIIF manifest URL":
        if not lid or not (iiif_url or "").strip():
            st.error("Letter ID and manifest URL are both required.")
        else:
            try:
                with st.spinner("Downloading pages from the IIIF server …"):
                    pages = letters_from_iiif(iiif_url)
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not fetch the manifest: {type(e).__name__}: {e}")
                pages = []
            if pages:
                st.session_state.letters = [{
                    "letter_id": lid, "sender": sender, "receiver": receiver,
                    "year": year, "images": pages, "text": "",
                    "text_source": "opus_htr"}]
                st.success(f"Downloaded {len(pages)} pages.")
    elif mode == "Paste an existing transcription":
        if not lid or not pasted.strip():
            st.error("Letter ID and transcription are both required.")
        else:
            st.session_state.letters = [{
                "letter_id": lid, "sender": sender, "receiver": receiver,
                "year": year, "images": [], "text": pasted.strip(),
                "text_source": "manual"}]
    else:
        if not tfiles:
            st.error("Upload one or more text files, a zip or a table.")
        else:
            try:
                found = letters_from_transcriptions([(f.name, f.getvalue()) for f in tfiles])
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read the transcriptions: {e}")
                found = []
            if found:
                st.session_state.letters = found
                st.success(f"Loaded {len(found)} transcription(s).")
            elif tfiles:
                st.error("No transcriptions found in these files.")

    if previous:
        new_ids = {L["letter_id"] for L in st.session_state.letters}
        kept = [L for L in previous if L["letter_id"] not in new_ids]
        st.session_state.letters = kept + st.session_state.letters
        if st.session_state.letters != kept:
            st.info(f"The batch now holds {len(st.session_state.letters)} letters "
                    f"({len(kept)} kept, {len(new_ids)} added or replaced).")

letters = st.session_state.letters
if letters:
    st.subheader("Step 2 — transcription")
    # Duplicate check: has any of these letters been processed before?
    known = known_letters()
    dups = [(L["letter_id"], known[_dup_key(L["letter_id"])])
            for L in letters if _dup_key(L["letter_id"]) in known]
    if dups:
        lines = "\n".join(f"- **{lid}** — {why}" for lid, why in dups)
        st.warning(f"⚠️ {len(dups)} of these letters appear to have been processed "
                   f"before:\n{lines}\n\nYou can still continue — a re-run simply "
                   "produces a fresh CSV — but check first that this is intended.")
    needs_htr = [L for L in letters if not L["text"] and L["images"]]
    if needs_htr and do_htr:
        n_pages = sum(len(L["images"]) for L in needs_htr)
        est = n_pages * HTR_COST_PER_PAGE
        st.caption(f"💰 Estimated HTR cost: **≈ \\${est:.2f}** "
                   f"({n_pages} pages × \\${HTR_COST_PER_PAGE:.3f})")
        if st.button(f"Transcribe {len(needs_htr)} letter(s) with Claude Opus (≈ ${est:.2f})",
                     type="primary", disabled=not key_ok):
            total = 0.0
            prog = st.progress(0.0)
            for i, L in enumerate(needs_htr):
                with st.spinner(f"Transcribing {L['letter_id']} ({len(L['images'])} pages) …"):
                    try:
                        L["text"], cost, turned = run_htr(L["images"], straighten=do_straighten)
                        if turned:
                            st.caption(f"↻ {L['letter_id']}: " + "; ".join(turned))
                    except Exception as e:  # noqa: BLE001
                        if getattr(e, "status_code", None) == 402:
                            st.error(
                                f"OpenRouter refused the request for lack of credit (402), at "
                                f"{L['letter_id']}. Either the account balance is too low, or this "
                                "key's own credit limit is (nearly) used up. Check the balance under "
                                "Credits and the key's usage and limit under Keys on openrouter.ai; "
                                "the credits must be in the same account or organisation as the "
                                f"key.\n\nOpenRouter says: {e}")
                            st.stop()
                        raise
                    # seed the widget state, or the (empty) text area rendered
                    # before the click would shadow the new transcription
                    st.session_state[f"txt_{L['letter_id']}"] = L["text"]
                    total += cost
                prog.progress((i + 1) / len(needs_htr))
            st.success(f"Done — HTR cost ≈ ${total:.3f}")
    elif needs_htr and not do_htr:
        st.info("HTR is toggled off — paste or type the transcription below.")

    for L in letters:
        with st.expander(f"📄 {L['letter_id']} — {L.get('sender') or '?'} → "
                         f"{L.get('receiver') or '?'}, {L.get('year') or '?'}",
                         expanded=len(letters) == 1):
            L["text"] = st.text_area("Transcription (editable before entity extraction)",
                                     value=L["text"], height=220, key=f"txt_{L['letter_id']}")
            c1, c2, c3 = st.columns(3)
            L["sender"] = c1.text_input("Sender", value=L.get("sender", ""),
                                        key=f"snd_{L['letter_id']}")
            L["receiver"] = c2.text_input("Recipient", value=L.get("receiver", ""),
                                          key=f"rcv_{L['letter_id']}")
            L["year"] = c3.text_input("Year", value=L.get("year", ""),
                                      key=f"yr_{L['letter_id']}")

    ready = [L for L in letters if L["text"].strip()]
    st.subheader("Step 3 — entities, clusters, links")
    if do_adjudicate:
        st.caption(f"💰 LLM check: ≈ ${ADJ_COST_PER_CLUSTER:.2f} per ambiguous surname found "
                   "(usually a handful per batch — the exact clusters are only known after "
                   "entity extraction). All other enabled stages are free.")
    else:
        st.caption("💰 All enabled stages below are free (they run locally).")
    problem = ner_problem() if do_ner else None
    if not do_ner:
        st.info("NER is toggled off — enable it to produce the mention CSV.")
    elif problem:
        st.error(problem)
    elif st.button(f"Run the pipeline on {len(ready)} letter(s)", type="primary",
                   disabled=not ready):
        try:
            with st.spinner("Extracting entities …"):
                df = build_mentions(ready)
        except (Exception, SystemExit) as e:  # noqa: BLE001 - show the reason instead of a silent stop
            log("entity extraction FAILED:")
            traceback.print_exc()
            st.error(f"The entity extraction stopped: {type(e).__name__}: {e}\n\nThe first run also "
                     "downloads the d'AlemBERT settings from Hugging Face, so it needs an internet "
                     "connection. The full error is in the terminal window.")
            st.stop()
        log(f"entity extraction done: {len(df)} person mentions")
        st.session_state.processed = [{k: L.get(k, "") for k in ("letter_id", "sender", "receiver", "year",
                                                                  "text", "text_source")} for L in ready]
        st.session_state.pop("chatbot_sent", None)
        if df.empty:
            st.warning("No PER / PER_NOM mentions found.")
        else:
            if do_cluster:
                clusters = cluster_mentions(df)
            else:
                clusters = pd.Series("m::" + df["mention_id"], index=df.index)
            notes = []
            if do_cluster and do_adjudicate and key_ok:
                with st.spinner("LLM adjudication of shared surnames …"):
                    clusters, notes = adjudicate(df, clusters)
            if do_link:
                with st.spinner("Linking persons against the glossary …"):
                    df = link_mentions(df, clusters)
            else:
                df["suggested_cluster"] = clusters.values
                df["cluster_size"] = df.groupby("suggested_cluster")["mention_id"].transform("size")
                for col in ("suggested_person_id", "suggested_person_name",
                            "link_stage", "link_score", "person_candidates"):
                    df[col] = pd.Series("", index=df.index, dtype=object)
            st.session_state.result = df[CSV_COLUMNS]
            st.session_state.adj_notes = notes
            record_processed(df)
            try:
                save_run(df[CSV_COLUMNS], st.session_state.processed, st.session_state.get("ner_spans", {}))
            except OSError as e:
                st.warning(f"The run could not be saved to disk ({e}); download the CSV below.")

if "result" in st.session_state:
    df = st.session_state.result
    n_per = int((df["label"] == "PER").sum())
    linked = df[(df["label"] == "PER") & (df["suggested_person_id"] != "")
                & (df["suggested_person_id"] != NIL)]
    st.success(f"{len(df)} mentions ({n_per} PER) · "
               f"{len(linked)} linked to a nodegoat person · "
               f"{df['suggested_cluster'].nunique()} clusters")
    for note in st.session_state.get("adj_notes", []):
        st.caption(f"LLM adjudication — {note}")
    st.dataframe(
        df[["letter_id", "label", "surface", "suggested_cluster", "cluster_size",
            "suggested_person_name", "link_stage", "link_score"]].astype(str),
        width="stretch", height=380)
    st.download_button(
        "⬇️ Download mention CSV",
        df.to_csv(index=False).encode("utf-8-sig"),
        file_name="mentions.csv", mime="text/csv", type="primary")

if st.session_state.get("processed"):
    st.subheader("Step 4 — send to the Navapp chatbot")
    processed = st.session_state.processed
    st.caption("Adds these letters, with their metadata and the entities found above, to the chatbot. "
               "A letter you send again replaces its earlier version. 💰 Embedding costs well under "
               "a cent per ten letters.")
    st.dataframe(pd.DataFrame([{"letter": L["letter_id"], "sender": L["sender"], "recipient": L["receiver"],
                                "year": L["year"], "transcription": L["text_source"], "characters": len(L["text"])}
                               for L in processed]), width="stretch", hide_index=True)
    gaps = [L["letter_id"] for L in processed if not (L["sender"] and L["receiver"] and L["year"])]
    if gaps:
        st.warning(f"{len(gaps)} letter(s) lack a sender, recipient or year: {', '.join(gaps[:8])}"
                   f"{' …' if len(gaps) > 8 else ''}. Where the letter is in your nodegoat overview, those "
                   "values are filled in from there; otherwise the chatbot cannot filter or cite them by "
                   "correspondent or date. You can fill them in under Step 2 and run the pipeline again.")
    checked = st.checkbox("I checked and corrected these transcriptions",
                          help="Marks them as corrected HTR in the chatbot, so its answers do not warn "
                               "about possible reading errors.")
    if st.button(f"➕ Send {len(processed)} letter(s) to the chatbot", type="primary", disabled=not key_ok):
        _, indexer, ingest = chatbot_modules()
        res = st.session_state.get("result")
        chunks = chatbot_chunks(processed, res, st.session_state.get("ner_spans", {}), checked)
        ingest.add_letters(chunks)
        bar = st.progress(0.0, "Updating the chatbot index …")
        try:
            st.session_state.chatbot_sent = indexer.rebuild_index(
                progress=lambda f, m: bar.progress(min(max(f, 0.0), 1.0), m))
        except RuntimeError as e:
            st.error(f"The letters were saved, but the index could not be updated: {e}. "
                     "Try again later from the 'Chatbot data' page.")
    if not key_ok:
        st.caption("Sending to the chatbot needs the OpenRouter key (Settings in the sidebar).")
    sent = st.session_state.get("chatbot_sent")
    if sent:
        st.success(f"Done: the chatbot now knows {sent['letters']:,} letters (index {sent['version']}, "
                   f"{sent['newly_embedded']} new fragments, ${sent['cost_usd']:.4f}). "
                   "Open the **Chatbot** page to ask questions about them.")

# -----------------------------------------------------------------------------
# Deployment notes for navapp.ugent.be
# -----------------------------------------------------------------------------
# 1. Copy to the VM (checkpoint only once; ~430 MB):
#       scp pipeline_app.py tedjaegh@navapp.ugent.be:~/pipeline_app/
#       scp -r .streamlit tedjaegh@navapp.ugent.be:~/pipeline_app/
#       scp -r HTR_pipeline NER_pipeline CLUSTERING_pipeline tedjaegh@navapp.ugent.be:~/pipeline_app/
#       scp -r NER_pipeline/ckpts/dalembert_fold2/checkpoint-450 \
#              tedjaegh@navapp.ugent.be:~/pipeline_app/NER_pipeline/ckpts/dalembert_fold2/
#       scp data/letters_overview.csv tedjaegh@navapp.ugent.be:~/pipeline_app/data/
#       scp NER_pipeline/eval/gold_561_merged.jsonl \
#              tedjaegh@navapp.ugent.be:~/pipeline_app/NER_pipeline/eval/
#       scp config/key_openrouter.txt tedjaegh@navapp.ugent.be:~/pipeline_app/config/
#       ssh tedjaegh@navapp.ugent.be chmod 600 ~/pipeline_app/config/key_openrouter.txt
#
# 2. On the VM, in a venv (CPU-only torch keeps it small):
#       pip install streamlit pandas requests openai pillow rapidfuzz jellyfish \
#                   transformers safetensors
#       pip install torch --index-url https://download.pytorch.org/whl/cpu
#
# 3. systemd service + nginx `location /pipeline/` exactly as in
#    letters_overview_app.py, with two additions for the large zip uploads and
#    long HTR runs (inside the location block):
#       client_max_body_size 4g;
#       proxy_read_timeout 3600s;
#    Streamlit's own upload cap is set in .streamlit/config.toml
#    (maxUploadSize = 4096, i.e. 4 GB — copy .streamlit to the VM too).
#    Fien can save/replace the OpenRouter key and the person list from the
#    app's Settings sidebar; set a monthly spend limit on the key at
#    openrouter.ai. Replaced person lists are backed up under
#    CLUSTERING_pipeline/glossary/_archive/.
