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

import io
import json
import re
import sys
import tempfile
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
CONTEXT_WORDS = 25

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
        ckpt = resolve_checkpoint(CKPT_ROOT)
        model = load_model(ckpt, "cpu")
        id2label = {int(k): v for k, v in model.config.id2label.items()}
        tok_src = ckpt if (ckpt / "tokenizer_config.json").exists() else BASE_MODEL
        tokenizer = AutoTokenizer.from_pretrained(tok_src, add_prefix_space=True)
    except SystemExit as e:
        raise RuntimeError(str(e)) from None
    return tokenizer, model, id2label


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
def run_htr(image_bytes: list[tuple[str, bytes]]) -> tuple[str, float]:
    """Transcribe one letter (all its pages) with the production prompt."""
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
    return text, cost


def run_ner(text: str) -> list[tuple[int, int, str]]:
    import torch

    tokenizer, model, id2label = ner_resources()
    windows = encode_letter(tokenizer, text)
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


def context_window(text: str, start: int, end: int) -> tuple[str, str]:
    before = " ".join(text[:start].split()[-CONTEXT_WORDS:])
    after = " ".join(text[end:].split()[:CONTEXT_WORDS])
    return before, after


def build_mentions(letters: list[dict]) -> pd.DataFrame:
    """NER over every letter -> one row per PER/PER_NOM mention."""
    import hashlib

    rows = []
    all_spans = st.session_state.setdefault("ner_spans", {})
    for L in letters:
        spans = run_ner(L["text"])
        all_spans[L["letter_id"]] = spans   # every label, also LOC/ORG/ART/EVT, for the chatbot
        for s, e, lab in spans:
            if lab not in ("PER", "PER_NOM"):
                continue
            surface = L["text"][s:e]
            before, after = context_window(L["text"], s, e)
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
        df[col] = ""
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
                out[k] = {"sender": r.Source, "recipient": r.Target, "date": r.Date, "archive": r.Archive,
                          "permalink": PERMALINK.format(idn=r.IDN) if r.IDN else ""}
    return out


def chatbot_chunks(processed: list[dict], result: pd.DataFrame | None, ner_spans: dict,
                   checked: bool) -> list[dict]:
    """Chunk records for the chatbot, in the same format as the research corpus."""
    corpus, _, _ = chatbot_modules()
    ov = overview_metadata()
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
        out += corpus.make_chunks(lid, L["text"], meta, persons, others, source)
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

st.subheader("Step 1 — letter input")
mode = st.radio("Input type",
                ["Images of one letter", "Zip with several letters",
                 "Folder on this computer", "IIIF manifest URL",
                 "Paste an existing transcription"],
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
    else:
        c1, c2, c3, c4 = st.columns(4)
        lid = c1.text_input("Letter ID *")
        sender = c2.text_input("Sender")
        receiver = c3.text_input("Recipient")
        year = c4.text_input("Year")
        pasted = st.text_area("Transcription", height=200)
    submitted = st.form_submit_button("Load letter(s)", type="primary")

if submitted:
    st.session_state.letters = []
    st.session_state.pop("result", None)
    # clear stale per-letter widget state from a previous load, otherwise old
    # (possibly empty) text areas shadow freshly loaded/transcribed content
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
    else:
        if not lid or not pasted.strip():
            st.error("Letter ID and transcription are both required.")
        else:
            st.session_state.letters = [{
                "letter_id": lid, "sender": sender, "receiver": receiver,
                "year": year, "images": [], "text": pasted.strip(),
                "text_source": "manual"}]

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
                    L["text"], cost = run_htr(L["images"])
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
        except Exception as e:  # noqa: BLE001 - show the reason instead of a silent stop
            st.error(f"The entity extraction stopped: {e}\n\nThe first run also downloads the "
                     "d'AlemBERT settings from Hugging Face, so it needs an internet connection.")
            st.stop()
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
                    df[col] = ""
            st.session_state.result = df[CSV_COLUMNS]
            st.session_state.adj_notes = notes
            record_processed(df)

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
            "suggested_person_name", "link_stage", "link_score"]],
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
