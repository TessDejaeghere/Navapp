#!/usr/bin/env python3
"""Run the fine-tuned d'AlemBERT token classifier over every letter in the corpus.

The sliding-window encoding and the BIO decoder are ported verbatim from
``NER_pipeline/notebooks/colab_ner_encoders.ipynb`` so corpus predictions are
produced by the same procedure as the cross-validation predictions that the
evaluation tables report.

Text precedence per letter: corrected HTR > manual transcription > Opus HTR.

Usage (from repo root):
    python NER_pipeline/predict_dalembert_corpus.py --limit 5      # smoke test
    python NER_pipeline/predict_dalembert_corpus.py                # full corpus
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

OVERVIEW_CSV = REPO / "data" / "letters_overview.csv"
MANUAL_DIR = REPO / "data" / "manually_transcribed_letters"
CORRECTIONS_DIR = REPO / "data" / "labelstudio_upload" / "HTR_corrections_finished"
DEFAULT_CKPT = REPO / "NER_pipeline" / "ckpts" / "dalembert_fold0"
DEFAULT_OUT = REPO / "data" / "ner_corpus" / "dalembert_corpus_spans.jsonl"

# Frozen to match the fine-tuning recipe; changing either silently degrades
# quality because the model never saw windows of a different shape.
MAX_SEQ_LEN = 512
STRIDE = 128
BASE_MODEL = "pjox/dalembert"


# --------------------------------------------------------------------------
# Key normalisation
# --------------------------------------------------------------------------
def norm_key(s: str) -> str:
    """Canonical shelfmark form: ``Ms. II 70/<tome>/<number>``."""
    s = (s or "").strip()
    if not s:
        return ""
    if s.endswith(".txt"):
        s = s[:-4]
    s = s.replace("\\", "/").replace("_", "/")
    s = re.sub(r"\s+", " ", s).strip()
    m = re.match(r"^ms\.?\s*ii\s*70\s*/?\s*(.*)$", s, re.IGNORECASE)
    if m:
        rest = re.sub(r"\s*/\s*", "/", m.group(1).strip()).strip("/")
        return f"Ms. II 70/{rest}" if rest else "Ms. II 70"
    return s


# --------------------------------------------------------------------------
# Corpus text assembly
# --------------------------------------------------------------------------
def _corrected_texts() -> dict[str, str]:
    """Annotator-corrected transcriptions from finished Label Studio exports."""
    out: dict[str, str] = {}
    if not CORRECTIONS_DIR.exists():
        return out
    for path in sorted(CORRECTIONS_DIR.glob("HTR_corrections_*.json")):
        try:
            tasks = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  [warn] {path.name}: {exc}")
            continue
        for task in tasks:
            data = task.get("data") or {}
            key = norm_key(data.get("ms_key") or data.get("letter_id") or "")
            if not key:
                continue
            anns = task.get("annotations") or []
            non_cancelled = [a for a in anns if not a.get("was_cancelled")]
            if not non_cancelled:
                continue
            for res in non_cancelled[-1].get("result", []):
                if res.get("from_name") != "transcription":
                    continue
                txt = (res.get("value") or {}).get("text")
                if isinstance(txt, list):
                    txt = txt[0] if txt else None
                if isinstance(txt, str) and txt.strip():
                    out[key] = txt
                    break
    return out


def _manual_texts() -> dict[str, str]:
    out: dict[str, str] = {}
    if not MANUAL_DIR.exists():
        return out
    for path in sorted(MANUAL_DIR.glob("*.txt")):
        txt = path.read_text(encoding="utf-8", errors="replace")
        if txt.strip():
            out[norm_key(path.stem)] = txt
    return out


def _opus_path(raw: str) -> Path | None:
    """Resolve ``HTR_OPUS_path``, which holds ``; ``-joined duplicates."""
    for part in (raw or "").split(";"):
        part = part.strip().replace("\\", "/")
        if not part:
            continue
        p = Path(part)
        if not p.is_absolute():
            p = REPO / p
        if p.exists():
            return p
    return None


def load_corpus(limit: int = 0) -> list[dict]:
    """Return ``[{letter_id, text, text_source}, ...]`` for every letter with text."""
    corrected = _corrected_texts()
    manual = _manual_texts()
    print(f"  corrected HTR available : {len(corrected)}")
    print(f"  manual transcriptions   : {len(manual)}")

    letters: list[dict] = []
    seen: set[str] = set()
    with OVERVIEW_CSV.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = norm_key(row.get("Ms_key", ""))
            # The overview repeats some Ms_keys across rows (multi-page scans).
            if not key or key in seen:
                continue
            seen.add(key)
            if key in corrected:
                text, source = corrected[key], "corrected_htr"
            elif key in manual:
                text, source = manual[key], "manual"
            else:
                path = _opus_path(row.get("HTR_OPUS_path", ""))
                if path is None:
                    continue
                text, source = path.read_text(encoding="utf-8", errors="replace"), "opus_htr"
            if text.strip():
                letters.append({"letter_id": key, "text": text, "text_source": source})
            if limit and len(letters) >= limit:
                break
    return letters


# --------------------------------------------------------------------------
# Checkpoint resolution
# --------------------------------------------------------------------------
def resolve_checkpoint(root: Path) -> Path:
    """Return the directory holding the best checkpoint's weights.

    Prefers ``trainer_state.json -> best_model_checkpoint`` (the notebook trains
    with ``load_best_model_at_end=True``, ``metric_for_best_model='f1'``), then
    falls back to the highest ``checkpoint-N``.
    """
    if not root.exists():
        raise SystemExit(f"Checkpoint path does not exist: {root}")
    if (root / "config.json").exists() or (root / "model.safetensors").exists():
        return root

    subdirs = sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not subdirs:
        raise SystemExit(
            f"No config.json and no checkpoint-* directories under {root}.\n"
            "The folder appears to contain no model weights."
        )

    for cand in reversed(subdirs):
        state_file = cand / "trainer_state.json"
        if not state_file.exists():
            continue
        try:
            best = json.loads(state_file.read_text(encoding="utf-8")).get("best_model_checkpoint")
        except json.JSONDecodeError:
            continue
        if best:
            best_dir = root / Path(best).name
            if (best_dir / "config.json").exists() or (best_dir / "model.safetensors").exists():
                print(f"  best checkpoint (from trainer_state): {best_dir.name}")
                return best_dir

    print(f"  [warn] no trainer_state.json; falling back to latest: {subdirs[-1].name}")
    return subdirs[-1]


def bio_labels() -> list[str]:
    """Rebuild the training label order from the gold file (notebook cell 3)."""
    gold_path = REPO / "NER_pipeline" / "eval" / "gold_561_merged.jsonl"
    raw = set()
    with gold_path.open(encoding="utf-8") as f:
        for line in f:
            for span in json.loads(line)["gold_spans"]:
                raw.add(span[2])
    return ["O"] + [f"{p}-{lab}" for lab in sorted(raw) for p in ("B", "I")]


def load_model(ckpt: Path, device: str):
    """Load the classifier, reconstructing config for weights-only exports."""
    from transformers import AutoConfig, AutoModelForTokenClassification

    if (ckpt / "config.json").exists():
        return AutoModelForTokenClassification.from_pretrained(ckpt).to(device).eval()

    weights = ckpt / "model.safetensors"
    if not weights.exists():
        raise SystemExit(f"No config.json and no model.safetensors in {ckpt}")

    labels = bio_labels()
    print(f"  [warn] no config.json; rebuilding from {BASE_MODEL} with {len(labels)} BIO labels")
    config = AutoConfig.from_pretrained(
        BASE_MODEL,
        num_labels=len(labels),
        id2label={i: l for i, l in enumerate(labels)},
        label2id={l: i for i, l in enumerate(labels)},
    )
    model = AutoModelForTokenClassification.from_config(config)

    from safetensors.torch import load_file
    missing, unexpected = model.load_state_dict(load_file(str(weights)), strict=False)
    dropped = [k for k in missing if "position_ids" not in k]
    if dropped or unexpected:
        raise SystemExit(f"Weight mismatch. missing={dropped[:5]} unexpected={unexpected[:5]}")
    return model.to(device).eval()


# --------------------------------------------------------------------------
# Inference (ported from the Colab notebook)
# --------------------------------------------------------------------------
def encode_letter(tokenizer, text: str) -> list[dict]:
    enc = tokenizer(
        text,
        max_length=MAX_SEQ_LEN,
        truncation=True,
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )
    return [
        {
            "input_ids": enc["input_ids"][i],
            "attention_mask": enc["attention_mask"][i],
            "offset_mapping": enc["offset_mapping"][i],
        }
        for i in range(len(enc["input_ids"]))
    ]


def decode_windows_to_spans(pred_ids_per_window, provenance, id2label) -> dict[str, set]:
    from collections import defaultdict

    per_letter: dict[str, set] = defaultdict(set)
    for pred_ids, prov in zip(pred_ids_per_window, provenance):
        offsets = prov["offset_mapping"]
        lid = prov["letter_id"]
        cur_label = cur_start = cur_end = None
        for tag_id, (a, b) in zip(pred_ids, offsets):
            if b <= a:
                continue
            tag = id2label.get(int(tag_id), "O")
            if tag == "O":
                if cur_label is not None:
                    per_letter[lid].add((cur_start, cur_end, cur_label))
                    cur_label = None
                continue
            prefix, lab = tag[:1], tag[2:]
            if prefix == "B" or cur_label != lab:
                if cur_label is not None:
                    per_letter[lid].add((cur_start, cur_end, cur_label))
                cur_label, cur_start, cur_end = lab, a, b
            else:
                cur_end = b
        if cur_label is not None:
            per_letter[lid].add((cur_start, cur_end, cur_label))
    return per_letter


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None, help="cpu / cuda / cuda:0 (default: auto)")
    ap.add_argument("--limit", type=int, default=0, help="cap letters, for smoke tests")
    args = ap.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.out = args.out.resolve()

    import torch
    from transformers import AutoTokenizer

    print("Resolving checkpoint ...")
    ckpt = resolve_checkpoint(args.checkpoint.resolve())
    print(f"  using: {ckpt}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt, device)
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    print(f"  device: {device}   labels: {sorted(set(v[2:] for v in id2label.values() if v != 'O'))}")

    tok_src = ckpt if (ckpt / "tokenizer_config.json").exists() else BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(tok_src, add_prefix_space=True)

    print("Loading corpus texts ...")
    letters = load_corpus(args.limit)
    by_source: dict[str, int] = {}
    for r in letters:
        by_source[r["text_source"]] = by_source.get(r["text_source"], 0) + 1
    print(f"  {len(letters)} letters with text: {by_source}")
    if not letters:
        raise SystemExit("No letters with text found.")

    windows, provenance = [], []
    for r in letters:
        for w in encode_letter(tokenizer, r["text"]):
            windows.append(w)
            provenance.append({"letter_id": r["letter_id"], "offset_mapping": w["offset_mapping"]})
    print(f"  {len(windows)} windows to score")

    pred_ids: list[list[int]] = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(windows), args.batch_size):
            chunk = windows[i:i + args.batch_size]
            batch = tokenizer.pad(
                {"input_ids": [w["input_ids"] for w in chunk],
                 "attention_mask": [w["attention_mask"] for w in chunk]},
                return_tensors="pt",
            ).to(device)
            logits = model(**batch).logits.argmax(-1).cpu().tolist()
            # Trim padding so predictions stay aligned with offset_mapping.
            for w, row in zip(chunk, logits):
                pred_ids.append(row[:len(w["input_ids"])])
            done = min(i + args.batch_size, len(windows))
            if (i // args.batch_size) % 20 == 0:
                rate = done / (time.perf_counter() - t0)
                eta = (len(windows) - done) / rate if rate else 0
                print(f"    {done}/{len(windows)}  {rate:.1f} win/s  eta {eta / 60:.1f} min", flush=True)
    elapsed = time.perf_counter() - t0
    print(f"  inference: {elapsed:.1f}s for {len(windows)} windows "
          f"({len(windows) / elapsed:.2f} win/s, {elapsed / len(letters):.2f}s per letter)")

    spans = decode_windows_to_spans(pred_ids, provenance, id2label)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with args.out.open("w", encoding="utf-8") as f:
        for r in letters:
            ordered = sorted(spans.get(r["letter_id"], set()))
            total += len(ordered)
            f.write(json.dumps({
                "letter_id": r["letter_id"],
                "text_source": r["text_source"],
                "pred_spans": [[s, e, lab] for (s, e, lab) in ordered],
            }, ensure_ascii=False) + "\n")

    print(f"\nWrote {args.out.relative_to(REPO)}  ({len(letters)} letters, {total} spans)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
