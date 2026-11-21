#!/usr/bin/env python3
"""Experiment D — targeted LLM adjudication of shared-surname clusters.

Unlike experiment C (llm_cluster.py, whole-corpus clustering in one prompt),
this is a *post-pass on the shipped rule*: clustering stays rule-based
(honorific-stripped surface + conflict-only gender split), and only the risky
clusters are sent to the LLM, one prompt per cluster.

Trigger (production-faithful, no gold knowledge): a cluster whose key is a
bare single-token surname matched by >= 2 glossary persons, with >= 2 and
<= --max-cluster mentions. The size cap (default 50, dev-selected) keeps the
giant protagonist clusters out: there the rule is already reliable and a
single LLM misassignment costs more pairwise mass than every fix combined
(navez, 86 mentions: 2 misassigned mentions = -3.3 pair F1 on test).

Measured on pilot50 (claude-opus-4.8, 3 runs, identical scores):
    rule only          : test pair F1 0.883  B3 0.890  hard(no top-3) 0.665
    + LLM adjudication : test pair F1 0.888  B3 0.889  hard(no top-3) 0.680
The LLM uniquely separates jacques-louis David from the Rome-era Davids
(10 mentions, perfect) and splits the Roberts the priors could not (§3.4),
at ~$0.12 per corpus-pass over the capped groups.

The LLM sees each cluster's mentions with letter metadata + KWIC snippets and
the glossary candidates with life dates, and partitions the mentions into
persons. Everything else is untouched, so the score delta is attributable to
the adjudication alone.

Usage (from repo root):
    python CLUSTERING_pipeline/llm_adjudicate_shared_surnames.py --dry-run
    python CLUSTERING_pipeline/llm_adjudicate_shared_surnames.py
    python CLUSTERING_pipeline/llm_adjudicate_shared_surnames.py --repeat 3
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from eval_clustering_linking import (  # noqa: E402
    evaluate, load_gold, pairwise_prf, _gold_cluster_series,
)
from link_persons import (  # noqa: E402
    gender_of, load_glossary, norm, split_surface, strip_honorifics,
)

OUT_DIR = HERE / "llm_predictions"
KEY_FILE = REPO / "config" / "key_openrouter.txt"
DEFAULT_MODEL = "anthropic/claude-opus-4.8"

SYSTEM_PROMPT = """You are an art historian specialising in Franco-Belgian correspondence of the nineteenth century, in particular the circle of Francois-Joseph Navez (1787-1869), Belgian neo-classical painter, pupil of David in Rome, later director of the Academie in Brussels.

TASK: all the mentions below share the same (HTR-noisy) surname but may refer to DIFFERENT people. Partition them: group the mention ids that refer to the same real person.

You are given database candidates bearing this surname, with life dates. Use them as reference points, but a mention may also refer to a person absent from the database.

RULES:
- Use the letter context (sender, recipient, year, excerpt) to decide. A painter discussed in Rome in 1819 and a correspondent writing from Brussels in 1855 may be different bearers of the same name.
- Chronology: a person cannot be mentioned as alive after death or active before birth. Posthumous discussion of famous artists is common though.
- Gender: Mme/Mlle/veuve/comtesse mentions are never the same person as M./baron/comte mentions.
- When the evidence is genuinely insufficient to separate, keep mentions together (the default grouping assumes one person per surname).
- Every id must appear in exactly one group.

RESPONSE: valid JSON only, no surrounding text:
{"groups": [{"ids": [<int>, ...],
             "person": "<database candidate name, or NEW:<description>, or UNSURE>",
             "reasoning": "<at most 20 words>"}]}"""


def call_openrouter(api_key: str, model: str, system_prompt: str, user_prompt: str,
                    max_tokens: int = 4000, timeout: int = 600) -> dict:
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/tessd/Navez",
            "X-Title": "Navez shared-surname adjudication (experiment D)",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def shipped_rule_clusters(g: pd.DataFrame) -> pd.Series:
    """The corpus fallback: honorific-strip + conflict-only gender split."""
    base = g["surface"].map(strip_honorifics)
    gender = g["surface"].map(gender_of)
    df = pd.DataFrame({"k": base, "g": gender})
    conflicted = {k for k, grp in df.groupby("k") if {"M", "F"} <= set(grp["g"])}
    out = [k + "::F" if (k in conflicted and gen == "F") else k
           for k, gen in zip(base, gender)]
    return pd.Series(out, index=g.index), base


def risky_groups(g: pd.DataFrame, base: pd.Series, gloss: pd.DataFrame,
                 max_cluster: int = 50) -> dict[str, pd.DataFrame]:
    gloss = gloss.copy()
    gloss["surname"] = gloss["bare"].map(lambda s: split_surface(s)[2])
    surname_counts = gloss.groupby("surname").size()
    out = {}
    for key, grp in g.groupby(base):
        if not key or " " in key:
            continue
        if int(surname_counts.get(key, 0)) >= 2 and 2 <= len(grp) <= max_cluster:
            out[key] = grp
    return out


def cluster_prompt(key: str, grp: pd.DataFrame, gloss: pd.DataFrame) -> tuple[str, dict]:
    gl = gloss[gloss["surname"] == key]
    lines = [f"Surname group: \"{key}\" — {len(grp)} mentions.\n",
             "Database candidates:"]
    for _, c in gl.iterrows():
        def yr(v):
            return str(int(v)) if pd.notna(v) and v else "?"
        lines.append(f"  - {c['bare']} ({yr(c['born'])}–{yr(c['died'])})")
    lines.append("\nMentions. Format: id | surface | letter | sender -> recipient | year")
    id_map = {}
    for i, (_, r) in enumerate(sorted(grp.iterrows(),
                                      key=lambda t: (str(t[1]["year"]), t[1]["letter_id"])), start=1):
        id_map[i] = r["mention_id"]
        lines.append(f"{i:>3} | {r['surface']} | {r['letter_id']} | "
                     f"{r['sender'] or '?'} -> {r['recipient'] or '?'} | {r['year'] or '?'}")
        snip = f"…{r.get('context_pre') or ''} «{r['surface']}» {r.get('context_post') or ''}…"
        lines.append("      excerpt: " + re.sub(r"\s+", " ", snip).strip())
    lines.append("\nJSON response:")
    return "\n".join(lines), id_map


def parse_partition(raw: str, id_map: dict) -> dict:
    txt = raw.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.IGNORECASE)
    data = json.loads(txt)
    assign = {}
    for gi, grp in enumerate(data.get("groups") or []):
        for raw_id in grp.get("ids") or []:
            try:
                i = int(raw_id)
            except (TypeError, ValueError):
                continue
            if i in id_map and id_map[i] not in assign:
                assign[id_map[i]] = gi
    return assign


def score(gold: pd.DataFrame, pred_series: pd.Series, mention_ids: pd.Series,
          split: str) -> dict:
    pred = pd.DataFrame({"mention_id": mention_ids, "pred_cluster_id": pred_series.astype(str)})
    mets = evaluate(gold, pred, split=split)
    o = mets[mets["stratum"] == "OVERALL"].iloc[0]
    # hard subset: top-3 gold entities of this split removed
    sub = gold[gold["split"] == split].merge(pred, on="mention_id")
    gc = _gold_cluster_series(sub["gold_nodegoat_id"], sub["mention_id"])
    gs = gc.set_axis(sub["mention_id"].values)
    ps = sub["pred_cluster_id"].set_axis(sub["mention_id"].values)
    top3 = set(gs.value_counts().head(3).index)
    keep = ~gs.isin(top3)
    _, _, hard_f1 = pairwise_prf(gs[keep.values], ps[keep.values])
    return {"pair_f1": o["pair_f1"], "b3_f1": o["b3_f1"], "hard_pair_f1": hard_f1}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--max-cluster", type=int, default=50,
                    help="skip surname groups larger than this (default 50)")
    ap.add_argument("--dry-run", action="store_true", help="print prompts, no API calls")
    args = ap.parse_args()

    gold = load_gold()
    g = gold.copy()
    gloss = load_glossary()
    gloss["surname"] = gloss["bare"].map(lambda s: split_surface(s)[2])

    rule, base = shipped_rule_clusters(g)
    risky = risky_groups(g, base, gloss, max_cluster=args.max_cluster)
    n_risky_mentions = sum(len(v) for v in risky.values())
    print(f"risky surname groups: {len(risky)} ({n_risky_mentions} of {len(g)} mentions): "
          f"{', '.join(sorted(risky))}")

    if args.dry_run:
        for key, grp in risky.items():
            prompt, _ = cluster_prompt(key, grp, gloss)
            print(f"\n================ {key} ================\n{prompt[:1500]}")
        return

    api_key = KEY_FILE.read_text(encoding="utf-8").strip()
    OUT_DIR.mkdir(exist_ok=True)

    for run in range(1, args.repeat + 1):
        pred = rule.copy()
        total_cost = 0.0
        raws = {}
        for key, grp in sorted(risky.items()):
            prompt, id_map = cluster_prompt(key, grp, gloss)
            t0 = time.time()
            resp = call_openrouter(api_key, args.model, SYSTEM_PROMPT, prompt)
            raw = resp["choices"][0]["message"]["content"]
            raws[key] = raw
            usage = resp.get("usage") or {}
            cost = float(usage.get("cost") or 0)
            total_cost += cost
            try:
                assign = parse_partition(raw, id_map)
            except json.JSONDecodeError as e:
                print(f"  [{key}] PARSE FAIL ({e}); cluster left unchanged")
                continue
            n_groups = len(set(assign.values()))
            covered = len(assign)
            # unassigned mentions keep the rule cluster
            for mid, gi in assign.items():
                pred.loc[g["mention_id"] == mid] = f"llm::{key}::{gi}"
            print(f"  [{key}] {covered}/{len(grp)} mentions -> {n_groups} group(s) "
                  f"in {time.time()-t0:.1f}s (${cost:.4f})")

        stem = f"{args.model.split('/')[-1]}_sharedsurname_run{run}"
        (OUT_DIR / f"{stem}_raw.json").write_text(
            json.dumps(raws, indent=1, ensure_ascii=False), encoding="utf-8")
        pd.DataFrame({"mention_id": g["mention_id"], "pred_cluster_id": pred.astype(str)}
                     ).to_csv(OUT_DIR / f"{stem}.csv", index=False)

        print(f"\nrun {run}: total LLM cost ${total_cost:.3f}")
        rows = []
        for name, series in (("shipped rule", rule), ("rule + LLM adjudication", pred)):
            for split in ("dev", "test"):
                m = score(gold, series, g["mention_id"], split)
                rows.append({"config": name, "split": split, **m})
        tbl = pd.DataFrame(rows)
        print(tbl.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        tbl.to_csv(OUT_DIR / f"{stem}_summary.csv", index=False)


if __name__ == "__main__":
    main()
