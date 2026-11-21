"""Evaluation harness for clustering + entity linking on the pilot50 gold.

Contract
--------
Every ablation (A.0 baseline, A.1 +engineered, A.2 +LLM, B.1 gazetteer,
B.2 +LLM NIL) produces a CSV of predictions with these columns:

    mention_id           str, matches pilot50_gold.csv
    pred_cluster_id      any hashable; one label per predicted cluster.
                         REQUIRED for clustering metrics.
    pred_nodegoat_id     nodegoat ID or "NIL". Optional; only needed for
                         linking metrics.

The harness left-joins predictions onto the frozen gold, restricts to a
split (default: `test`), and reports metrics per stratum + overall.

Clustering metrics
    pair P/R/F1, B-cubed P/R/F1.
    Convention: NIL gold mentions become singleton gold clusters (we do
    not know whether two unknown persons are the same).

Linking metrics (only if pred_nodegoat_id present)
    link_acc_non_nil : accuracy on mentions with non-NIL gold.
    nil_p/r/f1       : NIL as the positive class.
    Mentions with a missing pred_nodegoat_id are counted as NIL predictions.

Usage (module):
    from eval_clustering_linking import evaluate, load_gold
    metrics = evaluate(load_gold(), pred_df, split="test")

Usage (CLI):
    python eval_clustering_linking.py --pred path/to/predictions.csv --split test
    python eval_clustering_linking.py --pred ... --out results.csv
    python eval_clustering_linking.py --sanity           # smoke test
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
GOLD_CSV = HERE / "pilot50_gold.csv"
NIL = "NIL"


# ---------------------------------------------------------------------------
# Clustering metrics (label-agnostic; NIL semantics applied by caller)
# ---------------------------------------------------------------------------
def _pairs(labels: pd.Series) -> set[frozenset]:
    buckets: dict = defaultdict(list)
    for idx, lab in labels.items():
        buckets[lab].append(idx)
    return {
        frozenset((a, b))
        for members in buckets.values() if len(members) > 1
        for a, b in combinations(members, 2)
    }


def pairwise_prf(gold: pd.Series, pred: pd.Series) -> tuple[float, float, float]:
    g, p = _pairs(gold), _pairs(pred)
    tp = len(g & p)
    prec = tp / len(p) if p else 0.0
    rec  = tp / len(g) if g else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def bcubed_prf(gold: pd.Series, pred: pd.Series) -> tuple[float, float, float]:
    idx = list(gold.index)
    g_cl: dict = defaultdict(set)
    p_cl: dict = defaultdict(set)
    for i in idx:
        g_cl[gold[i]].add(i)
        p_cl[pred[i]].add(i)
    prec_sum = rec_sum = 0.0
    for i in idx:
        gi, pi = g_cl[gold[i]], p_cl[pred[i]]
        inter = len(gi & pi)
        prec_sum += inter / len(pi)
        rec_sum  += inter / len(gi)
    n = len(idx)
    prec = prec_sum / n
    rec  = rec_sum  / n
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


# ---------------------------------------------------------------------------
# Linking metrics
# ---------------------------------------------------------------------------
def linking_metrics(gold_link: pd.Series, pred_link: pd.Series) -> dict:
    df = pd.concat({"g": gold_link, "p": pred_link}, axis=1).dropna(subset=["g"])
    df["p"] = df["p"].fillna(NIL)  # no prediction => treat as NIL

    is_g_nil = df["g"] == NIL
    is_p_nil = df["p"] == NIL

    non_nil = df[~is_g_nil]
    link_acc = float((non_nil["g"] == non_nil["p"]).mean()) if len(non_nil) else float("nan")

    tp = int((is_g_nil & is_p_nil).sum())
    fp = int((~is_g_nil & is_p_nil).sum())
    fn = int((is_g_nil & ~is_p_nil).sum())
    nil_p = tp / (tp + fp) if (tp + fp) else 0.0
    nil_r = tp / (tp + fn) if (tp + fn) else 0.0
    nil_f1 = 2 * nil_p * nil_r / (nil_p + nil_r) if (nil_p + nil_r) else 0.0

    return {
        "link_acc_non_nil": link_acc,
        "nil_p": nil_p,
        "nil_r": nil_r,
        "nil_f1": nil_f1,
        "n_non_nil": int((~is_g_nil).sum()),
        "n_nil":     int(is_g_nil.sum()),
    }


# ---------------------------------------------------------------------------
# Top-level evaluate
# ---------------------------------------------------------------------------
def load_gold(path: Path = GOLD_CSV) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"gold_nodegoat_id": "string"})


def _gold_cluster_series(gold_nid: pd.Series, mention_id: pd.Series) -> pd.Series:
    """NIL gold mentions become their own singleton cluster."""
    return gold_nid.where(gold_nid != NIL, "NIL_" + mention_id.astype(str))


def evaluate(gold_df: pd.DataFrame, pred_df: pd.DataFrame, split: str | None = "test") -> pd.DataFrame:
    if split is not None:
        gold_df = gold_df[gold_df["split"] == split]

    merged = gold_df.merge(pred_df, on="mention_id", how="left")

    # Missing predictions => each becomes its own singleton predicted cluster.
    merged["pred_cluster_id"] = merged["pred_cluster_id"].astype("string")
    miss = merged["pred_cluster_id"].isna()
    if miss.any():
        merged.loc[miss, "pred_cluster_id"] = "__missing_" + merged.loc[miss, "mention_id"].astype(str)

    has_linking = "pred_nodegoat_id" in merged.columns

    rows = []
    strata = ["OVERALL"] + sorted(merged["stratum"].dropna().unique().tolist())
    for stratum in strata:
        sub = merged if stratum == "OVERALL" else merged[merged["stratum"] == stratum]
        if sub.empty:
            continue

        gold_cluster = _gold_cluster_series(sub["gold_nodegoat_id"], sub["mention_id"])
        gold_series = gold_cluster.set_axis(sub["mention_id"].values)
        pred_series = sub["pred_cluster_id"].set_axis(sub["mention_id"].values)

        pp, pr, pf = pairwise_prf(gold_series, pred_series)
        bp, br, bf = bcubed_prf(gold_series, pred_series)
        row = {
            "stratum": stratum,
            "n_mentions": len(sub),
            "n_gold_clusters": gold_series.nunique(),
            "n_pred_clusters": pred_series.nunique(),
            "pair_p": pp, "pair_r": pr, "pair_f1": pf,
            "b3_p":   bp, "b3_r":   br, "b3_f1":   bf,
        }
        if has_linking:
            gl = sub.set_index("mention_id")["gold_nodegoat_id"]
            pl = sub.set_index("mention_id")["pred_nodegoat_id"]
            row.update(linking_metrics(gl, pl))
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sanity smoke test — no external prediction file needed.
# ---------------------------------------------------------------------------
def _sanity(gold_df: pd.DataFrame) -> None:
    print("=" * 60)
    print("SANITY 1: perfect prediction (gold clusters + gold links)")
    print("=" * 60)
    perfect = gold_df[["mention_id", "gold_nodegoat_id"]].rename(
        columns={"gold_nodegoat_id": "pred_cluster_id"}
    )
    # For linking we need a separate column
    perfect["pred_nodegoat_id"] = gold_df["gold_nodegoat_id"].values
    # But for clustering, NIL rows must be singletons in pred (matching gold convention).
    is_nil = perfect["pred_cluster_id"] == NIL
    perfect.loc[is_nil, "pred_cluster_id"] = "NIL_" + perfect.loc[is_nil, "mention_id"].astype(str)
    m = evaluate(gold_df, perfect, split="test")
    print(m.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print()
    print("=" * 60)
    print("SANITY 2: singletons (every mention its own cluster, no links)")
    print("=" * 60)
    singletons = gold_df[["mention_id"]].copy()
    singletons["pred_cluster_id"] = "s_" + singletons["mention_id"].astype(str)
    m = evaluate(gold_df, singletons, split="test")
    print(m.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, help="predictions CSV")
    ap.add_argument("--gold", type=Path, default=GOLD_CSV)
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--out", type=Path, default=None, help="write metrics table to CSV")
    ap.add_argument("--sanity", action="store_true", help="run smoke test on the gold itself")
    args = ap.parse_args()

    gold = load_gold(args.gold)

    if args.sanity:
        _sanity(gold)
        return

    if args.pred is None:
        ap.error("--pred is required unless --sanity is set")

    pred = pd.read_csv(args.pred)
    metrics = evaluate(gold, pred, split=None if args.split == "all" else args.split)
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if args.out:
        metrics.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    _cli()
