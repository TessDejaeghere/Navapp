#!/usr/bin/env python3
"""Person-ID linking for PER mentions, evaluated on the pilot50 gold.

Two stages, per the design under test:

  1. EXACT   - normalise each mention in a cluster and look it up against the
               glossary display name with the life-dates stripped. A single
               unambiguous hit is propagated to every mention in that cluster.
  2. FUZZY   - clusters with no exact hit are scored against every glossary
               entry with a configurable string-similarity method. The score
               for a candidate is the best score over the cluster's mentions;
               below ``--threshold`` the cluster is left as NIL.

Clusters come either from the gold (oracle, isolates linking from clustering
error) or from a predictions CSV with ``mention_id,pred_cluster_id``.
``--clusters none`` treats every mention as its own cluster, which is the
baseline that shows what propagation buys.

Usage (from repo root):
    python CLUSTERING_pipeline/link_persons.py --sweep          # compare methods on dev
    python CLUSTERING_pipeline/link_persons.py --method token_set --threshold 88
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

GOLD_CSV = HERE / "pilot50_gold.csv"
GLOSSARY_CSV = HERE / "glossary" / "PERSON_export_2708.csv"
NIL = "NIL"

# Titles carry no identifying information and would otherwise dominate every
# similarity score against a glossary full of "Given Family" strings.
HONORIFICS = {
    "m", "mr", "mrs", "mme", "mlle", "monsieur", "madame", "mademoiselle",
    "monseigneur", "mons", "messieurs", "docteur", "dr", "professeur",
    "chevalier", "citoyen", "veuve", "vve", "sieur", "le", "la",
}

YEARS_RE = re.compile(r"\s*\([^)]*\)\s*$")

# --- role markers, ported from entity_resolution/dedupe_per_pilot_dedupeio.ipynb
# so the linker reads titles the same way the trained clustering model does.
ROLE_HONORIFICS = {
    "m": "m", "mr": "m", "mons": "m", "monsieur": "m",
    "mme": "mme", "madame": "mme", "veuve": "mme", "vve": "mme",
    "mlle": "mlle", "mademoiselle": "mlle",
    "dr": "dr", "docteur": "dr",
    "abbe": "abbe", "pere": "pere", "frere": "frere",
    "baron": "baron", "baronne": "baronne",
    "comte": "comte", "comtesse": "comtesse",
    "sig": "m", "signor": "m", "signore": "m",
    "signora": "mme", "signorina": "mlle",
    "don": "m", "donna": "mme",
    "dottor": "dr", "dottore": "dr",
    "padre": "pere", "fratello": "frere", "abate": "abbe",
    "conte": "comte", "contessa": "comtesse",
    "barone": "baron", "baronessa": "baronne",
    "marchese": "marchese", "marchesa": "marchesa",
    "mrs": "mme", "mistress": "mme",
    "mmes": "mme", "mesdames": "mme",
    "mlles": "mlle", "mesdemoiselles": "mlle",
    "mm": "m", "messieurs": "m",
}
PARTICLES = {"de", "du", "des", "d", "la", "le", "von", "van", "der", "den"}
TRAILING_ROLES = {"fils", "pere", "aine", "jeune", "cadet", "junior", "senior"}
ROLE_CONFLICTS = [{"pere", "fils"}, {"baron", "baronne"},
                  {"comte", "comtesse"}, {"m", "mme", "mlle"},
                  {"aine", "jeune"}, {"junior", "senior"},
                  {"marchese", "marchesa"}]
FEMALE_MARKERS = {"mme", "mlle", "baronne", "comtesse", "marchesa"}
MALE_MARKERS = {"m", "baron", "comte", "marchese", "pere", "frere", "abbe"}

_AGGLUTINABLE = ("mademoiselle", "monsieur", "docteur", "madame",
                 "signore", "signora", "signorina", "dottore",
                 "contessa", "baronessa", "marchese", "marchesa")

# "M.lle" / "M.me" / "M.r" must be glued before punctuation is stripped,
# otherwise the leading "m" alone reads as Monsieur. Ported verbatim from the
# dedupe notebook, where dropping it flips the gender of abbreviated titles.
_HONORIFIC_DOT_RE = re.compile(r"\bm\.(r|me|mes|rs|lle|lles)\b")

# Seed lexicon for candidates that carry no title. 19c French/Italian given
# names frequent in this correspondence; ambiguous ones are deliberately absent.
MALE_GIVEN = {
    "jean", "joseph", "pierre", "louis", "charles", "francois", "antoine",
    "henri", "guillaume", "adolphe", "alexandre", "auguste", "emile", "eugene",
    "gustave", "jules", "leopold", "victor", "theodore", "denis", "andre",
    "jacques", "paul", "philippe", "edouard", "ferdinand", "felix", "albert",
    "alfred", "achille", "hippolyte", "camille", "constant", "nicolas",
    "mathias", "michel", "etienne", "laurent", "vincent", "bernard", "martin",
    "raphael", "giuseppe", "antonio", "giovanni", "carlo", "luigi", "pietro",
    "johann", "friedrich", "wilhelm", "abraham", "isidore", "prosper", "nestor",
}
FEMALE_GIVEN = {
    "marie", "sophie", "adele", "caroline", "sylvie", "silvie", "charlotte",
    "louise", "anne", "jeanne", "elisabeth", "henriette", "pauline", "julie",
    "clementine", "virginie", "josephine", "therese", "catherine", "madeleine",
    "rosalie", "eugenie", "amelie", "victoire", "adelaide", "constance",
    "emilie", "flore", "fanny", "maria", "teresa", "angelica", "marianne",
    "hortense", "celine", "octavie", "zoe", "leonie", "mathilde",
}


def _peel_leading_honorific(tok: str) -> str | None:
    if tok in ROLE_HONORIFICS:
        return ROLE_HONORIFICS[tok]
    for h in _AGGLUTINABLE:
        if tok.endswith(h) and 1 <= len(tok) - len(h) <= 2:
            return ROLE_HONORIFICS[h]
    return None


def split_surface(s: str) -> tuple[str, str, str, str]:
    """Return (honorific, given, surname, trailing_role) from a normalised name."""
    glued = _HONORIFIC_DOT_RE.sub(r"m\1", (s or "").lower())
    toks = [t for t in norm(glued).split() if t]
    honorific = role = "none"
    while toks:
        peeled = _peel_leading_honorific(toks[0])
        if peeled is None:
            break
        honorific = peeled
        toks.pop(0)
    while toks and toks[-1] in TRAILING_ROLES:
        role = toks.pop()
    if not toks:
        return honorific, "", "", role
    surname = [toks.pop()]
    while toks and toks[-1] in PARTICLES:
        surname.insert(0, toks.pop())
    return honorific, " ".join(toks), " ".join(surname), role


def role_signature(s: str) -> set[str]:
    honorific, _given, _surname, role = split_surface(s)
    return {honorific, role} - {"none"}


def surname_metaphone(s: str, min_len: int = 4) -> str:
    """Metaphone code of the surname anchor, or '' when too short to trust.

    Same length gate as the dedupe model's ``surname_metaphone`` variable:
    short codes (NFS for Navez) collide across unrelated names.
    """
    import jellyfish

    surname = split_surface(s)[2]
    if len(surname) < min_len:
        return ""
    code = jellyfish.metaphone(surname)
    return code if len(code) >= min_len else ""


def gender_of(s: str, maiden: str = "") -> str:
    """'F', 'M' or '?'. Conflicting title and given name abstain rather than guess:
    "M. adele" is more often an abbreviated Mlle or an HTR slip than a man."""
    honorific, given, _surname, _role = split_surface(s)
    by_title = "F" if honorific in FEMALE_MARKERS else "M" if honorific in MALE_MARKERS else "?"

    tokens = set(given.split())
    by_given = "?"
    if maiden.strip():
        by_given = "F"
    elif tokens & MALE_GIVEN and not tokens & FEMALE_GIVEN:
        by_given = "M"
    elif tokens & FEMALE_GIVEN and not tokens & MALE_GIVEN:
        by_given = "F"

    if by_title != "?" and by_given != "?" and by_title != by_given:
        return "?"
    return by_title if by_title != "?" else by_given


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^\w\s'-]", " ", s)
    return " ".join(s.split())


def strip_honorifics(s: str) -> str:
    tokens = norm(s).split()
    while tokens and tokens[0].strip(".") in HONORIFICS:
        tokens.pop(0)
    return " ".join(tokens)


def strip_years(s: str) -> str:
    return YEARS_RE.sub("", s or "").strip()


# ---------------------------------------------------------------------------
# Similarity methods under evaluation
# ---------------------------------------------------------------------------
def lcs_ratio(a: str, b: str) -> float:
    """Longest common substring, normalised by the shorter string."""
    if not a or not b:
        return 0.0
    match = SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
    return 100.0 * match.size / min(len(a), len(b))


def surname_anchored(a: str, b: str, family: str = "") -> float:
    """Reward an exact surname token match, then break ties on full similarity.

    The surname is the identifying part of a 19c mention ("Mr Quetelet"); the
    given name is usually absent, so whole-string scorers under-rank the right
    entry.
    """
    if not family:
        return 0.0
    fam = norm(family)
    tokens = a.split()
    if not fam or not tokens:
        return 0.0
    if fam in tokens or any(fuzz.ratio(t, fam) >= 90 for t in tokens):
        return 80.0 + 0.2 * fuzz.token_set_ratio(a, b)
    return 0.4 * fuzz.token_set_ratio(a, b)


METHODS = {
    "levenshtein": lambda a, b, fam: fuzz.ratio(a, b),
    "partial": lambda a, b, fam: fuzz.partial_ratio(a, b),
    "token_set": lambda a, b, fam: fuzz.token_set_ratio(a, b),
    "token_sort": lambda a, b, fam: fuzz.token_sort_ratio(a, b),
    "wratio": lambda a, b, fam: fuzz.WRatio(a, b),
    "lcs": lambda a, b, fam: lcs_ratio(a, b),
    "surname": surname_anchored,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_glossary() -> pd.DataFrame:
    g = pd.read_csv(GLOSSARY_CSV, dtype=str).fillna("")
    g = g[g["nodegoat ID"].str.strip() != ""].copy()
    g["bare"] = g["Name"].map(strip_years)
    g["key"] = g["bare"].map(norm)
    g["probe"] = g["bare"].map(strip_honorifics)
    g["family"] = g["family name"].fillna("")
    g["born"] = [_life_year(c, n, 0) for c, n in zip(g["[date of birth] Date Start"], g["Name"])]
    g["died"] = [_life_year(c, n, 1) for c, n in zip(g["[date of death] Date Start"], g["Name"])]
    g["gender"] = [gender_of(b, m) for b, m in zip(g["bare"], g["maiden name"])]
    g["roles"] = [role_signature(b) for b in g["bare"]]
    g["surname_meta"] = [surname_metaphone(b) for b in g["bare"]]
    g["n_name_tokens"] = [len(split_surface(b)[1].split()) for b in g["bare"]]
    g = g[g["key"].str.strip() != ""].drop_duplicates(subset=["nodegoat ID"])
    return g.reset_index(drop=True)


def _life_year(col_value: str, name: str, which: int) -> float:
    """Birth (which=0) or death (which=1) year, from the date column or the
    ``(1787-1869)`` suffix on the display name."""
    m = re.search(r"\b(1[5-9]\d{2})\b", col_value or "")
    if m:
        return float(m.group(1))
    m = re.search(r"\((1[5-9]\d{2})\s*[-–]\s*(1[5-9]\d{2})?\)", name or "")
    if m and m.group(which + 1):
        return float(m.group(which + 1))
    return float("nan")


def load_corpus_prior() -> dict[str, int]:
    """How often each glossary person appears as sender or recipient across the
    whole correspondence. Derived from letter metadata, never from the gold."""
    gloss = pd.read_csv(GLOSSARY_CSV, dtype=str).fillna("")
    name2id = {n.strip(): i for n, i in zip(gloss["Name"], gloss["nodegoat ID"]) if n.strip()}
    counts: dict[str, int] = {}
    ov_path = REPO / "data" / "letters_overview.csv"
    if not ov_path.exists():
        return counts
    ov = pd.read_csv(ov_path, encoding="utf-8-sig", dtype=str).fillna("")
    for col in ("Source", "Target"):
        for value in ov[col].str.strip():
            ngid = name2id.get(value)
            if ngid:
                counts[ngid] = counts.get(ngid, 0) + 1
    return counts


def load_gold() -> pd.DataFrame:
    g = pd.read_csv(GOLD_CSV, dtype=str).fillna("")
    g["gold_nodegoat_id"] = g["gold_nodegoat_id"].replace("", NIL)
    return g


def assign_clusters(gold: pd.DataFrame, mode: str, pred_csv: Path | None) -> pd.Series:
    """Return a cluster label per row of `gold`."""
    if mode == "none":
        return gold["mention_id"]
    if mode == "gold":
        # NIL mentions are singletons: we do not know that two unknown people
        # are the same person.
        return [
            gid if gid != NIL else f"NIL::{mid}"
            for gid, mid in zip(gold["gold_nodegoat_id"], gold["mention_id"])
        ]
    if mode == "pred":
        if pred_csv is None:
            raise SystemExit("--clusters pred requires --cluster-csv")
        p = pd.read_csv(pred_csv, dtype=str).fillna("")
        m = dict(zip(p["mention_id"], p["pred_cluster_id"]))
        return [m.get(mid, f"UNCLUSTERED::{mid}") for mid in gold["mention_id"]]
    raise SystemExit(f"unknown --clusters mode: {mode}")


# ---------------------------------------------------------------------------
# Linking
# ---------------------------------------------------------------------------
def cluster_candidates(gold: pd.DataFrame, gloss: pd.DataFrame, clusters,
                       method: str, tiebreak: set[str] | None = None,
                       prior: dict[str, int] | None = None,
                       topk: int = 60, gender_guard: bool = False,
                       bare_penalty: float = 0.0,
                       phonetic_floor: float = 0.0) -> pd.DataFrame:
    """Rank glossary entries per cluster, before any NIL decision is taken.

    Returns one row per cluster with the exact-match verdict and the top two
    fuzzy scores, so a NIL rule can be swept without re-scoring. String score
    alone leaves large ties (any bare surname scores 100 against every glossary
    entry sharing it), so `tiebreak` reorders equal-scoring candidates using
    letter-level evidence.

    `gender_guard` drops candidates whose role marker conflicts with the
    mention's title (Mme X -> a male entry). `bare_penalty` docks the score when
    the only evidence is a surname and the candidate carries given names, which
    is the case that saturates the token-set metric. `phonetic_floor` lifts the
    score of candidates whose surname metaphone matches, which is what rescues
    HTR spellings the edit-distance metrics miss (Otevare / Odevaere).
    """
    tiebreak = tiebreak or set()
    prior = prior or {}

    exact_index: dict[str, set[str]] = {}
    for key, ngid in zip(gloss["key"], gloss["nodegoat ID"]):
        exact_index.setdefault(key, set()).add(ngid)

    scorer = METHODS[method]
    probes = gloss["probe"].tolist()
    fams = gloss["family"].tolist()
    ids = gloss["nodegoat ID"].tolist()
    born = gloss["born"].tolist()
    died = gloss["died"].tolist()
    genders = gloss["gender"].tolist()
    roles = gloss["roles"].tolist()
    metas = gloss["surname_meta"].tolist()
    n_given = gloss["n_name_tokens"].tolist()

    name2id = {n.strip(): i for n, i in zip(gloss["Name"], gloss["nodegoat ID"]) if n.strip()}

    df = gold.copy()
    df["cluster"] = list(clusters)
    df["surface_probe"] = df["surface"].map(strip_honorifics)

    cache: dict[str, list[tuple[int, float]]] = {}
    rows = []
    for cid, block in df.groupby("cluster", sort=False):
        surfaces = [s for s in block["surface_probe"].unique() if s]
        raw_surfaces = [s for s in block["surface"].unique() if str(s).strip()]

        hits = {i for s in surfaces for i in exact_index.get(s, set())}
        if len(hits) == 1:
            hit = hits.pop()
            rows.append({"cluster": cid, "cand_id": hit, "exact": True,
                         "top1": 100.0, "top2": 0.0, "margin": 100.0,
                         "alts": [(hit, 100.0)]})
            continue

        years = [float(y) for y in block["year"] if str(y).strip().isdigit()]
        parties = {name2id.get(v.strip()) for col in ("sender", "recipient") for v in block[col]}
        parties.discard(None)

        # Mention-side role evidence, pooled over the cluster. A cluster whose
        # own titles already conflict tells us nothing, so it abstains.
        m_roles: set[str] = set()
        for s in raw_surfaces:
            m_roles |= role_signature(s)
        if not _roles_ok(m_roles):
            m_roles = set()
        m_gender = "?"
        seen = {gender_of(s) for s in raw_surfaces} - {"?"}
        if len(seen) == 1:
            m_gender = seen.pop()
        bare_probe = all(len(s.split()) <= 1 for s in surfaces)
        m_metas = {m for m in (surname_metaphone(s) for s in raw_surfaces) if m}

        best: dict[int, float] = {}
        for s in surfaces:
            if s not in cache:
                sc = [scorer(s, p, f) for p, f in zip(probes, fams)]
                order = sorted(range(len(sc)), key=sc.__getitem__, reverse=True)[:topk]
                cache[s] = [(j, sc[j]) for j in order]
            for j, v in cache[s]:
                if v > best.get(j, -1.0):
                    best[j] = v

        # Phonetic candidates never enter the top-k on string score alone.
        if phonetic_floor and m_metas:
            for j, code in enumerate(metas):
                if code and code in m_metas:
                    best[j] = max(best.get(j, 0.0), phonetic_floor)

        ranked = []
        for j, base in best.items():
            if "dates" in tiebreak and years and born[j] == born[j] and max(years) < born[j]:
                continue  # cannot be mentioned before being born
            if gender_guard:
                if m_gender != "?" and genders[j] != "?" and m_gender != genders[j]:
                    continue
                if not _roles_ok(m_roles | roles[j]):
                    continue
            if bare_penalty and bare_probe and n_given[j] >= 1:
                base -= bare_penalty
            ranked.append((base, _date_rank(j, years, born, died) if "dates" in tiebreak else 0,
                           1 if ("context" in tiebreak and ids[j] in parties) else 0,
                           prior.get(ids[j], 0) if "prior" in tiebreak else 0,
                           j))
        if not ranked:
            rows.append({"cluster": cid, "cand_id": NIL, "exact": False,
                         "top1": -1.0, "top2": -1.0, "margin": 0.0, "alts": []})
            continue

        ranked.sort(key=lambda t: (t[0], t[2], t[1], t[3]), reverse=True)
        top1 = ranked[0][0]
        top2 = ranked[1][0] if len(ranked) > 1 else 0.0
        rows.append({"cluster": cid, "cand_id": ids[ranked[0][4]], "exact": False,
                     "top1": top1, "top2": top2, "margin": top1 - top2,
                     "alts": [(ids[t[4]], t[0]) for t in ranked[:3]]})

    return pd.DataFrame(rows)


def _roles_ok(sig: set[str]) -> bool:
    sig = sig - {"none"}
    return all(len(sig & conflict) <= 1 for conflict in ROLE_CONFLICTS)


def _date_rank(j: int, years: list[float], born: list[float], died: list[float]) -> int:
    """2 = alive at the letter date, 1 = dates unknown, 0 = posthumous mention.

    Posthumous is demoted rather than excluded: dead artists are discussed
    constantly in this correspondence.
    """
    b, d = born[j], died[j]
    if b != b and d != d:
        return 1
    if not years:
        return 1
    for y in years:
        if (b != b or y >= b) and (d != d or y <= d):
            return 2
    return 0


def apply_nil_rule(gold: pd.DataFrame, clusters, cands: pd.DataFrame,
                   threshold: float, min_margin: float = 0.0) -> pd.DataFrame:
    """Turn ranked candidates into predictions, sending weak ones to NIL."""
    pred, stage = {}, {}
    for r in cands.itertuples():
        if r.exact:
            pred[r.cluster], stage[r.cluster] = r.cand_id, "exact"
        elif r.top1 >= threshold and r.margin >= min_margin:
            pred[r.cluster], stage[r.cluster] = r.cand_id, "fuzzy"
        else:
            pred[r.cluster], stage[r.cluster] = NIL, "nil"

    df = gold.copy()
    df["cluster"] = list(clusters)
    df["pred_nodegoat_id"] = df["cluster"].map(pred)
    df["pred_cluster_id"] = df["cluster"]
    df["stage"] = df["cluster"].map(stage)
    return df


def link(gold: pd.DataFrame, gloss: pd.DataFrame, clusters,
         method: str, threshold: float, min_margin: float = 0.0,
         tiebreak: set[str] | None = None, prior: dict[str, int] | None = None,
         gender_guard: bool = False, bare_penalty: float = 0.0,
         phonetic_floor: float = 0.0) -> pd.DataFrame:
    cands = cluster_candidates(gold, gloss, clusters, method, tiebreak, prior,
                               gender_guard=gender_guard, bare_penalty=bare_penalty,
                               phonetic_floor=phonetic_floor)
    return apply_nil_rule(gold, clusters, cands, threshold, min_margin)


def score(df: pd.DataFrame, split: str | None = "test") -> dict:
    d = df if split is None else df[df["split"] == split]
    non_nil = d[d["gold_nodegoat_id"] != NIL]
    nil = d[d["gold_nodegoat_id"] == NIL]
    acc = float((non_nil["gold_nodegoat_id"] == non_nil["pred_nodegoat_id"]).mean()) if len(non_nil) else 0.0

    tp = int(((d["gold_nodegoat_id"] == NIL) & (d["pred_nodegoat_id"] == NIL)).sum())
    fp = int(((d["gold_nodegoat_id"] != NIL) & (d["pred_nodegoat_id"] == NIL)).sum())
    fn = int(((d["gold_nodegoat_id"] == NIL) & (d["pred_nodegoat_id"] != NIL)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(d),
        "n_non_nil": len(non_nil),
        "n_nil": len(nil),
        "link_acc": acc,
        "nil_p": p,
        "nil_r": r,
        "nil_f1": 2 * p * r / (p + r) if p + r else 0.0,
        "overall_acc": float((d["gold_nodegoat_id"] == d["pred_nodegoat_id"]).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="surname", choices=sorted(METHODS))
    ap.add_argument("--threshold", type=float, default=85.0)
    ap.add_argument("--min-margin", type=float, default=0.0,
                    help="require top1-top2 >= this, else NIL")
    ap.add_argument("--clusters", default="gold", choices=["gold", "none", "pred"])
    ap.add_argument("--cluster-csv", type=Path, default=None)
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--sweep", action="store_true",
                    help="grid over methods x thresholds on dev, then report the winner on test")
    ap.add_argument("--nil-sweep", action="store_true",
                    help="sweep the NIL decision (score threshold x margin) for one method")
    ap.add_argument("--tiebreak", default="",
                    help="comma list of tie-breakers: dates,prior,context (or 'all')")
    ap.add_argument("--gender-guard", action="store_true",
                    help="drop candidates whose role marker conflicts with the mention title")
    ap.add_argument("--bare-penalty", type=float, default=0.0,
                    help="score penalty when a surname-only probe matches a full name")
    ap.add_argument("--phonetic-floor", type=float, default=0.0,
                    help="score floor for candidates whose surname metaphone matches")
    ap.add_argument("--ablate-phonetic", action="store_true",
                    help="sweep the phonetic floor on dev and test")
    ap.add_argument("--ablate-guards", action="store_true",
                    help="evaluate gender guard x bare-token penalty on dev and test")
    ap.add_argument("--ablate-tiebreak", action="store_true",
                    help="evaluate every tie-breaker combination on dev and test")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    tiebreak = ({"dates", "prior", "context"} if args.tiebreak == "all"
                else {t.strip() for t in args.tiebreak.split(",") if t.strip()})

    gold = load_gold()
    gloss = load_glossary()
    clusters = assign_clusters(gold, args.clusters, args.cluster_csv)
    print(f"gold: {len(gold)} mentions, {gold['gold_nodegoat_id'].nunique()} distinct ids "
          f"({(gold['gold_nodegoat_id'] == NIL).sum()} NIL)")
    print(f"glossary: {len(gloss)} persons")
    print(f"clusters ({args.clusters}): {len(set(clusters))}\n")

    if args.sweep:
        prior = load_corpus_prior()
        rows = []
        for method in sorted(METHODS):
            for thr in (0, 60, 70, 75, 80, 85, 88, 90, 95):
                d = link(gold, gloss, clusters, method, thr, 0.0, tiebreak, prior)
                s = score(d, "dev")
                rows.append({"method": method, "threshold": thr, **s})
        res = pd.DataFrame(rows)
        best = res.sort_values("overall_acc", ascending=False).iloc[0]
        print("=== dev sweep (top 12 by overall accuracy) ===")
        cols = ["method", "threshold", "link_acc", "nil_f1", "overall_acc"]
        print(res.sort_values("overall_acc", ascending=False).head(12)[cols].to_string(index=False))
        print(f"\n=== best on dev: {best['method']} @ {best['threshold']:g} -> test ===")
        d = link(gold, gloss, clusters, best["method"], best["threshold"], 0.0, tiebreak, prior)
        for k, v in score(d, "test").items():
            print(f"  {k:14s} {v:.4f}" if isinstance(v, float) else f"  {k:14s} {v}")
        if args.out:
            res.to_csv(args.out, index=False)
            print(f"\nWrote sweep table to {args.out}")
        return 0

    if args.ablate_phonetic:
        prior = load_corpus_prior()
        rows = []
        for floor in (0, 85, 90, 92, 95, 100):
            cands = cluster_candidates(gold, gloss, clusters, args.method, tiebreak, prior,
                                       gender_guard=args.gender_guard,
                                       bare_penalty=args.bare_penalty, phonetic_floor=floor)
            for split in ("dev", "test"):
                d = apply_nil_rule(gold, clusters, cands, args.threshold, args.min_margin)
                rows.append({"phonetic_floor": floor, "split": split, **score(d, split)})
        res = pd.DataFrame(rows)
        cols = ["phonetic_floor", "link_acc", "nil_p", "nil_r", "nil_f1", "overall_acc"]
        for split in ("dev", "test"):
            print(f"=== {split} (method={args.method} @ {args.threshold:g}, "
                  f"gender_guard={args.gender_guard}) ===")
            print(res[res["split"] == split][cols].to_string(index=False))
            print()
        if args.out:
            res.to_csv(args.out, index=False)
            print(f"Wrote phonetic ablation to {args.out}")
        return 0

    if args.ablate_guards:
        prior = load_corpus_prior()
        rows = []
        for guard in (False, True):
            for penalty in (0, 3, 6, 11, 16):
                cands = cluster_candidates(gold, gloss, clusters, args.method, tiebreak,
                                           prior, gender_guard=guard, bare_penalty=penalty)
                for split in ("dev", "test"):
                    d = apply_nil_rule(gold, clusters, cands, args.threshold, args.min_margin)
                    rows.append({"gender_guard": guard, "bare_penalty": penalty,
                                 "split": split, **score(d, split)})
        res = pd.DataFrame(rows)
        cols = ["gender_guard", "bare_penalty", "link_acc", "nil_p", "nil_f1", "overall_acc"]
        for split in ("dev", "test"):
            print(f"=== {split} (method={args.method} @ {args.threshold:g}, "
                  f"tiebreak={'+'.join(sorted(tiebreak)) or 'none'}) ===")
            print(res[res["split"] == split][cols].to_string(index=False))
            print()
        if args.out:
            res.to_csv(args.out, index=False)
            print(f"Wrote guard ablation to {args.out}")
        return 0

    if args.ablate_tiebreak:
        from itertools import combinations
        prior = load_corpus_prior()
        print(f"corpus prior: {len(prior)} persons with letter-metadata counts\n")
        combos = [set()]
        for r in (1, 2, 3):
            combos += [set(c) for c in combinations(["dates", "prior", "context"], r)]
        rows = []
        for combo in combos:
            cands = cluster_candidates(gold, gloss, clusters, args.method, combo, prior)
            for split in ("dev", "test"):
                d = apply_nil_rule(gold, clusters, cands, args.threshold, args.min_margin)
                rows.append({"tiebreak": "+".join(sorted(combo)) or "none",
                             "split": split, **score(d, split)})
        res = pd.DataFrame(rows)
        cols = ["tiebreak", "link_acc", "nil_f1", "overall_acc"]
        for split in ("dev", "test"):
            print(f"=== {split} (method={args.method} @ {args.threshold:g}) ===")
            sub = res[res["split"] == split].sort_values("link_acc", ascending=False)
            print(sub[cols].to_string(index=False))
            print()
        if args.out:
            res.to_csv(args.out, index=False)
            print(f"Wrote ablation table to {args.out}")
        return 0

    if args.nil_sweep:
        cands = cluster_candidates(gold, gloss, clusters, args.method, tiebreak,
                                   load_corpus_prior(), gender_guard=args.gender_guard,
                                   bare_penalty=args.bare_penalty,
                                   phonetic_floor=args.phonetic_floor)
        rows = []
        for thr in range(50, 101, 5):
            for margin in (0, 1, 2, 5, 10, 20):
                d = apply_nil_rule(gold, clusters, cands, thr, margin)
                rows.append({"threshold": thr, "min_margin": margin, **score(d, "dev")})
        res = pd.DataFrame(rows)
        cols = ["threshold", "min_margin", "link_acc", "nil_p", "nil_r", "nil_f1", "overall_acc"]
        print(f"=== NIL sweep on dev, method={args.method} ===")
        print("top 8 by NIL F1:")
        print(res.sort_values("nil_f1", ascending=False).head(8)[cols].to_string(index=False))
        print("\ntop 8 by overall accuracy:")
        print(res.sort_values("overall_acc", ascending=False).head(8)[cols].to_string(index=False))
        best = res.sort_values("overall_acc", ascending=False).iloc[0]
        print(f"\n=== best dev point: threshold={best['threshold']:g} "
              f"margin={best['min_margin']:g} -> test ===")
        d = apply_nil_rule(gold, clusters, cands, best["threshold"], best["min_margin"])
        for k, v in score(d, "test").items():
            print(f"  {k:14s} {v:.4f}" if isinstance(v, float) else f"  {k:14s} {v}")
        if args.out:
            res.to_csv(args.out, index=False)
            print(f"\nWrote NIL sweep table to {args.out}")
        return 0

    d = link(gold, gloss, clusters, args.method, args.threshold, args.min_margin,
             tiebreak, load_corpus_prior(), args.gender_guard, args.bare_penalty,
             args.phonetic_floor)
    split = None if args.split == "all" else args.split
    print(f"=== {args.method} @ {args.threshold:g}, split={args.split} ===")
    for k, v in score(d, split).items():
        print(f"  {k:14s} {v:.4f}" if isinstance(v, float) else f"  {k:14s} {v}")
    print(f"\n  stage mix: {d['stage'].value_counts().to_dict()}")

    if args.out:
        d[["mention_id", "pred_cluster_id", "pred_nodegoat_id", "stage"]].to_csv(args.out, index=False)
        print(f"\nWrote predictions to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
