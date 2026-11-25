#!/usr/bin/env python3
"""
Train a small CNN that tells whether handwriting is upright, or turned
90 / 180 / 270 degrees. Labels come for free: we take our own scans (which
are upright) and rotate them.

  1. `prepare`: cut ink-rich square patches out of every scan (after a random
     skew of up to +-6 degrees, so the model also copes with crooked pages) and
     save them to an .npz. Pages are split into train/val/test *by writer*
     (sender in the Nodegoat export), stratified by how much each writer
     wrote, so the test set only contains hands the model never saw.
  2. `train`: each patch is turned by a random multiple of 90 degrees; the CNN
     learns to predict that turn. Saves the weights to models/orientation_cnn.pt.

A page is classified by letting ~24 patches vote (sum of log-probabilities);
see `orient_pages.classifier_rotation`.

Usage:
    python HTR_pipeline/train_orientation_classifier.py prepare --out data/orientation_patches.npz \
        --scans data/tomes_KBR data/_pages_ocmw_batch2 data/_pages_batch2_kbr_iiif
    python HTR_pipeline/train_orientation_classifier.py train --data data/orientation_patches.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import orient_pages as op  # noqa: E402

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "orientation_cnn.pt"
PATCH = op.CLF_PATCH
PATCHES_PER_PAGE = 8


def letter_id(path: Path) -> str:
    parts = path.stem.split("_")
    return parts[1] if len(parts) > 1 else path.stem


def senders_by_letter(nodegoat_csv: Path) -> dict[str, str]:
    """KBR letter ID (= nodegoat '[Archive] IDN') -> sender ('Source')."""
    raw = nodegoat_csv.read_bytes()
    try:
        txt = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        txt = raw.decode("cp1252")
    rows = list(csv.reader(txt.splitlines()))
    src_i, idn_i = rows[0].index("Source"), rows[0].index("[Archive] IDN")
    return {r[idn_i].strip(): r[src_i].strip() for r in rows[1:]
            if len(r) > idn_i and r[idn_i].strip() and r[src_i].strip()}


def writer_split(files: list[Path], senders: dict[str, str], seed: int = 0,
                 test_frac: float = 0.20, val_frac: float = 0.10) -> dict[str, str]:
    """Assign whole *writers* (senders) to train / val / test, so the test set only
    has hands the model never saw. Stratified by writer size (pages): large
    (>= 40), medium (5-39), small (< 5); within each stratum about test_frac of
    the pages goes to test and val_frac to val. Letters without a known sender
    always go to train (their hand might match a test writer)."""
    pages_per = defaultdict(int)
    for f in files:
        s = senders.get(letter_id(f))
        if s:
            pages_per[s] += 1
    strata = {"large": [], "medium": [], "small": []}
    for s, n in pages_per.items():
        strata["large" if n >= 40 else "medium" if n >= 5 else "small"].append(s)
    rng = np.random.default_rng(seed)
    assign: dict[str, str] = {}
    for name, ws in strata.items():
        ws = sorted(ws)
        rng.shuffle(ws)
        total = sum(pages_per[w] for w in ws)
        t = v = 0
        for w in ws:
            if t < test_frac * total:
                assign[w] = "test"; t += pages_per[w]
            elif v < val_frac * total:
                assign[w] = "val"; v += pages_per[w]
            else:
                assign[w] = "train"
        print(f"  stratum {name}: {len(ws)} writers, {total} pages -> test {t}, val {v}")
    return {f.name: assign.get(senders.get(letter_id(f), ""), "train") for f in files}


def _prepare_one(args):
    path, seed = args
    rng = np.random.default_rng(seed)
    try:
        with Image.open(path) as im:
            im = im.convert("L")
            im.thumbnail((op.CLF_DIM * 2, op.CLF_DIM * 2))
            sk = float(rng.uniform(-6, 6))
            im = im.rotate(sk, resample=Image.BILINEAR, expand=True, fillcolor=int(np.median(np.asarray(im))))
            patches = op.sample_patches(im, PATCHES_PER_PAGE, rng)
    except Exception as e:  # unreadable scan: skip
        return path, None, str(e)
    return path, patches, ""


def prepare(args) -> None:
    files, seen = [], set()
    for d in args.scans:
        for p in sorted(Path(d).rglob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg"} and p.name not in seen:
                seen.add(p.name)              # same scan sometimes sits in two batch folders
                files.append(p)
    print(f"{len(files)} unique scans")
    senders = senders_by_letter(args.nodegoat)
    splits = writer_split(files, senders)
    split_file = args.out.with_suffix(".splits.json")
    split_file.write_text(json.dumps(
        {str(f): {"split": splits[f.name], "letter": letter_id(f),
                  "writer": senders.get(letter_id(f), "")} for f in files},
        ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"split per page (with writer) -> {split_file}")
    X, split, page_idx, names = [], [], [], []
    t = time.time()
    with Pool(args.workers) as pool:
        for i, (p, patches, err) in enumerate(pool.imap_unordered(
                _prepare_one, [(f, k) for k, f in enumerate(files)], chunksize=4)):
            if patches is None or len(patches) == 0:
                continue                      # blank page or unreadable
            s = splits[p.name]
            X.append(patches)
            split += [s] * len(patches)
            page_idx += [len(names)] * len(patches)
            names.append(p.name)
            if i % 250 == 0:
                print(f"  {i}/{len(files)} pages, {time.time() - t:.0f}s", flush=True)
    X = np.concatenate(X)
    np.savez_compressed(args.out, X=X, split=np.array(split), page=np.array(page_idx), names=np.array(names))
    print(f"saved {X.shape} patches from {len(names)} pages to {args.out}")
    for s in ("train", "val", "test"):
        print(f"  {s}: {sum(1 for x in split if x == s)} patches")


def build_model():
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
                             nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
                             nn.MaxPool2d(2))
    return nn.Sequential(block(1, 24), block(24, 48), block(48, 96), block(96, 128),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, 4))


def train(args) -> None:
    import torch
    import torch.nn.functional as F
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    d = np.load(args.data)
    X, split, page = d["X"], d["split"], d["page"]
    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    print(f"train {tr.sum()}  val {va.sum()}  test {te.sum()} patches")
    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    Xtr = X[tr]
    steps = args.epochs * (len(Xtr) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    rng = np.random.default_rng(0)

    def batch_tensor(patches, ks):
        b = np.stack([np.rot90(p, k) for p, k in zip(patches, ks)])
        return op.patches_to_tensor(b)

    def evaluate(mask):
        """Patch accuracy and page accuracy (all 4 turns of every page)."""
        model.eval()
        Xs, pg = X[mask], page[mask]
        ok_patch = tot = 0
        page_ok = page_tot = 0
        with torch.no_grad():
            for k in range(4):
                logp = []
                for i in range(0, len(Xs), 512):
                    xb = batch_tensor(Xs[i:i + 512], [k] * len(Xs[i:i + 512]))
                    logp.append(F.log_softmax(model(xb), 1).numpy())
                logp = np.concatenate(logp)
                ok_patch += int((logp.argmax(1) == k).sum()); tot += len(logp)
                for p in np.unique(pg):
                    page_ok += int(logp[pg == p].sum(0).argmax() == k); page_tot += 1
        model.train()
        return ok_patch / tot, page_ok / page_tot, page_tot

    t = time.time()
    for ep in range(args.epochs):
        perm = rng.permutation(len(Xtr))
        loss_sum = 0.0
        for i in range(0, len(perm) - args.batch + 1, args.batch):
            idx = perm[i:i + args.batch]
            ks = rng.integers(0, 4, len(idx))
            xb = batch_tensor(Xtr[idx], ks)
            # light photometric augmentation: contrast / brightness jitter
            xb = xb * torch.empty(len(idx), 1, 1, 1).uniform_(0.7, 1.3) + torch.empty(len(idx), 1, 1, 1).uniform_(-0.3, 0.3)
            loss = F.cross_entropy(model(xb), torch.as_tensor(ks))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            loss_sum += loss.item() * len(idx)
        pa, pga, n = evaluate(va)
        print(f"epoch {ep + 1}/{args.epochs}  loss {loss_sum / len(perm):.3f}  "
              f"val patch acc {pa:.3f}  val page acc {pga:.3f} ({n} page-turns)  {time.time() - t:.0f}s", flush=True)
    pa, pga, n = evaluate(te)
    print(f"TEST patch acc {pa:.3f}  page acc {pga:.3f} ({n} page-turns)")
    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    meta = {"patch": PATCH, "dim": op.CLF_DIM, "epochs": args.epochs, "train_patches": int(tr.sum()),
            "test_patch_acc": round(pa, 4), "test_page_acc": round(pga, 4), "data": str(args.data)}
    MODEL_PATH.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {MODEL_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--scans", nargs="+", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--nodegoat", type=Path, default=HERE.parent / "data" / "nodegoat_export_2106.csv",
                   help="Nodegoat export with sender per letter (for the writer split)")
    a.add_argument("--workers", type=int, default=12)
    b = sub.add_parser("train")
    b.add_argument("--data", type=Path, required=True)
    b.add_argument("--epochs", type=int, default=8)
    b.add_argument("--batch", type=int, default=128)
    b.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    prepare(args) if args.cmd == "prepare" else train(args)


if __name__ == "__main__":
    main()
