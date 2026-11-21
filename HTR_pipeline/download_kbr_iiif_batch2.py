#!/usr/bin/env python3
"""
Download page images for Batch-2 KBR letters via their IIIF v3 manifests.

Input:  data/jpeg_lists/batch2_kbr_iiif_plan.json
        (list of {brief, object_id, idn, iiif, digitaal}, built from
         to_do/navez-brieven-stavaza-csv.csv joined to nodegoat_export_2106.csv)

Output: data/batch2_kbr_iiif/BE-KBR00_<IDN>_00000000_01_00_00_0_01_<page>_<n>.jpeg
        — the same filename convention as the KBR tome scans, so
        HTR_pipeline/run_htr_opus.py groups pages per letter via parts[1]=IDN
        and parts[8]=page index.
        Finally zipped to data/batch2_kbr_iiif.zip for run_htr_opus.py.

Resumable: pages already on disk are skipped. Letters whose manifest reports
more than --max-pages canvases are skipped and reported (likely a whole
microfilm reel / volume, not a single letter).

Usage:
    python HTR_pipeline/download_kbr_iiif_batch2.py
    python HTR_pipeline/download_kbr_iiif_batch2.py --max-pages 30 --zip
"""
from __future__ import annotations

import argparse
import json
import re
import time
import zipfile
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / "data" / "jpeg_lists" / "batch2_kbr_iiif_plan.json"
OUT_DIR = REPO / "data" / "batch2_kbr_iiif"
ZIP_PATH = REPO / "data" / "batch2_kbr_iiif.zip"

HEADERS = {"User-Agent": "Navez-correspondence-research (UGent GhentCDH)"}
BIB_RE = re.compile(r"uurl\.kbr\.be/bib/(\d+)")


def fetch_json(url: str, retries: int = 3) -> dict:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == retries:
                raise
            time.sleep(4 * attempt)
    raise RuntimeError("unreachable")


def manifest_image_urls(manifest: dict) -> list[str]:
    """Full-resolution image URL per canvas, in canvas order."""
    urls: list[str] = []
    for canvas in manifest.get("items", []):
        for page in canvas.get("items", []):
            for anno in page.get("items", []):
                body = anno.get("body") or {}
                if body.get("type") == "Image" and body.get("id"):
                    urls.append(body["id"])
    return urls


def manifest_bib_id(manifest: dict) -> str:
    m = BIB_RE.search(json.dumps(manifest))
    return m.group(1) if m else ""


def download(url: str, dest: Path, retries: int = 3) -> None:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=120)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return
        except Exception:
            if attempt == retries:
                raise
            time.sleep(4 * attempt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", type=Path, default=PLAN)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--max-pages", type=int, default=30,
                    help="Skip letters whose manifest has more canvases than this")
    ap.add_argument("--zip", action="store_true",
                    help="Write data/batch2_kbr_iiif.zip when done")
    args = ap.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ok = skipped_big = failed = 0
    oversized: list[tuple[str, str, int]] = []
    mismatches: list[tuple[str, str, str]] = []

    for i, row in enumerate(plan, start=1):
        idn, brief, url = row["idn"], row["brief"], row["iiif"]
        # some CSV cells hold several pipe-separated manifests (multi-part letters)
        manifest_urls = [u.strip() for u in url.split("|") if u.strip()]
        urls: list[str] = []
        bib = ""
        try:
            for murl in manifest_urls:
                manifest = fetch_json(murl)
                bib = bib or manifest_bib_id(manifest)
                urls.extend(manifest_image_urls(manifest))
        except Exception as e:
            print(f"[FAIL manifest] {brief} ({url}): {type(e).__name__}: {e}")
            failed += 1
            continue

        if bib and bib != idn:
            mismatches.append((brief, idn, bib))

        if not urls:
            print(f"[FAIL pages] {brief}: manifest has no image annotations")
            failed += 1
            continue
        if len(urls) > args.max_pages:
            print(f"[skip big] {brief} ({idn}): {len(urls)} pages > {args.max_pages}")
            oversized.append((brief, idn, len(urls)))
            skipped_big += 1
            continue

        n_new = 0
        for page_idx, img_url in enumerate(urls, start=1):
            # trailing token is cosmetic (parse_letter_page ignores it) but must
            # be deterministic so resumed runs find already-downloaded pages
            fname = (f"BE-KBR00_{idn}_00000000_01_00_00_0_01_"
                     f"{page_idx:04d}_{idn}{page_idx:02d}.jpeg")
            dest = args.out_dir / fname
            if dest.exists() and dest.stat().st_size > 0:
                continue
            try:
                download(img_url, dest)
                n_new += 1
                time.sleep(0.3)
            except Exception as e:
                print(f"[FAIL page] {brief} p{page_idx}: {type(e).__name__}: {e}")
                failed += 1
                break
        else:
            ok += 1
            print(f"[ok {i:3d}/{len(plan)}] {brief} ({idn}): "
                  f"{len(urls)} pages ({n_new} new)")
            continue

    print("\n=== Done ===")
    print(f"  letters ok: {ok}   skipped (too many pages): {skipped_big}   failed: {failed}")
    if oversized:
        print("  Oversized manifests (review manually):")
        for brief, idn, n in oversized:
            print(f"    {brief} ({idn}): {n} pages")
    if mismatches:
        print("  IDN vs manifest bib-id mismatches (nodegoat IDN kept):")
        for brief, idn, bib in mismatches:
            print(f"    {brief}: nodegoat {idn} / manifest bib {bib}")

    if args.zip:
        images = sorted(args.out_dir.glob("*.jpeg"))
        with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_STORED) as zf:
            for p in images:
                zf.write(p, p.name)
        print(f"  zip: {ZIP_PATH.relative_to(REPO)} ({len(images)} images)")


if __name__ == "__main__":
    main()
