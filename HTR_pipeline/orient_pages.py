#!/usr/bin/env python3
"""
Straighten scanned letter pages before HTR.

Detects whether a scan is rotated (90 / 180 / 270 degrees) or slightly skewed
and writes an upright copy next to the original. The original is never
modified. Every decision is appended to a CSV log so the historian can trace
what was done to which page.

Three steps:

1. Text axis (0 vs 90 degrees) and fine skew, with OpenCV + projection
   profiles: ink pixels are projected onto the vertical axis for a range of
   angles; the angle where the text lines give the sharpest profile wins.
   Both the 0 and the 90 degree family are scored; the higher one gives the
   axis of the writing.
2. Up/down (0 vs 180 degrees). Two options:
   - "heuristic": in French script ascenders (l, d, b, h, t, capitals) carry
     more ink than descenders (p, q, g, j). Per word blob we measure ink above
     vs below the x-height band. Free, but weaker on florid hands.
   - "claude": a cheap call on a small thumbnail. The model gets the page in
     both candidate orientations (A, B = A turned 180) and answers which one
     is readable.
   - "hybrid": heuristic first, Claude only when the heuristic is unsure, or
     when it says "upside down" (a costly mistake).
   - "classifier": a small CNN trained on our own scans decides the quarter
     turn (train_orientation_classifier.py); abstains when unsure.
   - "cascade" (default for the pipeline): classifier, then heuristic, then
     hybrid only when both are unsure. Best on unseen writers (see ORIENTATION.md).
3. Apply the rotation losslessly (transpose) and, if |skew| >= 1 degree,
   rotate by the skew angle with paper-coloured fill.

Command-line use (writes <out>/<name> and <out>/orientation_log.csv):

    python HTR_pipeline/orient_pages.py scans/*.jpeg --out data/_upright --mode heuristic
    python HTR_pipeline/orient_pages.py scans/ --out data/_upright --mode hybrid \
        --key-file config/key_openrouter.txt

In the HTR script: `run_htr_opus.py ... --orient hybrid` (see that file).
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import re
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True

ANALYSIS_DIM = 1600          # long edge used for analysis (px)
SKEW_RANGE = 12.0            # search +-12 degrees around each axis
SKEW_MIN_APPLY = 1.0         # do not resample for smaller skews (Opus copes with ~1 degree)
MIN_INK_FRACTION = 0.002     # below this the page is ~blank: leave it alone
UPDOWN_SURE = 0.12           # |heuristic score| above this counts as sure
AXIS_SURE_HI = 1.7           # axis ratio above: writing surely horizontal
AXIS_SURE_LO = 0.6           # axis ratio below: writing surely vertical (turn 90)
CLAUDE_MODEL = "anthropic/claude-haiku-5.5"
CLAUDE_THUMB_DIM = 900
CLF_DIM = 1400               # long edge at which classifier patches are cut
CLF_PATCH = 112              # square patch size (px) -> roughly 3-4 lines of writing
CLF_PATCHES = 24             # patches per page at inference
CLF_SURE = 0.80              # mean probability of the winning turn needed to act
CLF_MODEL_PATH = Path(__file__).resolve().parent / "models" / "orientation_cnn.pt"

LOG_FIELDS = ["file", "output", "rotation_cw", "skew_deg", "axis", "axis_ratio",
              "updown_heuristic", "updown_claude", "updown_source", "ink_fraction",
              "changed", "note"]


@dataclass
class Orientation:
    rotation_cw: int = 0          # 0/90/180/270: clockwise rotation applied to make upright
    skew_deg: float = 0.0         # extra counter-clockwise rotation applied after that (PIL sign)
    axis: str = "horizontal"      # writing axis in the *input* image
    axis_ratio: float = 1.0       # best horizontal score / best vertical score
    updown_heuristic: float = 0.0  # >0 upright, <0 upside down (after axis fix)
    updown_claude: str = ""       # "A" / "B" / "" (not asked) / "error: ..."
    updown_source: str = "heuristic"
    ink_fraction: float = 0.0
    note: str = ""
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- image helpers
def _to_gray_small(im: Image.Image, dim: int = ANALYSIS_DIM) -> np.ndarray:
    g = np.asarray(im.convert("L"))
    s = dim / max(g.shape)
    if s < 1:
        g = cv2.resize(g, (int(g.shape[1] * s), int(g.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return g


def _paper_box(gray: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box of the paper (largest bright region), shrunk a little so the
    dark scanner background and the paper edge do not count as ink."""
    blur = cv2.GaussianBlur(gray, (0, 0), 6)
    _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return 0, 0, gray.shape[1], gray.shape[0]
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[i, :4]
    if w * h < 0.25 * gray.size:          # no clear paper region: use everything
        return 0, 0, gray.shape[1], gray.shape[0]
    mx, my = int(0.03 * w), int(0.03 * h)
    return x + mx, y + my, w - 2 * mx, h - 2 * my


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask (uint8 0/1) of handwriting strokes inside the paper area."""
    x, y, w, h = _paper_box(gray)
    roi = gray[y:y + h, x:x + w]
    block = max(15, (min(roi.shape) // 40) | 1)
    m = cv2.adaptiveThreshold(roi, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 12)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # Drop specks and very large blobs (stains, seals, binding thread, ruler).
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    area = stats[:, cv2.CC_STAT_AREA]
    bw, bh = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    big = max(roi.shape) * 0.25
    keep = (area >= 6) & (bw < big) & (bh < big)
    keep[0] = False
    m = keep[lab].astype(np.uint8)
    # Erase long straight lines (paper edges, the stitched gutter of a bifolium,
    # rulers, folds): they give razor-sharp profile peaks that are not text and
    # can outvote the writing when it comes to choosing the axis.
    lines = cv2.HoughLinesP(m * 255, 1, np.pi / 360, threshold=60,
                            minLineLength=int(0.06 * max(roi.shape)), maxLineGap=6)
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(m, (int(x1), int(y1)), (int(x2), int(y2)), 0, thickness=7)
    full = np.zeros_like(gray, dtype=np.uint8)
    full[y:y + h, x:x + w] = m
    return full


# --------------------------------------------------------------------------- axis + skew
def _profile_score(ys: np.ndarray, length: int) -> float:
    """Sharpness of a projection profile: sum of squared differences between
    neighbouring bins, normalised so it does not depend on the amount of ink."""
    hist = np.bincount(np.clip(ys, 0, length - 1), minlength=length).astype(np.float64)
    hist = np.convolve(hist, np.ones(3) / 3, mode="same")
    # sqrt: one long straight stroke (an underline, a stray rule) must not
    # outweigh many ordinary lines of writing.
    hist = np.sqrt(hist)
    d = np.diff(hist)
    return float((d * d).sum() / (hist.sum() ** 2 / length + 1e-9))


def _best_angle(xs: np.ndarray, ys: np.ndarray, base_deg: float) -> tuple[float, float]:
    """Search the angle (around base_deg) at which rows of text are sharpest."""
    cx, cy = xs.mean(), ys.mean()
    x0, y0 = xs - cx, ys - cy
    span = int(np.hypot(x0.max() - x0.min(), y0.max() - y0.min())) + 4

    def score(a):
        t = np.deg2rad(a)
        proj = (-x0 * np.sin(t) + y0 * np.cos(t)) + span / 2
        return _profile_score(proj.astype(np.int64), span)

    coarse = np.arange(base_deg - SKEW_RANGE, base_deg + SKEW_RANGE + 1e-6, 1.0)
    sc = [score(a) for a in coarse]
    a0 = coarse[int(np.argmax(sc))]
    fine = np.arange(a0 - 1.0, a0 + 1.0 + 1e-6, 0.1)
    sf = [score(a) for a in fine]
    k = int(np.argmax(sf))
    return float(fine[k]), float(sf[k])


def detect_axis_and_skew(mask: np.ndarray, max_points: int = 150_000, seed: int = 0):
    """Return (skew_h, skew_v, ratio). Skews are in image coordinates (degrees,
    y down): the angle the text lines make with the x axis (if the writing is
    horizontal) or with the y axis (if vertical). ratio = horizontal score /
    vertical score: >1 means horizontal writing."""
    ys, xs = np.nonzero(mask)
    if len(xs) > max_points:
        idx = np.random.default_rng(seed).choice(len(xs), max_points, replace=False)
        xs, ys = xs[idx], ys[idx]
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)
    a_h, s_h = _best_angle(xs, ys, 0.0)
    a_v, s_v = _best_angle(xs, ys, 90.0)
    ratio = s_h / (s_v + 1e-9)
    return a_h, a_v - 90.0, ratio


# --------------------------------------------------------------------------- up / down
def updown_score(mask: np.ndarray) -> tuple[float, int]:
    """>0: text is upright, <0: upside down. Assumes text lines are horizontal.

    Words are found by smearing ink horizontally. Within each word the x-height
    band is the set of rows holding >= 45% of the word's peak row ink; ink
    above that band (ascenders) is compared with ink below it (descenders).
    """
    n_rows = mask.shape[0]
    # Estimate stroke/letter size from component heights to size the smear kernel.
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    hs = stats[1:, cv2.CC_STAT_HEIGHT]
    if len(hs) < 20:
        return 0.0, 0
    xh = max(4, int(np.median(hs)))
    smear = cv2.dilate(mask, np.ones((max(1, xh // 3), xh * 2), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(smear, 8)
    above = below = 0.0
    used = 0
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if w < 3 * xh or h < xh or h > 8 * xh:
            continue
        sub = mask[y:y + h, x:x + w] * (lab[y:y + h, x:x + w] == i)
        prof = sub.sum(axis=1).astype(np.float64)
        if prof.sum() < 30:
            continue
        core = np.nonzero(prof >= 0.45 * prof.max())[0]
        top, bot = core.min(), core.max()
        a_ink, b_ink = prof[:top].sum(), prof[bot + 1:].sum()
        if a_ink + b_ink == 0:
            continue
        above += a_ink
        below += b_ink
        used += 1
    if used < 5 or above + below == 0:
        return 0.0, used
    return float((above - below) / (above + below)), used


# --------------------------------------------------------------------------- Claude check
def _thumb_uri(im: Image.Image, dim: int = CLAUDE_THUMB_DIM) -> str:
    im = im.convert("RGB")
    im.thumbnail((dim, dim), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


CLAUDE_PROMPT = (
    "The {n} images below are the same scanned 19th-century handwritten letter, "
    "each turned differently. Which image shows the main handwritten text the right "
    "way up, so that it can be read normally (lines horizontal, not upside down, "
    "not sideways)? Answer with a single letter: {letters}."
)


def claude_pick_upright(candidates: list[Image.Image], api_key: str,
                        model: str = CLAUDE_MODEL, timeout: int = 60) -> str:
    """Ask Claude (via OpenRouter) which candidate image is upright.
    Returns 'A', 'B', ... (index into candidates) or 'error: ...'."""
    letters = "ABCD"[:len(candidates)]
    content = [{"type": "text", "text": CLAUDE_PROMPT.format(
        n=len(candidates), letters=" or ".join(letters))}]
    for L, im in zip(letters, candidates):
        content.append({"type": "text", "text": f"Image {L}:"})
        content.append({"type": "image_url", "image_url": {"url": _thumb_uri(im)}})
    # Reasoning off + room to spare: with a tiny budget the model sometimes
    # spends every token on hidden reasoning and returns an empty answer.
    body = {"model": model, "temperature": 0, "max_tokens": 400,
            "reasoning": {"enabled": False},
            "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                 "X-Title": "Navez HTR - orientation check"},
    )
    last = "error: no answer"
    for attempt in (1, 2, 3, 4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            txt = (data["choices"][0]["message"]["content"] or "").strip().upper()
            m = re.search(rf"\b([{letters}])\b", txt)
            if m:
                return m.group(1)
            # Some OpenRouter routes intermittently return an empty message: retry.
            last = f"error: unparsable answer {txt[:20]!r}"
        except Exception as e:  # network / rate limit: retry, then give up gracefully
            last = f"error: {type(e).__name__}"
        time.sleep(2 * attempt)
    return last


# --------------------------------------------------------------------------- CNN classifier
def sample_patches(im: Image.Image, n: int, rng: np.random.Generator) -> np.ndarray:
    """Cut up to n ink-rich square grey patches (uint8, n x P x P) from a page.
    Returns an empty array for (nearly) blank pages."""
    g = _to_gray_small(im, CLF_DIM)
    mask = ink_mask(g)
    if mask.mean() < MIN_INK_FRACTION:
        return np.zeros((0, CLF_PATCH, CLF_PATCH), np.uint8)
    P, half = CLF_PATCH, CLF_PATCH // 2
    dens = cv2.boxFilter(mask.astype(np.float32), -1, (P, P))
    dens[:half, :] = dens[-half:, :] = 0
    dens[:, :half] = dens[:, -half:] = 0
    ys, xs = np.nonzero(dens > 0.04)
    if len(ys) == 0:
        return np.zeros((0, P, P), np.uint8)
    w = dens[ys, xs].astype(np.float64)
    out = []
    for i in rng.choice(len(ys), size=min(n * 4, len(ys)), replace=False, p=w / w.sum()):
        y, x = ys[i], xs[i]
        # keep patches apart a little so they do not all show the same word
        if any(abs(y - a) < half and abs(x - b) < half for a, b, _ in out):
            continue
        out.append((y, x, g[y - half:y + half, x - half:x + half]))
        if len(out) == n:
            break
    return np.stack([p for _, _, p in out])


def patches_to_tensor(b: np.ndarray):
    """uint8 patches -> normalised float tensor N x 1 x P x P (ink = high values)."""
    import torch
    x = torch.as_tensor(np.ascontiguousarray(b), dtype=torch.float32)
    x = (255.0 - x) / 255.0
    x = (x - x.mean(dim=(1, 2), keepdim=True)) / (x.std(dim=(1, 2), keepdim=True) + 1e-3)
    return x.unsqueeze(1)


_clf_model = None


def _load_classifier():
    global _clf_model
    if _clf_model is None:
        import torch
        from train_orientation_classifier import build_model
        m = build_model()
        m.load_state_dict(torch.load(CLF_MODEL_PATH, map_location="cpu"))
        m.eval()
        _clf_model = m
    return _clf_model


def classifier_rotation(im: Image.Image, n: int = CLF_PATCHES, seed: int = 0) -> tuple[int | None, float]:
    """Let n patches vote. Returns (rotation_cw that makes the page upright,
    mean probability of that answer), or (None, 0) for a blank page."""
    import torch
    import torch.nn.functional as F
    patches = sample_patches(im, n, np.random.default_rng(seed))
    if len(patches) == 0:
        return None, 0.0
    with torch.no_grad():
        logp = F.log_softmax(_load_classifier()(patches_to_tensor(patches)), 1)
    k = int(logp.sum(0).argmax())
    # label k = page was turned k x 90 degrees counter-clockwise -> turn it back clockwise
    return (90 * k) % 360, float(logp.exp().mean(0)[k])


# --------------------------------------------------------------------------- main API
def _rotate_cw(im: Image.Image, deg: int) -> Image.Image:
    return {0: im, 90: im.transpose(Image.ROTATE_270),
            180: im.transpose(Image.ROTATE_180), 270: im.transpose(Image.ROTATE_90)}[deg % 360]


def _paper_colour(im: Image.Image) -> tuple:
    g = np.asarray(im.convert("RGB"))
    return tuple(int(v) for v in np.median(g.reshape(-1, 3), axis=0))


def _kept(o: Orientation) -> bool:
    """True when a stage abstained (left the page as is because it was unsure)."""
    return o.rotation_cw == 0 and any(s in o.note for s in ("kept", "not flipped", "unsure"))


def analyse(im: Image.Image, mode: str = "heuristic", api_key: str | None = None,
            model: str = CLAUDE_MODEL) -> Orientation:
    """Work out how to make `im` upright. Does not modify the image."""
    o = Orientation()
    gray = _to_gray_small(im)
    mask = ink_mask(gray)
    o.ink_fraction = round(float(mask.mean()), 5)
    if o.ink_fraction < MIN_INK_FRACTION:
        o.note = "too little ink; left as is"
        o.updown_source = "none"
        return o

    skew_h, skew_v, ratio = detect_axis_and_skew(mask)
    o.axis_ratio = round(ratio, 3)
    o.axis = "horizontal" if ratio >= 1 else "vertical"
    # A line at image angle `skew` (y down, i.e. sloping down to the right) is
    # undone by PIL rotate(+skew) (counter-clockwise). A 90-cw turn maps a
    # near-vertical line at 90+s to a near-horizontal one at s, so the same
    # correction holds in both families (and survives a later 180 flip).
    use_claude = mode in ("hybrid", "claude") and bool(api_key)

    def candidate(rot_cw: int) -> Image.Image:
        sk = skew_h if rot_cw % 180 == 0 else skew_v
        c = _rotate_cw(im, rot_cw)
        if abs(sk) >= SKEW_MIN_APPLY:
            c = c.rotate(sk, resample=Image.BILINEAR, fillcolor=_paper_colour(c))
        return c

    def finish(rot_cw: int) -> Orientation:
        sk = skew_h if rot_cw % 180 == 0 else skew_v
        o.rotation_cw = rot_cw % 360
        o.skew_deg = round(sk, 2) if abs(sk) >= SKEW_MIN_APPLY else 0.0
        return o

    def add_note(t: str) -> None:
        o.note = (o.note + "; " if o.note else "") + t

    # --- Cascade: classifier first (never wrong on unseen hands when sure), then
    # the OpenCV heuristic, and Claude only when both are unsure.
    if mode == "cascade":
        try:
            rot, prob = classifier_rotation(im)
        except Exception as e:  # no torch / no model file: skip this stage
            rot, prob = None, 0.0
            add_note(f"classifier unavailable ({type(e).__name__})")
        if rot is not None and prob >= CLF_SURE:
            o.extra["clf_prob"] = round(prob, 3)
            o.updown_source = "classifier"
            add_note(f"classifier p={prob:.2f}")
            return finish(rot)
        stage = f"classifier unsure (p={prob:.2f})" if rot is not None else "classifier: no usable patches"
        if not o.note:
            o.note = stage
        h = analyse(im, mode="heuristic")
        if not _kept(h):
            h.note = o.note + ("; " + h.note if h.note else "")
            h.extra["clf_prob"] = round(prob, 3)
            return h
        r = analyse(im, mode="hybrid", api_key=api_key, model=model)
        r.extra["clf_prob"] = round(prob, 3)
        if "4-way" in r.note and r.rotation_cw != rot:
            # Claude's 4-way pick is right only ~80% of the time on unseen hands.
            # Accept it only when the classifier's best guess agrees; otherwise
            # keep the scan as it is (h already holds that, with the 0-degree skew).
            h.updown_claude = r.updown_claude
            h.extra["clf_prob"] = round(prob, 3)
            h.note = (o.note + f"; claude 4-way said {r.rotation_cw}, classifier {rot}: "
                      "no agreement; orientation kept")
            return h
        r.note = o.note + ("; " + r.note if r.note else "")
        return r

    # --- CNN classifier: decides the quarter turn; OpenCV still gives the skew.
    if mode == "classifier":
        rot, prob = classifier_rotation(im)
        o.extra["clf_prob"] = round(prob, 3)
        o.updown_source = "classifier"
        if rot is None:
            add_note("classifier: no usable patches; orientation kept")
            return finish(0)
        if prob < CLF_SURE:
            add_note(f"classifier unsure (p={prob:.2f}); orientation kept")
            return finish(0)
        add_note(f"classifier p={prob:.2f}")
        return finish(rot)

    # --- 1. which axis does the writing run along?
    if AXIS_SURE_LO < ratio < AXIS_SURE_HI:
        # Ambiguous (little text, sideways address next to the body, ...).
        if use_claude:
            rots = [0, 90, 180, 270]
            ans = claude_pick_upright([candidate(r) for r in rots], api_key, model)
            o.updown_claude = ans
            if ans in ("A", "B", "C", "D"):
                o.updown_source = "claude"
                add_note("axis uncertain; asked claude (4-way)")
                return finish(rots["ABCD".index(ans)])
            add_note(f"claude check failed ({ans})")
        # Prior: nearly all scans are upright, so do not turn on weak evidence.
        add_note("axis uncertain; orientation kept")
        o.updown_source = "none"
        return finish(0)

    base = 90 if ratio <= AXIS_SURE_LO else 0
    skew = skew_v if base == 90 else skew_h

    # --- 2. up or down? Score the ink mask with the writing horizontal and level.
    m2 = np.ascontiguousarray(np.rot90(mask, k=-1)) if base == 90 else mask
    if abs(skew) >= SKEW_MIN_APPLY:
        h, w = m2.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
        m2 = (cv2.warpAffine(m2 * 255, M, (w, h), flags=cv2.INTER_NEAREST) > 127).astype(np.uint8)
    o.updown_heuristic, used = updown_score(m2)
    o.updown_heuristic = round(o.updown_heuristic, 4)
    o.extra["words_used"] = used

    flip = o.updown_heuristic < 0
    sure = abs(o.updown_heuristic) >= UPDOWN_SURE
    ask = mode == "claude" or (mode == "hybrid" and (not sure or flip))
    if ask and use_claude:
        ans = claude_pick_upright([candidate(base), candidate(base + 180)], api_key, model)
        o.updown_claude = ans
        if ans in ("A", "B"):
            flip = ans == "B"
            o.updown_source = "claude"
        else:
            add_note(f"claude check failed ({ans}); used heuristic")
    elif ask:
        add_note("no API key; used heuristic")
    if not sure and o.updown_source == "heuristic" and flip:
        # Weak evidence for upside down and no second opinion: do not flip.
        flip = False
        add_note("weak up/down evidence; not flipped")
        if base == 90:
            # Sideways page, but 90 vs 270 unclear: turning it might leave it
            # upside down, so leave the scan as it is and let the log flag it.
            add_note("sideways but direction unclear; orientation kept")
            return finish(0)
    return finish(base + (180 if flip else 0))


def apply(im: Image.Image, o: Orientation) -> Image.Image:
    out = _rotate_cw(im, o.rotation_cw)
    if o.skew_deg:
        # A skew measured before a 180 flip keeps its sign after it.
        out = out.rotate(o.skew_deg, resample=Image.BICUBIC, expand=True,
                         fillcolor=_paper_colour(im))
    return out


def straighten_file(src: Path, out_dir: Path, log_csv: Path | None = None, mode: str = "heuristic",
                    api_key: str | None = None, model: str = CLAUDE_MODEL,
                    force: bool = False) -> tuple[Path, Orientation | None]:
    """Write an upright copy of `src` into `out_dir` and log the decision.

    Returns the path to feed to HTR: the corrected copy, or `src` itself when
    nothing had to change (so unchanged pages are sent byte-for-byte as before).
    If the page was already processed (row in the log), the logged result is reused.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log_csv = log_csv or out_dir / "orientation_log.csv"
    dst = out_dir / src.name
    if not force and log_csv.exists():
        with log_csv.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["file"] == str(src):
                    return (Path(row["output"]) if row["changed"] == "1" else src), None
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        o = analyse(im, mode=mode, api_key=api_key, model=model)
        changed = bool(o.rotation_cw or o.skew_deg)
        if changed:
            fixed = apply(im, o).convert("RGB")
            fixed.save(dst, quality=95)
    new = not log_csv.exists()
    with log_csv.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        row = {k: v for k, v in asdict(o).items() if k in LOG_FIELDS}
        row.update(file=str(src), output=str(dst) if changed else "", changed=int(changed))
        w.writerow(row)
    return (dst if changed else src), o


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="image files or folders")
    ap.add_argument("--out", type=Path, required=True, help="folder for corrected copies + log")
    ap.add_argument("--mode", choices=["heuristic", "hybrid", "claude", "classifier", "cascade"], default="heuristic")
    ap.add_argument("--key-file", type=Path, help="OpenRouter key file (needed for hybrid/claude)")
    ap.add_argument("--model", default=CLAUDE_MODEL)
    ap.add_argument("--force", action="store_true", help="re-analyse pages already in the log")
    args = ap.parse_args()
    key = args.key_file.read_text(encoding="utf-8").strip() if args.key_file else None
    files = []
    for p in args.inputs:
        files += sorted(q for q in p.rglob("*") if q.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}) \
            if p.is_dir() else [p]
    for p in files:
        dst, o = straighten_file(p, args.out, mode=args.mode, api_key=key, model=args.model, force=args.force)
        if o is None:
            print(f"  [logged] {p.name}")
        else:
            print(f"  {p.name}: rotate {o.rotation_cw:>3} cw, skew {o.skew_deg:+.2f}, "
                  f"up/down {o.updown_heuristic:+.3f} ({o.updown_source}{' ' + o.updown_claude if o.updown_claude else ''})"
                  f"{'  ' + o.note if o.note else ''}")


if __name__ == "__main__":
    main()
