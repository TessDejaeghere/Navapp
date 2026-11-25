# Page orientation before HTR

`orient_pages.py` straightens scans that are sideways, upside down or skewed
before they go to Claude Opus. `run_htr_opus.py --orient ...` uses it.

## Use

```
pip install opencv-python-headless torch   # both are in requirements.txt

# in the HTR run: cascade is the default
python HTR_pipeline/run_htr_opus.py data/brieven_te_HTRen/<batch>.zip
python HTR_pipeline/run_htr_opus.py data/brieven_te_HTRen/<batch>.zip --orient hybrid      # previous default
python HTR_pipeline/run_htr_opus.py data/brieven_te_HTRen/<batch>.zip --orient heuristic   # no model calls
python HTR_pipeline/run_htr_opus.py data/brieven_te_HTRen/<batch>.zip --orient off         # exact pilot method

# on its own, e.g. to check a folder of scans first
python HTR_pipeline/orient_pages.py data/tomes_KBR/_pages_<batch> --out data/_upright_<batch> --mode heuristic
```

Modes:

| mode | what decides | cost |
|---|---|---|
| `cascade` (default) | the classifier when it is sure; otherwise OpenCV when it is sure; otherwise `hybrid` (Claude) | about 1 Claude call per 5 turned pages; fewer on normal, upright scans |
| `heuristic` | OpenCV only | free |
| `hybrid` | OpenCV; a Claude Haiku thumbnail check when OpenCV is unsure, or before a page is turned upside down | a few cents per hundred pages |
| `claude` | the up/down call is always made by Claude | about one small call per page |
| `classifier` | a small CNN trained on our own scans (`models/orientation_cnn.pt`, needs `torch`); OpenCV for skew | free, CPU |

The classifier is trained with `train_orientation_classifier.py` (prepare, then
train; about 30 min on a laptop CPU). It cuts 112 px ink patches from every
scan, turns each patch by a random quarter turn, and learns to predict that
turn. About 24 patches vote on each page. If the mean vote is under 0.80, the
classifier abstains. Pages are split into train/val/test **by writer** (the
sender in the Nodegoat export), stratified by how many pages each writer has.
That way the test set only holds hands the model never saw. Sender stands in
for the hand: a secretary or copyist can hide behind one sender.

If `torch` or the model file is missing, `cascade` skips the classifier, writes
a note in the log, and carries on with OpenCV and Claude.

## What it does

1. **Axis and skew.** Ink is binarised (adaptive threshold). Long straight
   lines are erased first: paper edges, the stitched gutter of a bifolium,
   and folds. Then the ink is projected at angles from −12° to +12° around 0°
   and around 90°. The angle with the sharpest text-line profile gives the skew.
   Comparing the best 0° score with the best 90° score gives the axis. If the
   ratio is between 0.6 and 1.7, the axis counts as uncertain. In that case
   `hybrid` asks Claude to pick among four turned copies. `heuristic` leaves
   the page as it is.
2. **Up or down.** In French script, ascenders (l, d, h, b, t, capitals) carry
   more ink than descenders (p, q, g, j). Within each word, the ink above the
   x-height band is compared with the ink below it. A weak score never flips
   a page on its own.
3. **Apply.** Quarter turns are lossless (pixel transpose). Skew is only
   corrected when it is at least 1°.

Safety rules: pages with almost no ink (blank versos) are left alone. Unsure
cases are left as they are and get a note in the log. Pages that need no
change are sent to Opus byte for byte as before.

## Traceability

- Originals are never modified. Corrected copies go to `_upright_<zip-stem>/`,
  next to `_pages_<zip-stem>/`.
- `data/htr_opus_outputs/<batch>/orientation_log.csv` has one row per page:
  `rotation_cw` (clockwise turn applied), `skew_deg` (extra counter-clockwise
  turn), the axis ratio, the up/down score, Claude's answer, which source
  decided, and a `note` for every uncertain case.
- `usage.jsonl` records, for each letter, the `--orient` mode and which pages
  were straightened.

Filter the log on `note` to find pages the historian should look at by eye.

## Test (Nov 2026)

Test set: the 3 lesson samples plus 40 random KBR/OCMW scans. Each was turned
by 0/90/180/270° and skewed by 0, +3 and −5°. Six of the scans were blank
versos (rightly skipped), leaving 444 cases with text.

| | correct | left unchanged (unsure) | wrong turn |
|---|---|---|---|
| heuristic | 390 / 444 (87.8%) | 54 | 0 |
| hybrid (Claude Haiku 5.5) | 444 / 444 (100%) | 0 | 0 |

- Upright pages were never turned by mistake in either mode (111 / 111).
- Hybrid made 264 Claude calls; all 264 answers were correct. Total cost was
  under $0.10 (about $0.0002 per call). On real, upright scans, 8 of 37 pages
  (about 1 in 5) were unsure enough to ask Claude. For the whole corpus that
  comes to roughly 1,300 calls, about $0.30.
- Remaining skew after correction: median 0.1°. Skews under 1° are left alone
  on purpose (90th percentile 0.8°).
- The Claude call runs with reasoning turned off and a 400-token limit. With a
  very small limit, Haiku sometimes spent every token on hidden reasoning and
  returned an empty answer.
- The cases left unchanged are sparse pages, and spreads with a sideways
  address panel next to the letter body.
- One real scan, 21329079 p.2 (an address panel), is sideways in the KBR scan
  itself. It was turned upright correctly.

## Test 2: unseen writers (Nov 2026)

This test uses the 50 writers in the classifier's test split, so none of the
three modes had seen these hands. It takes up to 3 pages per writer: 121 pages.
Each page was turned 0/90/180/270° and skewed 0, +3 and −5°, which gives 1,452
cases per mode. The confidence interval resamples writers, not pages.

| | correct (95% CI over writers) | wrong turn | upright page turned | Claude calls |
|---|---|---|---|---|
| heuristic | 76.4% (70–83) | 13 | 3 / 363 | 0 |
| classifier | 70.2% (64–76) | **0** | **0** / 363 | 0 |
| hybrid | 93.9% (90–97) | 89 | 16 / 363 | 957 |
| cascade, first version (measured) | 95.0% (91–98) | 45 | 18 / 363 | 290 |
| **cascade, final (default)** | **94.6%** | **15** | **5** / 363 | 290 |

All other cases are pages left as they are because the mode was unsure. That
is the same as no preprocessing, so it does less harm than a wrong turn.

- These pages are harder than the ones in test 1, so all scores are lower.
- The classifier never made a wrong turn. In the cascade it decided 864 cases
  and got all 864 right. It holds back on the rest.
- 88 of hybrid's 89 wrong turns come from Claude. Most of them come from the
  four-way question asked when the axis is uncertain. That question was right
  in only 216 of 271 cascade cases (80%). The two-way up/down question is
  much more reliable.
- The **final cascade** therefore accepts Claude's four-way answer only when
  the classifier's best guess (even below its 0.80 bar) agrees. Otherwise the
  scan stays as it is, with a note in the log. This drops wrong turns from 45
  to 15 and upright pages turned from 18 to 5, at the cost of 5 correct
  answers. Its row is the measured first-version run, with the 271 four-way
  cases replayed under the new rule. The Claude answers come from that run
  and the classifier is deterministic, so no other case changes.
- The remaining 15 wrong turns are pages where the classifier found no usable
  ink patches (12), or OpenCV was sure and wrong (3).
- An earlier estimate of 96% with 0 upright pages turned, combined from the
  three separate runs, was too optimistic. It counted "unsure, kept upright"
  as a sure answer.
- Weakest hands for every mode: du Bus de Gisignies, Voisin, Vanden Berghe,
  Potvin. These are sparse or very slanted pages.
