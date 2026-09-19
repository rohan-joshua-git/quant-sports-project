# Decision Log

Structural and scoping decisions for the Quant Sports Trading Project, logged with
dated rationale as they are made, per the process discipline in `CLAUDE.md`. This
covers decisions, problems encountered, and how each was resolved (or left open).
Factor-level decisions belong in `factor_log.md` once factor testing begins, not here.

---

## 2026-09-18: Sport = soccer `[DECIDED]`

Soccer selected over alternative sports, primarily on the strength of publicly
available broadcast video and computer-vision research infrastructure (SoccerNet in
particular), which no other sport currently matches at comparable scale and
documentation quality.

---

## 2026-09-18: Market universe = team-level match markets `[DECIDED]`

Team-level match markets (win probability, in-play repricing) selected over
player-level proposition markets. Player-level factors were rejected for this first
version on three grounds: match markets have materially better liquidity and
odds-data coverage; SoccerNet's annotation coverage is strongest at match/event
level; point-in-time alignment is simpler against a single match clock than a
many-player panel. Player-level factors remain a candidate second phase.

---

## 2026-09-18: Video/CV source selected, then access lost same day

**Original decision:** DFL Bundesliga Data Shootout (Kaggle), 2022 Bundesliga
season, selected as primary analysis footage. Usable for this personal project per
written confirmation from Kaggle's Data Team (confirmation email kept on file).

**Problem discovered:** the Kaggle competition's data download is no longer
available (competition closed, download disabled). This effectively voids the
practical value of the written Kaggle confirmation, since the data it covers is no
longer accessible through that channel regardless of the permission's validity.

**Investigation performed:**
- Kaggle's own competition/dataset pages are JS-rendered, so automated fetching
  could not directly confirm whether this was a full delisting vs. a gated
  re-accept-rules flow; user confirmed directly it shows "competition closed,
  download unavailable."
- Checked SoccerNet's main corpus as the already-considered alternative: it covers
  the 2014-2017 seasons, which is *older* than DFL's 2022 footage, not closer to
  the 2024-onward market data on Kalshi/Polymarket. Switching to SoccerNet does not
  resolve the temporal-mismatch blocker (see below); if anything it worsens it.
- Identified but did not adopt: a re-uploaded Kaggle dataset (`ghrangel/bundesliga`)
  and a Hugging Face mirror (`dbal0503/Bundesliga`). Neither is covered by the
  original Kaggle Data Team confirmation, so licensing status for either is
  unverified and not to be assumed clean.
- Checked DFL's own official site for an alternate post-competition access route:
  none found.

**Interim solution:** a local copy of at least one DFL clip (`08fd33_4.mp4`),
downloaded before the Kaggle listing closed, exists locally and is being used for
private technical smoke-testing only (see YOLO detection test below). This is not
being redistributed and is not yet treated as the settled primary dataset for the
full research build.

**Status: `[OPEN]`.** Primary video/CV source for the actual research run (as
opposed to local smoke-testing) is unresolved. Losing DFL does not change the
tracking-accuracy validation decision (SoccerNet-Tracking + HOTA, below), which was
always a separate proxy dataset.

---

## 2026-09-18: Tracking-accuracy validation = SoccerNet-Tracking + HOTA `[DECIDED]`

SoccerNet-Tracking (Swiss Super League, 12 games, labeled bounding boxes and
tracklet IDs) used to validate tracking pipeline architecture, scored via HOTA
rather than MOTA, since HOTA weights detection and identity-association quality
more evenly and association is where trackers typically fail across camera cuts.
Explicitly a proxy validation (different league/footage from the primary analysis
video), not a substitute for a manual spot-check on the actual footage in use.
Unaffected by the DFL access problem above.

---

## 2026-09-18: Market platform = Kalshi/Polymarket `[DECIDED in principle, blocked]`

Selected initially for API accessibility and confirmed (2026) continuous in-play
soccer markets plus historical-data APIs. Not treated as final; blocked by the
temporal mismatch below.

---

## 2026-09-18: Critical blocker - video/market temporal mismatch `[OPEN]`

DFL video (2022) predates Polymarket's price history (2024+) and Kalshi's
sports-market expansion (post-2022). Three options remain undecided:
1. Decouple: validate the CV signal against realized outcomes only now; treat
   market-fit as a later, forward-looking phase with fresh video capture.
2. Source period-matched 2022 odds from Betfair Historical Data or
   football-data.co.uk instead of Kalshi/Polymarket.
3. Abandon archival video; build forward-looking against live/recent matches
   (reopens broadcast-copyright/scraping risk).

**Update from the DFL-access problem above:** since the natural fallback video
source (SoccerNet, 2014-2017) is even further from 2024+ market data than DFL was,
any archival-video choice pushes this problem in the same direction. This makes
option 1 (decouple) the practically favored path, but it has not been formally
locked as a decision yet and remains open.

---

## 2026-09-18: Smoke-test folder structure `[DECIDED]`

Technical spikes/diagnostics (e.g. running a pretrained detector to see what
breaks) placed under `experiments/`, kept separate from `research/` (formal
scoping/decision documents) to avoid overloading the "Phase 0" name, which refers
specifically to the scoping phase. Spikes do not require pre-registration (no
factor or dataset decision is being tested), but material findings from them are
logged (in the relevant notebook, and here for structural implications) rather than
left undocumented, consistent with the institutional-grade standard for this
project.

Practical note: derived video frames/outputs from these spikes (`runs/`, `*.avi`,
`*.jpg`/`*.png` detection outputs, the `Tester video/` source folder) must be
gitignored before any commit, since CLAUDE.md's public-repo rule excludes raw video
and derived video frames from public output.

---

## 2026-09-18: YOLO26 pretrained detection smoke test - environment problems and fixes

Ran a diagnostic-only (no training) pass of a pretrained YOLO26n checkpoint against
the locally-cached DFL clip `08fd33_4.mp4`, to check off-the-shelf detection
quality before committing to a build approach for Stage 1 (detection & tracking).

**Problems hit while setting this up, and fixes:**
- `from ultralytics import yolo` (lowercase) raised `ImportError`; the class is
  capitalized (`YOLO`).
- Windows file path passed as a plain string had its backslashes misinterpreted as
  escape sequences (`\r`, `\T`, etc.), corrupting the path; fixed with a raw string
  (`r"..."`).
- `ModuleNotFoundError: No module named 'ultralytics'` at runtime despite the
  package being installed: caused by the notebook's Jupyter kernel resolving to a
  Python 3.12 interpreter, while `ultralytics` was installed under a separate
  Python 3.14 interpreter. Fixed by switching the notebook's kernel (via VS Code's
  kernel picker) to the Python 3.14 interpreter, which already had a registered
  Jupyter kernelspec.
- Model version assumption was stale: initial guidance assumed YOLOv8 was current;
  corrected to YOLO26 (Ultralytics, released January 2026), which is the version
  actually used for this test. Same `ultralytics` package, no workflow change
  beyond checkpoint name (`yolo26n.pt`).

---

## 2026-09-18: YOLO26 pretrained detection smoke test - findings

Reviewing the annotated output surfaced three problems with the stock,
COCO-pretrained detector, all visible in a single reference frame kept in
`experiments/yolo_smoke_test.ipynb`:

1. **Ball detection unreliable.** `sports ball` fires intermittently at low
   confidence (~0.3), consistent with known weaknesses of generic COCO-pretrained
   weights on small/fast objects.
2. **False positive on broadcast graphics.** Part of the on-screen scoreboard/graphic
   misclassified as `tv` (~0.26 confidence).
3. **Sideline personnel misclassified as players.** Coaches, medical staff, and bench
   personnel are detected as `person`, indistinguishable from players, with no
   pitch-boundary or role filtering.

**Conclusion:** stock pretrained YOLO26 is not sufficient as-is for Stage 1. Full
detail logged in the notebook itself.

---

## 2026-09-18: Decision - retrain detector on a Roboflow dataset `[DECIDED]`

**Decision:** fine-tune YOLO26 on the **"Football Players Detection"** dataset
(Roboflow Universe), instead of relying on generic COCO weights, to address all
three findings above from a single source.

**Rationale:**
- Player detection: soccer-specific, should reduce class confusion vs. generic COCO
  weights.
- Ball detection: soccer-specific ball annotations directly target the unreliable
  `sports ball` detections.
- Referee/sideline personnel: annotated as a distinct class from players, directly
  addressing the sideline false-positive finding without a separate pitch-mask step.

**Licensing:** checked, permissive (CC BY 4.0). To be kept on file alongside the
DFL Kaggle confirmation.

**Status:** decision made; fine-tuning run not yet executed.

---

## 2026-09-18: Compute infrastructure - local GPU training abandoned after hardware failure `[DECIDED]`

**Problem:** while attempting to run the Roboflow fine-tuning job locally on the
machine's GPU (NVIDIA GTX 1650, 4GB VRAM), a capacitor blew, likely a short circuit
related to the power adapter under the sustained load of training. Fixed, no
lasting damage to the user or the machine confirmed after the fact.

**Decision:** training runs will use an external/cloud GPU (Google Colab, or
similar) rather than sustained local training on this hardware. Local hardware
remains fine for short inference-only smoke tests (e.g. the original YOLO detection
test), which do not place the same sustained power/thermal load as a full training
run.

**Status:** decided; closes the "compute budget for CV training/inference" open
item below as far as *where* training runs, though a numeric cost/budget figure for
Colab usage (free tier vs. paid compute units) is still not set.

---

## Still open (not decided as of 2026-09-18)

- Primary video/CV source for the full research build, following loss of DFL Kaggle
  access (see above).
- Resolution of the video/market temporal mismatch (decouple vs. period-matched odds
  vs. forward-looking live capture).
- SoccerNet NDA text: not yet requested/reviewed.
- Market data redistribution terms (Kalshi/Polymarket/Betfair/football-data.co.uk):
  not yet reviewed.
- Gambling-advice/publication regulatory exposure for the applicable jurisdiction:
  not yet resolved.
- Numeric maximum drawdown tolerance: not yet set.
- Numeric compute budget/cost ceiling for cloud GPU usage (Colab or similar): not yet
  set, though the local-vs-cloud question itself is now decided (see above).
- Git repository not yet initialized; `prereg/`, `factor_log.md`, and
  `data_dictionary.md` not yet created.
- Out-of-sample holdout set not yet locked.
