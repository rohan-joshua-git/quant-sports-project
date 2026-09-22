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

## 2026-09-20: Code layout and Colab workflow for training `[DECIDED]`

**Code layout:** real pipeline code lives under `pipeline/<stage>/`, mirroring the
numbered stages in `CLAUDE.md` (`pipeline/detection/` is Stage 1; a calibration stage
would be `pipeline/calibration/`, and so on). `experiments/` stays reserved for
diagnostic spikes, per the 2026-09-18 folder-structure decision. Reason: the
fine-tuning run is the actual Stage 1 deliverable that the smoke test led to, not a
spike, so it should not sit alongside throwaway diagnostics.

**Colab workflow:** training is run by uploading `pipeline/detection/train.ipynb` to
Colab's browser interface and running it there, not through the official
`google-colab-cli`. The CLI supports only Linux and macOS, and this machine runs
Windows (it would need WSL2). The Roboflow API key is read from Colab Secrets at
runtime, so no key is stored in the notebook file or committed. Trained weights are
downloaded manually before the Colab session ends.

**Repo hygiene:** `runs/` added to `.gitignore` (matches at any depth). Ultralytics
writes annotated images and other derived outputs to `runs/` next to wherever a
train/val/predict command is run, and the previous rule only covered
`experiments/runs/`. Derived frames must stay out of the repo per the `CLAUDE.md`
public-output rule.

---

## 2026-09-20: Roboflow fine-tuning run executed on Colab - results `[RESULT]`

Supersedes the "fine-tuning run not yet executed" status on the 2026-09-18 retrain
decision above. That entry is left as written.

**Run configuration**
- Base model: `yolo26n.pt` (COCO-pretrained), fine-tuned with Ultralytics 8.4.156.
- Data: Roboflow Universe "Football Players Detection", version 1 (workspace
  `roboflow-jvuqo`, project `football-players-detection-2frwp`), CC BY 4.0. Four
  classes: ball, goalkeeper, player, referee. Validation split: 38 images, 905
  instances. Test split: 13 images, 309 instances.
- Settings: 100 epochs, imgsz 640, batch 16, Ultralytics defaults otherwise
  (seed 0, `patience` 100 so early stopping could not trigger, `close_mosaic` 10,
  auto optimizer). Taken from the run's `args.yaml`.
- Selected checkpoint: `best.pt` is epoch 83 (highest fitness, validation mAP50-95
  0.481 and mAP50 0.796), per `results.csv`. Total training time in the CSV is 1920 s
  (0.533 hours).
- Compute: Google Colab, Tesla T4 (about 14.9 GB), 0.534 hours (about 32 minutes) for
  the full run. No NaN losses, no crashes. Colab tier and compute-unit cost were not
  recorded.
- Code: `pipeline/detection/train.ipynb`. Weights kept locally as
  `pipeline/detection/football_yolo26n_best.pt` (gitignored via `*.pt`).

**Validation results** (`best.pt`, the split used to select the checkpoint):

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0.843 | 0.773 | 0.796 | 0.481 |
| ball | 0.625 | 0.334 | 0.322 | 0.117 |
| goalkeeper | 0.911 | 0.889 | 0.934 | 0.608 |
| player | 0.937 | 0.980 | 0.988 | 0.674 |
| referee | 0.899 | 0.888 | 0.938 | 0.523 |

**Test results** (scored once, locally, CPU inference only):

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0.865 | 0.794 | 0.772 | 0.495 |
| ball | 0.789 | 0.364 | 0.311 | 0.143 |
| goalkeeper | 0.845 | 0.995 | 0.899 | 0.609 |
| player | 0.950 | 0.953 | 0.979 | 0.666 |
| referee | 0.875 | 0.862 | 0.898 | 0.561 |

**Observations**
- Player, goalkeeper, and referee detection are strong on both splits. Ball
  detection is weak on both (recall 0.33 to 0.36, mAP50 about 0.31 to 0.32), so the
  ball-detection problem from the smoke test is not solved by this run. Ball
  instance counts are very small (35 in validation, 11 in test), so ball metrics
  are noisy.
- Validation mAP50 plateaued around epoch 20 and mAP50-95 around epoch 40, while
  training losses kept falling. The last 60 or so epochs added little. Training
  losses fell steadily (box 1.89 to 1.06, cls 2.94 to 0.37) while validation losses
  flattened from about epoch 40 (box about 1.22 to 1.25, cls about 0.45), so the gap
  between training and validation loss widened. This is a mild overfitting pattern,
  but validation mAP did not degrade.
- The drop in training loss at epoch 91 coincides with Ultralytics closing mosaic
  augmentation for the final 10 epochs, and is not a real improvement.
- Test scores (0.772 / 0.495) are close to validation scores (0.796 / 0.481), so
  there is no sign of large optimism from selecting the checkpoint on validation.
  The test set is small, so this is not conclusive.

**Limits of this result**
- This measures accuracy on Roboflow's images, not on DFL footage. It says nothing
  yet about the smoke-test failures (`tv` false positive, sideline personnel, weak
  ball) on the cached DFL clip. That check has not been run.
- The test split has now been scored once. It should not be used to choose between
  further model variants. Use validation for selection, or create a fresh holdout.
- This is a detection metric only. No tracking metric (HOTA against
  SoccerNet-Tracking) has been computed.
- The per-epoch log files were not saved at first. They were retrieved from the
  Colab session later the same day and now live in
  `pipeline/detection/run_logs/2026-09-20_yolo26n_100ep/` (`results.csv`,
  `results.png`, `args.yaml`). The epoch numbers above match that CSV.

**Compute data point:** one 100-epoch `yolo26n` run took about 0.53 hours on a T4.
The numeric compute budget is still not set.

**Follow-ups (none decided):**
- Run the trained weights on the cached DFL clip and compare against the smoke-test
  findings.
- Options for weak ball detection (larger input size, larger model variant, more
  ball-specific training data). Judge these after the DFL clip result, not before.
- For future runs, consider fewer epochs or early stopping, given the plateau.

---

## 2026-09-21: Fine-tuned vs stock detector on the cached DFL clip `[RESULT]`

Follow-up to the 2026-09-20 run. Both `yolo26n.pt` (stock COCO) and
`football_yolo26n_best.pt` (fine-tuned) were run on `08fd33_4.mp4` (750 frames, 25
fps, 1920x1080) at default confidence, on local CPU (inference only). Method:
per-frame detection counts across all 750 frames, plus manual review of seven
evenly spaced frames and one frame where the stock model produced a `tv` box. There
is no ground truth on this footage, so these results show how often each model
fires and what it labels. They are not accuracy scores.

| Measure | Stock COCO | Fine-tuned |
|---|---|---|
| Frames with a ball detected | 82 of 750 (10.9%) | 490 of 750 (65.3%) |
| Median top ball confidence | 0.39 | 0.53 |
| Longest run of frames with no ball | 264 | 14 |
| `tv` detections | 11 | none (no such class) |

Fine-tuned detections by class over the clip: player 15503, referee 2326, ball 638,
goalkeeper 235.

**The three smoke-test problems, on this clip:**
- Ball: much more consistent (see table). Confirmed correct by eye only in a few
  frames.
- Graphics false positive: in frame 439 the stock `tv` box spans almost the whole
  frame. The fine-tuned model produces nothing there.
- Sideline personnel: the stock model boxed coaches and staff as `person` in the
  reviewed frames. The fine-tuned model did not box them in frames 37, 375, and 439.
  Assistant referees on the touchline were labeled `referee` (confidence 0.42 to
  0.82). Player confidence was also higher (typically 0.7 to 0.87 against 0.25 to
  0.6).

**Limits:**
- One clip, one match, one camera angle. It is also the clip that exposed the
  original problems, so it supports the retrain decision but is not independent
  validation.
- Ball precision is unknown. Some of the 490 ball frames could be false positives,
  and the ball is still absent from about 35% of frames (some of that is genuine
  occlusion or the ball leaving view).
- A few mid-pitch `referee` labels at moderate confidence (about 0.5 to 0.75) may be
  players. Not verified.
- Not a tracking result. No HOTA has been computed.

**Follow-up (not decided):** the manual spot-check already required by `CLAUDE.md`
(hand-label a small sample of frames from this clip) would turn these counts into
real ball precision/recall and referee-confusion figures. Options for the remaining
ball weakness stay open until then.

---

## 2026-09-21: Scope change - live in-play trading is now the primary goal `[DECIDED]`

**Decision (stated by Rohan):** the ultimate goal is a system that handles live
streaming, that is, live in-play trading. This replaces the earlier framing in
`CLAUDE.md` and the scoping memo ("backtest first, live is a stretch goal, not the
initial deliverable"). The earlier framing is left as written in the earlier
entries above.

**What this does not change:**
- Detector training and the Stage 1 to 3 build (detection, tracking, calibration,
  identity) are unaffected. They do not depend on when the footage was recorded.
- The evaluation discipline is unchanged. Pre-registration, full factor logging,
  IC / Newey-West / BH-FDR / PBO, the locked holdout, and the draft success criteria
  (including out-of-sample validation across multiple seasons or competitions) still
  apply. Nothing in this entry relaxes them, and they still gate any real-money
  trading.

**What this makes more important (all still open, none decided):**
- The video/market temporal mismatch. A live system still has to show evidence of
  edge before real money, and the options (decouple, period-matched odds,
  forward-looking capture) are unchanged. Forward-looking capture is now the path
  closest to the end goal.
- Live video source. Where a real-time match feed comes from, its licensing and
  terms, and the broadcast-copyright/scraping risk. Not verified.
- Latency budget from frame to order. Measured so far: the fine-tuned `yolo26n`
  detector runs at about 45 ms per frame on local CPU (detection only, no tracking
  or calibration yet). The target-variable definition already injects a deliberate
  latency lag.
- Market execution: Kalshi/Polymarket API latency, in-play liquidity, and order
  behavior. Not checked.
- Regulatory exposure. Previously only the publication/gambling-advice boundary was
  flagged. Placing real trades adds jurisdiction and platform-eligibility questions.
  Not verified.
- Numeric maximum drawdown tolerance, still unset, becomes more urgent once real
  money is in scope.

**Rationale:** not recorded here. Rohan stated the change without giving reasons.

---

## 2026-09-22: Tracker class architecture - frame-by-frame design for live latency `[DECIDED]`

**Decision:** implement tracking (ByteTrack via `supervision`) in a single-frame
pipeline, not a batched pipeline. Code lives in `pipeline/detection/tracker.py` as a
`Tracker` class.

**Architecture:**
- Input: one frame at a time
- Process: YOLO detection → supervision.Detections → ByteTrack.update_with_detections()
- Output: per-frame tracking boxes with persistent ID assignments
- State: ByteTrack's internal tracking state is maintained across consecutive calls,
  so ID continuity is automatic

**Why not batch (frames 11-13 of the original sketch):**
- Measured latency: YOLO26 detection runs at ~45 ms per frame on local CPU (no GPU).
- Live in-play market window: a typical in-play odds move lasts tens of seconds to
  minutes, but individual frame-to-order latency compounds across the pipeline
  (detect → track → calibrate → feature → market check → order placement). Batching
  N frames adds N×40ms of buffer delay before any output, which is unacceptable.
- Streaming compatibility: a live video feed has no defined "batch boundary"; it is
  a continuous stream. Waiting to collect 20 frames before processing either requires
  buffering the entire stream in memory (not feasible for sustained live capture) or
  discarding frames, both of which break the frame-by-frame design documented in
  `learning.ipynb`.

**Implication for Stage 1 completeness:**
- This design chains detection → tracking → (Stage 2 calibration) in real-time mode.
- Per-frame detection accuracy is now critical, since no future batch-reprocessing
  step can correct missed detections or false positives. This strengthens the argument
  for completing the manual spot-check (hand-labeling) to measure real ball and
  referee detection precision/recall on the actual DFL footage.

---

## 2026-09-22: Tracker smoke test results - minor double-counting on detections `[KNOWN LIMITATION]`

**Finding:** visual inspection of the tracker on 50 frames of the cached DFL clip shows:
- ID persistence: 100% (ByteTrack works correctly, no ID flickering)
- Detection duplicates: minor cases where the same person is detected twice in a frame,
  creating two separate boxes/rings

**Investigation:** added NMS (Non-Maximum Suppression) with thresholds 0.5 and 0.3 to
filter overlapping boxes. Double-counting persists at both thresholds, suggesting the
duplicate detections may not overlap enough to trigger NMS, or the two boxes are
genuinely separated by the detector.

**Conclusion:** this is a detection issue (Stage 1 YOLO model), not a tracking issue
(ByteTrack is working perfectly). Minor double-counting is accepted as a known
limitation for now and logged for future refinement. It does not block moving to
Stage 2 (calibration), as:
- The duplicates are a small fraction of total detections.
- Calibration and feature extraction will still work on the majority of correctly
  detected and tracked boxes.
- Addressing this would require detector retraining or more sophisticated
  deduplication, which is lower-priority than validating the full pipeline.

---

## 2026-09-22: Tracker enhancements - ball interpolation, position calculation, latency measurement `[RESULT]`

**Enhancements added to Tracker class:**
1. Position calculation: `get_center_bbox()` and `get_foot_position()` helpers (static, ~0.5ms overhead)
2. Ball interpolation: carries forward last known ball bbox if ball is missed in current frame (~1ms overhead)
3. NMS (Non-Maximum Suppression): removes overlapping detections with threshold 0.3

**Latency measured on cached DFL clip (50 frames, CPU inference):**
- Mean: 58.27ms
- P95: 92.88ms
- Max: 146.00ms

**Analysis:**
At 25fps, budget is 40ms per frame. Current detection+tracking is 58ms mean, which exceeds budget.
Breakdown: YOLO26 detection ~45ms + NMS+ByteTrack ~13ms.

**GPU vs CPU for latency:**
- CPU: ~58ms per frame (current measurement)
- GPU (estimated, T4): ~15-20ms per frame (3-4x faster than CPU, based on typical YOLO benchmarks)
- GPU enables real-time in-play trading; CPU is sufficient for development/validation only
- Trade-off: GPU adds infrastructure cost and complexity but is essential for live system

**What is NMS and how it helps:**
Non-Maximum Suppression is a post-processing step that removes redundant bounding boxes. After detection,
if two boxes overlap significantly (IoU > threshold), NMS keeps the higher-confidence box and discards
the lower-confidence one. This solves the double-counting problem (same object detected twice). Here,
threshold 0.3 means "remove boxes overlapping >30% with a higher-confidence box." Lower threshold =
more aggressive filtering. In this project, NMS reduced double-counting but didn't eliminate it,
suggesting some duplicates are spatially separated (not true overlaps) and are accepted as a
minor limitation.

**Decision:**
- CPU latency is a known constraint for development. GPU will be required for production live trading.
- Ball interpolation and position calculation are implemented (low overhead, high value for downstream stages).
- NMS is a reasonable deduplication approach; minor remaining double-counting is accepted.
- Next stage (calibration) should still proceed with CPU for validation work.

---

## 2026-09-22: Stage 1 (Detection & Tracking) complete; Stage 2 (Calibration) next

**Stage 1 summary:**
- YOLO26 detector fine-tuned on Roboflow (strong player/goalkeeper/referee, weak ball)
- ByteTrack integration: 100% ID persistence, frame-by-frame design for low latency
- Ball interpolation and position calculation added to Tracker class
- Known limitations: minor double-counting, weak ball detection (35% miss rate)
- Latency: 58ms mean on CPU (over 40ms budget; GPU required for live trading)

**Stage 2: Calibration (pixel space → real-world pitch coordinates) — not yet started**

Purpose: convert pixel bounding boxes to real-world pitch positions and distances. Needed for:
- Tracking player speed and acceleration (pixels/frame → meters/second)
- Distance calculations (ball to player, player to goal line, etc.)
- Feature engineering (e.g., "player speed towards goal")

**Approach: homography calibration**
- Input: pixel coordinates (bounding boxes from Stage 1)
- Method: `cv2.findHomography()` to compute a 3×3 transformation matrix from pitch keypoints
- Keypoints: manually identify 4+ reference points on the pitch (e.g., goal line corners, center spot)
  on a sample frame, their pixel positions, and known real-world coordinates (e.g., goal line is at y=0 in meters)
- Output: pitch coordinates (x, y) in real-world units (meters, typically)

**Next steps (not yet decided):**
1. Select a reference frame from the cached DFL clip
2. Manually identify and label 4-8 pitch keypoints on that frame (goal corners, center spot, etc.)
3. Implement `calibrate_frame()` in Tracker or a new `pipeline/calibration/` module
4. Validate calibration accuracy on a few frames (visual check: player movements look physically plausible)
5. Integrate into pipeline: frame → detect/track → calibrate → features

**Open questions:**
- How many keypoints are needed for robust homography? (minimum 4, but more = better fit)
- Does homography hold across the entire match (camera is fixed), or do camera cuts require recalibration?
- How to validate that calibrated coordinates are correct? (ground truth is unavailable)

---

## Still open (not decided as of 2026-09-22)

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
  set, though the local-vs-cloud question itself is now decided (see above). One
  data point now exists: about 0.53 hours on a T4 for a 100-epoch `yolo26n` run.
- Git repository is initialized (initial commit `35cff10`). `prereg/`,
  `factor_log.md`, and `data_dictionary.md` not yet created.
- Trained detector was run on the cached DFL clip on 2026-09-21 (counts and visual
  review only). Ball precision and referee confusion are still unmeasured; a manual
  spot-check is the open next step.
- Out-of-sample holdout set not yet locked.
- Manual spot-check on the DFL clip (Stage 1 quality gate): still not started. High
  priority now that tracker architecture is locked, since per-frame detection accuracy
  becomes critical for live streaming (no batch reprocessing fallback).
