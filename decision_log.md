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

**Stage 2: Calibration (pixel space → real-world pitch coordinates): not yet started**

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

## 2026-09-22: Tracker review findings - open bugs in Stage 1 `[OPEN]`

A code review of `pipeline/detection/tracker.py` and
`experiments/tracker_smoke_test.ipynb` found problems that qualify the "Stage 1
complete" entry above. That entry is left as written; this records what was found
afterwards.

- Interpolated ball is assigned `class_id = 2`, which is `player` in the dataset's
  class order (`ball, goalkeeper, player, referee`). Ball is class 0.
- Interpolated ball uses a hardcoded `tracker_id = 1`, which can collide with a real
  ByteTrack ID.
- Carry-forward has no maximum age, so a ball that leaves the frame stays at its last
  position indefinitely.
- The visual-check cell draws `detection_raw` (raw YOLO output, before NMS and
  ByteTrack), not `tracks`. The earlier findings "100% ID persistence" and "NMS did
  not remove duplicates" were therefore not checked on tracker output.
- The "persistence rate" metric counts IDs seen in 2+ frames. It cannot detect ID
  switches, which is the failure that matters.
- `with_nms()` is class-aware by default, so one person detected as both `player` and
  `referee` keeps both boxes. `class_agnostic=True` is worth testing.
- The same `Tracker` instance is reused across cells without a reset.
- The latency split (YOLO about 45 ms, NMS + ByteTrack about 13 ms) and the GPU figure
  (15 to 20 ms) are estimates, not measurements. The mean includes model warmup.

Separately, the manual hand-label spot-check that CLAUDE.md treats as the Stage 1
quality gate has not been done. Stage 1 should be read as "built, not yet validated".

---

## 2026-09-22: Research plan for quantitative methods, and the edge hypothesis `[DECIDED]`

**Context:** a review of which quantitative finance techniques fit this project, done
with input from an external LLM (whose citations were partly wrong and were checked
by web search) and a follow-up review here. The external model declined to cover
market-side, execution and sizing material; that part was done separately.

**Key finding that shapes the plan:** Croxson & Reade (2014, Economic Journal,
DOI 10.1111/ecoj.12033) show soccer betting prices absorb goals swiftly and fully.
Broadcast video lags the live event by seconds or more, and market makers price from
faster official data feeds. So any edge from this project cannot come from detecting
events (goals, cards, shots) before the market. The working hypothesis is instead
**state estimation**: slow-moving tactical and physical state that is not in event
feeds (sustained pressure, territorial dominance, shape changes after substitutions,
fatigue proxies) may not be fully priced. Fees also set a high bar: Kalshi's taker
fee is reportedly about 0.07 x P x (1 - P) per $1 contract (1.75 cents at a 50 cent
price), per third-party summaries not yet checked against Kalshi's own fee schedule.

**Hypothesis ladder (to be written into `prereg/`):**
- H0: tracking-derived factors add no information over a conventional baseline
  (score, time remaining, pre-match team strength).
- H1: tracking factors add information about outcomes over the next few minutes.
- H2: the added information exists only in particular latent match states.
- H3: the added information survives realistic video latency.
- H4: the effect holds across seasons and competitions.

A negative result under this design is still a valid, reportable result.

**Plan, four parallel tracks feeding one final test:**

Track 1, finish Stage 1 properly (now):
1. Fix the tracker bugs logged above, then redo the visual check on tracked output
   with ID labels.
2. Replace the carry-forward ball with a Kalman filter (predicted position plus
   uncertainty during misses).
3. Hand-label a sample of DFL frames to measure detector error rates. These rates
   are reused later for the CV-error perturbation test.

Track 2, market event study (no video needed):
1. Check Kalshi and Polymarket data terms of service first.
2. Pull 2024+ in-play soccer price history. Measure speed and completeness of price
   reaction to goals and red cards (Croxson & Reade method). Aligning video goal
   times with market jumps also measures broadcast delay directly.
3. Test market-price calibration (including favorite-longshot bias) and profile
   spreads, depth and fees by match minute and price level.

Track 3, outcome baseline (no video needed):
1. Dixon-Coles pre-match team strength, with hierarchical (partial-pooling) shrinkage
   toward the league average. Licensing of the historical results source must be
   checked first.
2. Dixon-Robinson in-play goal hazard, updated by score and time remaining.
3. Monte Carlo of the remaining match for P(win/draw/loss) at any minute. This is the
   baseline every CV factor must beat.

Track 4, pre-registration (before any feature is tested):
1. Power analysis first: how many matches are needed to detect rank-IC of 0.02 at
   Newey-West t of at least 2. If the answer exceeds the data available, the success
   criteria must be revisited before building further.
2. Resolve the video source.
3. Write the H0 to H4 ladder into `prereg/`, with a small factor family (5 to 10 slow
   state factors, including the forward-only HMM state).
4. Lock the holdout set and a latency delay grid (2, 5, 10, 30 seconds).

Final test (after Stages 2 to 4: calibration, GMM team assignment, features):
1. Neutralize each factor against the baseline probability (keep the residual), then
   test incremental information: IC and IC decay by horizon (1, 5, 15, 30 minutes),
   match-clustered standard errors, Diebold-Mariano forecast comparison clustered by
   match, BH across the full family, combinatorial purged CV, PBO, Deflated Sharpe
   Ratio.
2. Robustness: CV-error perturbation using Track 1's measured error rates, and the
   latency grid.
3. Only if the signal survives: latency-injected market replay against tradeable
   order-book prices, trading only when edge exceeds fee plus half-spread plus a
   buffer, markout analysis for adverse selection, shrunk Kelly (Baker & McHale 2013)
   with one allocation per match and per-match caps, and a drawdown halt rule set from
   a match-block bootstrap of backtest P&L (this also supplies the open numeric
   max-drawdown tolerance).

**Methods demoted or rejected:**
- EGARCH and Markov-switching on market log-odds: demoted from core to descriptive at
  most. Bounded prices, event jumps, suspensions and short histories make them a poor
  fit, and they risk just finding "pre-goal / post-goal" regimes. This reverses an
  earlier recommendation made in conversation.
- Markowitz optimization: rejected. Binary payoffs break mean-variance assumptions
  and covariance estimates would be noise at this sample size. Kelly variants replace
  it.
- Technical indicators, deep order-book models, reinforcement learning for
  execution, and market making as a strategy: rejected (no mechanism, too data
  hungry, or adverse selection for a slow trader).
- LLM techniques and blockchain logic: out of scope for the research pipeline.
- Pitch control, Voronoi and passing-lane features: deferred. Broadcast video usually
  shows only part of the pitch, so features needing all 22 players are unreliable.
  Ball-local features are preferred.

**Why this structure:** Tracks 2 and 3 do not depend on the unresolved video source,
so work can progress while that blocker stays open. Track 2 also tests whether the
live-trading premise can work before months go into the full pipeline.

**Key references (checked by search on 2026-09-22):** Dixon & Robinson (1998),
JRSS D 47(3), DOI 10.1111/1467-9884.00152; Angelini, De Angelis & Singleton (2022),
IJF 38(1), DOI 10.1016/j.ijforecast.2021.05.012; Baker & McHale (2013), Decision
Analysis 10(3), DOI 10.1287/deca.2013.0271; Whitrow (2007), JRSS C 56(5),
DOI 10.1111/j.1467-9876.2007.00594.x; Arian, Norouzi Mobarekeh & Seco (2024),
Knowledge-Based Systems 305, DOI 10.1016/j.knosys.2024.112477. Dixon & Coles (1997)
and Bailey & Lopez de Prado (2014, 2017) are cited from memory and should be checked
before being relied on. Citations from the external model found to be wrong: a
Spearman (2018) "Markov model for the value of defending" paper (not found), and
incorrect author lists for SoccerNet-Tracking (actual first author Cioppa, CVPRW 2022)
and SoccerNet Game State Reconstruction (actually Somers et al., CVPRW 2024,
arXiv 2404.11335).

---

## 2026-09-25: Tracker review follow-up - fixes applied, new problems found `[RESULT]`

Follow-up to the 2026-09-22 review entry above, which is left as written.

**Status of each review item:**
- Interpolated ball `class_id`: fixed. The ball class is now looked up by name from
  `model.names`, not hardcoded.
- Hardcoded `tracker_id = 1`: fixed. The interpolated ball uses `-1`, which ByteTrack
  never assigns (its IDs start at 1).
- No maximum age on carry-forward: fixed. `max_ball_miss` (default 5 frames, 0.2 s at
  25 fps); after that the stale box is dropped.
- Visual check drew raw detections: fixed. New `annotate_tracks()` in
  `pipeline/common/drawing.py` draws tracker output with ID labels under each ring,
  and draws a carried-forward ball as an outline so a guess never looks like an
  observation. The new video has not been reviewed by eye yet.
- Persistence metric: replaced by track-break (fragmentation) statistics. It still
  cannot detect ID switches; that needs ground truth.
- Class-aware NMS: now a constructor setting (`nms_class_agnostic`), default still
  `False`. Not tested yet.
- No reset between cells: fixed. New `reset()` method, called before each rerun.
- Latency figures: frame 0 (model warmup) is now excluded. Remeasured on 50 frames:
  mean 47.04 ms, P50 43.90 ms, P95 64.23 ms, max 120.82 ms. This supersedes the
  58.27 ms mean from 2026-09-22. Still over the 40 ms budget. The split between YOLO
  and NMS + ByteTrack is still unmeasured.

**Track-break statistics, frames 0 to 99 of `08fd33_4.mp4`:** 24 objects on frame 0,
30 unique IDs issued, 6 new IDs after frame 0, 2 tracks shorter than 5 frames, 16 IDs
present in all 100 frames, median track lifetime 100 frames. These counts may include
the `-1` interpolated-ball ID, which slightly inflates "new IDs". In the 50-frame
visual check, 5 frames used a carried-forward ball.

**New problems found in the 2026-09-25 read-through:**
1. The ball goes through ByteTrack before the code looks for it. In `supervision`
   0.30.4, ByteTrack only starts a new track at confidence 0.35 or higher
   (`track_activation_threshold` 0.25 plus 0.1). Detections between 0.1 and 0.25 are
   only used to extend an existing track by box overlap, which a small, fast ball
   often fails. So many ball detections are discarded even though the detector runs
   at `conf=0.1`. It also means the ball-resolution test (not yet run) would measure
   detector plus ByteTrack, not the detector alone as its notebook says.
2. `sv.ByteTrack` is deprecated and will be removed in `supervision` 0.31. The smoke
   test notebook installed `supervision` without a version pin.
3. The notebooks ran on the global Python 3.14 interpreter, not `.venv`. Package
   versions match except torch: global is a CPU build, `.venv` is a CUDA build
   (`2.14.0+cu130`). All latency figures so far are CPU-only.
4. With more than one ball detection in a frame, the first in array order was kept,
   not the most confident.

**Decision: tuning / spot-check frame split `[DECIDED]`.** From now on, frames 0 to
299 of `08fd33_4.mp4` are used for tuning (inference size, NMS setting, ball filter
settings). Frames 300 to 749 are held back for the hand-label spot-check, so error
rates are not measured on the frames the settings were tuned on. Limit: this split
is not pristine. The 2026-09-21 comparison ran both models over all 750 frames and
frames 375 and 439 were reviewed by eye. No setting was tuned on them, but they
have been seen.

**Plan for the rest of Track 1** (steps, not yet done): pin the environment; take
the ball out of ByteTrack and pick it straight from detector output; run the
resolution test and a per-stage timing breakdown; test class-agnostic NMS; replace
carry-forward with a forward-only Kalman filter; redo the visual check; hand-label
spot-check on frames 300 to 749.

---

## 2026-09-25: Ball taken out of ByteTrack `[DECIDED]` `[RESULT]`

**Change:** in `Tracker.track_frame`, detections are split by class. Players,
goalkeepers and referees go through NMS and ByteTrack as before. The ball skips
both: the most confident ball detection at or above a new `ball_conf` setting is
taken straight from the detector and appended with `tracker_id = -1`. A new `device`
setting (default `"cpu"`) keeps inference off the local GPU unless chosen.

**Result, frames 0 to 299 of `08fd33_4.mp4`** (tuning range, `conf` and `ball_conf`
0.1, imgsz 640, CPU):

| Approach | Frames with a ball in tracker output |
|---|---|
| Old: ball through ByteTrack | 122 of 300 |
| New: ball straight from detector | 261 of 300 (equals the raw YOLO count) |
| New, plus carry-forward (`max_ball_miss=5`) | 291 of 300 (30 carried forward) |

ByteTrack was discarding more than half of the frames where the detector saw a ball.
Every earlier ball figure taken from tracker output undercounts detection. The
2026-09-21 detector comparison (65.3% of frames) counted raw detections, so it is
not affected.

**Limits:** these are counts, not correct detections. Ball precision is unknown
until the hand-label spot-check. The ball now has no identity across frames beyond
the carry-forward; the Kalman filter (Track 1 step 6) replaces that.

---

## 2026-09-25: Inference resolution stays at 640; latency breakdown measured `[DECIDED]` `[RESULT]`

Run in `experiments/ball_resolution_test.ipynb` on frames 0 to 299 (tuning range),
CPU, `conf` and `ball_conf` 0.1, carry-forward off. `Tracker` now records per-stage
timings (`last_timings`).

| imgsz | Frames with ball | Mean ms | P95 ms | YOLO ms | ByteTrack ms |
|---|---|---|---|---|---|
| 640 | 87.0% | 42.9 | 51.2 | 35.9 | 6.1 |
| 960 | 92.3% | 62.5 | 66.6 | 55.7 | 6.0 |
| 1280 | 92.0% | 100.9 | 109.2 | 94.0 | 6.1 |

**Decision: keep imgsz 640.** Higher resolution adds about 5 points of ball coverage
at 960 (nothing more at 1280) for 20 ms or more per frame. Downscaling is not the
main cause of weak ball detection.

**Latency:** YOLO is about 84% of per-frame time. Split plus NMS is about 0.7 ms,
ByteTrack about 6 ms, ball handling about 0.1 ms. This replaces the 2026-09-22
estimate of about 13 ms for NMS plus ByteTrack. The pipeline is 42.9 ms mean at 640
on CPU, just over the 40 ms budget, so any speed-up has to come from the detector or
hardware.

**Multiple ball detections:** 157 of 300 frames have two or more ball boxes at 0.1
or higher. The top two are a median 3 px apart, so most are duplicate boxes on the
same ball (the ball no longer goes through NMS), handled by keeping the most
confident. About 10% are over 700 px apart, which are genuine false balls, left for
the Kalman gate.

**Ball gaps at 640:** 8 gaps, longest 9 frames (0.36 s), median 5. This sets the
Kalman filter's bridging target at about 10 frames.

**Limit:** counts, not correct detections. Ball precision is still unmeasured.

---

## 2026-09-25: Cross-class duplicate removal via a second NMS pass at IoU 0.7 `[DECIDED]` `[RESULT]`

**Test** (frames 0 to 299, 640, CPU): with the original class-aware NMS, 24 frames
had a pair of people boxes of different classes overlapping at IoU above 0.3. All 24
were checked by eye from crops:
- 21 were one referee boxed twice, as `referee` and as `player`, IoU 0.76 to 0.97.
  Genuine duplicates. They inflate the player count and can give one person two IDs.
- 3 (frames 248 to 250) were two different people, a referee directly in front of a
  player, IoU 0.30 to 0.56. Plain class-agnostic NMS at 0.3 would delete a real
  person there.

**Decision:** keep class-aware NMS at 0.3, then run a second, class-agnostic NMS at
IoU 0.7 (`cross_class_nms_threshold`, replaces the `nms_class_agnostic` flag). With
it, cross-class pairs drop from 24 to 3 (the three real ones), people boxes per
frame from 22.27 to 22.18, unique IDs from 40 to 39. Short tracks unchanged (2).

**Limits:** the 0.7 threshold sits in the gap between 0.56 and 0.76 seen on 24 pairs
from one clip, so it is a tuned value, not a general one. Two real people
overlapping above 0.7 would lose one box. Same-class NMS at 0.3 can already merge
two overlapping teammates; that is unchanged and unmeasured.

---

## 2026-09-25: Kalman filter replaces ball carry-forward `[DECIDED]` `[RESULT]`

**Change:** new `pipeline/detection/ball_filter.py` (`BallKalman`), a
constant-velocity Kalman filter on the ball centre in pixels (state x, y, vx, vy),
forward-only so it is point-in-time correct and live-compatible. In
`Tracker._update_ball`:
- Each frame the filter predicts. Ball detections at or above `ball_conf` are
  accepted only inside a gate (squared Mahalanobis distance 9.21 or less, the 99%
  point for 2 degrees of freedom); the most confident one inside the gate updates
  the filter.
- If the filter has missed for 3 or more frames and a ball appears outside the gate,
  the filter restarts on it (a kick it could not follow).
- A missed frame reports the prediction (`interpolated=True`) with its uncertainty
  in `tracks.data['ball_sigma']` (pixels). After `max_ball_miss` misses (now 10, set
  from the longest 9-frame gap at 640) the ball is dropped.
- A detected ball is reported at its detected box, with `ball_sigma` 0.

**Tuning** (frames 0 to 299 only, detections cached): no ground truth, so each
setting was scored by predicting 1 to 10 frames ahead from every detected frame and
comparing with confident (0.35 or higher) later detections. Median error in pixels:

| Frames ahead | Kalman (accel 4, meas 2) | Old carry-forward |
|---|---|---|
| 1 | 1.4 | 4.6 |
| 3 | 3.6 | 13.6 |
| 5 | 6.4 | 22.5 |
| 9 | 13.8 | 40.1 |

About 3 times more accurate at every horizon. Results were nearly flat across a grid
of process noise 1 to 16 and measurement noise 1 to 4, so the defaults (4 and 2)
were kept rather than picking the grid's best cell.

**Gate check by eye:** 3 detections were rejected. Two (frames 259 and 260,
confidence 0.43 and 0.52, 426 px from the prediction) were a green player's boot, a
confident false positive; the filter correctly kept predicting the real ball,
stationary at the goalkeeper's feet. That real ball was clearly visible in frames
259 to 262 and not detected at all, a recall failure on a stationary ball.

**Smoke test after all 2026-09-25 changes** (frames 0 to 99): 26 unique IDs (was
30), 3 new IDs after frame 0 (was 6), 17 IDs lasting all 100 frames (was 16). Over
frames 0 to 299, 42 frames show a predicted ball. Latency varied between 44.6 and
53.9 ms mean on two runs of the same code, so single latency figures on this laptop
carry roughly 10 ms of run-to-run noise.

**Limits:** pixel-space velocity includes camera panning (Stage 2 homography fixes
this). A false ball that persists 3 or more frames while the real one is unseen can
capture the filter. Tuning used one clip.

**Settings frozen for the spot-check** (Tracker defaults as of this entry): conf
0.1, imgsz 640, NMS 0.3 class-aware then 0.7 cross-class, ball_conf 0.1, Kalman
accel 4 / meas 2, gate 9.21, reacquire after 3, max_ball_miss 10, CPU. Frames 300
to 749 have not been used for any of this.

---

## 2026-09-25: Visual review of tracker output - ID switches on overlap `[RESULT]` `[OPEN]`

Rohan watched `experiments/runs/tracker_visual_check.avi` (frames 0 to 299, all
2026-09-25 changes in). Only finding: when players overlap, typically while
contesting the ball, their track IDs sometimes swap. Frame numbers and frequency not
recorded; not yet counted.

**Why it happens:** ByteTrack matches by box position and overlap only, so it cannot
tell overlapping players apart.

**Considered: shirt colour.** The reference repo (`abdullahtarek/football_analysis`)
was checked: its tracker does not use colour; it assigns team colour after tracking
and caches it per track ID, so a swap carries the wrong team forward. Colour can
prevent swaps between opposing players, not between teammates. For the planned
team-level factors only cross-team swaps change the result, since a team total is
unchanged by two teammates swapping IDs. Candidate design (not built): per-frame team
label from shirt colour, one tracker per team, and a running vote per ID so a sudden
team flip flags a likely switch. This also means the research plan's "one fit per
track ID, then reuse" for team assignment should become per-frame labels with a vote.

**Next:** count switches (and how many cross teams) as part of the hand-label
spot-check, then decide whether team-split tracking is built now or in Stage 3.

---

## 2026-10-05: Hand-label spot-check of the Stage 1 tracker - results `[RESULT]`

**Method.** `experiments/spotcheck.py`, run on frames 300 to 749 of `08fd33_4.mp4`
(held out from all tuning). Frozen tracker settings from commit `f67433a` (as listed
in the 2026-09-25 Kalman entry), run from frame 0 so the tracker reached frame 300
warmed up. Rohan labelled blind: the ball labeller never showed the tracker's ball,
and the people labeller showed boxes but not predicted classes.
- Ball: 100 evenly spaced frames, ball centre clicked at full resolution (zoomed).
  96 visible, 1 not visible, 3 unsure (excluded).
- People: 20 frames, true class of all 452 tracker boxes, plus clicks on real people
  with no box.
- ID switches: the 450-frame ID video reviewed, each error marked.
Labels and frames are in `experiments/spotcheck/` (gitignored); scores in its
`results.json`.

**People (strong):**

| Measure | Result | 95% Wilson interval |
|---|---|---|
| Box is a real person (precision) | 451/452 = 99.8% | 98.8 to 100% |
| Real person got a box (recall) | 451/464 = 97.2% | 95.3 to 98.4% |
| Class correct, as labelled | 443/451 = 98.2% | 96.5 to 99.1% |

Referee boxed as player: 0; player boxed as referee: 1. The referee double-boxing
fixed by the 2026-09-25 cross-class NMS pass did not recur. Label review: 5 boxes on
one track (ID 60) were labelled `player` but the crops show the orange-kit
goalkeeper in the goalmouth wearing gloves, so the model's `goalkeeper` was right. If
those 5 labels are corrected, class accuracy is 448/451 = 99.3%; the label file was
not changed (Rohan has not confirmed). The black-kit goalkeeper (ID 27) was called
`player` twice by the model, which the labels catch correctly.

**Ball (weak):**

| Measure | Result | 95% Wilson interval |
|---|---|---|
| Detector precision (within 10 px) | 52/57 = 91.2% | 81.1 to 96.2% |
| Detector recall (within 10 px) | 52/96 = 54.2% | 44.2 to 63.8% |
| Ball found incl. Kalman predictions, within 10 px | 61/96 = 63.5% | 53.6 to 72.5% |
| Ball found incl. Kalman predictions, within 20 px | 70/96 = 72.9% | 63.3 to 80.8% |
| Truth inside the 2-sigma circle (expected about 95%) | 20/39 = 51.3% | 36.2 to 66.1% |

Median error of correct detections: 2.8 px. Results at 10 and 20 px radius are
identical for the detector: detections are either right or far off.

**Main failure: persistent false ball captured the Kalman filter.** From about frame
617 to 731 (about 4.5 s) the detector repeatedly called one distant green-kit
player's body a ball (confidence 0.1 to 0.39). Those detections kept landing inside
the gate, so the filter never counted a miss, never triggered the reacquire rule, and
ignored the real ball. All 5 wrong detections and most large prediction errors (up
to about 870 px) fall in this episode. The gate handles one-off false balls (the boot
at frame 259 on 2026-09-25) but not a false ball that keeps reappearing. Separately,
`ball_sigma` is overconfident: even before the capture (frames up to 613), the truth
was inside the 2-sigma circle in only 17/23 predicted frames (74%).

**ID errors:** over 18 s, 0 cross-team swaps, 0 teammate swaps, 7 "other" ID errors
(frames 300 twice, 307, 322, 354, 362, 527). Their nature was not recorded. The swaps
on overlap seen in frames 0 to 299 on 2026-09-25 were not marked as swaps here.

**Comparison with tuning frames.** Detector recall here (54%) is far below the share
of tuning frames with any ball detection (87%), but these are not the same measure:
the tuning figure counted detections of anything, and 41 of the 100 held-out ball
frames were Kalman predictions against about 14% in tuning. The held-out stretch is
harder for the detector, and the capture episode adds to it.

**Limits:** one clip, one match, one camera, one labeller who also built the tracker.
Neighbouring frames are correlated, so the intervals are narrower than the truth.
"Within 10 px" and the duplicate rule are definitions. Not a HOTA score; the
SoccerNet-Tracking proxy check is still not done (NDA unread).

**Consequence: frames 300 to 749 are now spent.** Any fix motivated by these results
(for example rejecting ball detections that sit on a player's torso, or preferring a
confident detection outside the gate over weak ones inside it) cannot be validated on
these frames. That would need fresh footage.

---

## 2026-10-05: Stage 1 closed as "validated with known limits" `[DECIDED]`

Stage 1 (detection and tracking) is closed, with the spot-check above as its measured
error rates. No numeric pass threshold was set in advance, which is a process gap:
this is a judgement that the error rates are known and usable, not a pass against a
pre-registered bar. Future stages should set their acceptance criteria before
measuring.

- Player, goalkeeper and referee detection and classification: strong (about 97 to
  100%). Usable for team-level features.
- Ball: detected in about half the frames, and a persistent false positive can
  capture the ball filter for seconds. Ball-dependent features (possession,
  ball-local pressure) must be treated as noisy, and the CV-error perturbation test in
  the final evaluation must use these measured rates.
- Carried forward as open: ball-capture fix (needs fresh footage to validate);
  overconfident `ball_sigma`; ID errors (team-split tracking candidate, see the
  2026-09-25 visual-review entry); SoccerNet-Tracking HOTA (blocked on NDA);
  latency over budget on CPU (GPU needed for live).

Next per the research plan: the Track 4 power analysis, then Track 2 market terms and
fees, then Stage 2 calibration.

---

## 2026-10-05: Power analysis - rank-IC 0.02 is not detectable with archival video `[RESULT]`

Track 4 step 1, in `prereg/power_analysis.ipynb` (simulation only, no real data,
fixed seed). Simulated matches: a standardised AR(1) factor per minute (persistence
phi 0.8 / 0.95 / 0.99), goals at about 1.4 per team per match with team A's rate
multiplied by exp(beta x factor) and team B's by exp(-beta x factor). Targets: goal
difference over the next 5 minutes, next 15 minutes, or rest of match. Test as in
the draft criteria: pooled rank-IC, match-clustered t, success at t of 2 or more.

**Effective information per match.** Minutes are far from independent (slow factor,
overlapping target windows). Effective independent observations per match at phi
0.95: about 18 (5-minute target), 7 (15-minute), 4 (rest of match), out of about 85
minute-rows. Naive standard errors that treat minutes as independent produced 17 to
33% false positives at t of 2 or more under no effect; clustered errors gave 2.5 to
3.4% (nominal 2.3%), confirming the clustering requirement.

**Matches needed for 80% power, phi 0.95 (base case / worst case):**

| Target | IC 0.02 | IC 0.03 | IC 0.05 |
|---|---|---|---|
| Next 5 min | 1,107 / 3,725 | 492 / 1,656 | 177 / 596 |
| Next 15 min | 3,030 / 10,195 | 1,347 / 4,531 | 485 / 1,631 |
| Rest of match | 5,574 / 18,756 | 2,478 / 8,336 | 892 / 3,001 |

Worst case multiplies by about 3.4: BH across 10 factors with one real effect
(1.65), a 30% locked holdout (1.43), and CV measurement error at reliability 0.7
(1.43). A brute-force check of the formula gave 75% simulated power where 80% was
predicted, so the base numbers are, if anything, slightly optimistic.

**In football terms:** a factor that shifts goal rates by about 11% per standard
deviation (beta 0.1) produces a rank-IC of only about 0.03 to 0.04; IC 0.02 is
roughly a 5 to 6% shift. Goals are rare, so even sizeable effects give small ICs.

**Rank-IC is not the most powerful test here.** A direct test of the goal-rate model
(score test, the same idea as a log-loss comparison) needed about 20 to 25% fewer
matches than 5-minute rank-IC (beta 0.1: 321 against 409 matches).

**Against available data:** the largest archive considered, SoccerNet's main corpus,
has about 500 matches (access and usable minutes unverified). With 500 matches the
smallest IC detectable at 80% power is about 0.030 (5-minute target), 0.049
(15-minute), 0.067 (rest of match), in the base case; about 1.8 times larger in the
worst case.

**Conclusion:** the draft criterion "rank-IC of 0.02 to 0.03 at t of 2 or more" cannot
be met with any archival video source identified so far. Data volume, not method, is
now the binding constraint. Per CLAUDE.md, the success criteria must be revisited
before building further.

**Limits:** the goal model is a simplification (independent minutes, log-linear
tilt, full 90 minutes visible; broadcast replays and close-ups lose minutes and make
it worse). Market-price targets were not simulated; they move continuously and may
carry more information per match, which needs Track 2 data to quantify.

## 2026-10-05: Success-criteria revision after the power analysis `[OPEN]`

Recommendations, not yet decided by Rohan:
1. Replace the fixed "IC 0.02" bar with a pre-registered **minimum detectable
   effect**: state the number of matches, the effect size the test has 80% power for,
   and that a null result only rules out effects larger than that.
2. Primary horizon 5 minutes; primary statistic a likelihood-based goal-rate test
   (log-loss comparison against the baseline, clustered by match), rank-IC reported
   as secondary.
3. Keep the factor family small (5 rather than 10) to limit the BH cost.
4. Weigh match **volume** heavily in the video-source decision, and consider whether
   market-price targets (forward-looking phase) offer more power once Track 2 data
   exists.

---

## 2026-10-05: Statistical success criterion revised `[DECIDED]`

Decided by Rohan, following the power analysis and the `[OPEN]` recommendations
entry above (which this resolves, except item 4, the data-source question). The
draft criterion "rank-IC of 0.02 to 0.03 at Newey-West t of 2 or more" is replaced.
Lowering the bar instead (dropping BH, accepting weaker t) was considered and
rejected: loosening the rules after seeing the power numbers is the self-deception
the process exists to prevent.

**New criterion:**
1. **Primary test:** per-minute Poisson goal-rate model. Factor measured at minute
   t-1, goals in minute t, the baseline expected rate (score, time, team strength) as
   an offset; test of the factor coefficient beta with match-clustered standard
   errors.
2. **Success stated as a minimum detectable effect:** before any factor test, fix
   the number of matches and report the goal-rate shift per SD of the factor that the
   test detects with 80% power. A null result only rules out effects larger than
   that.
3. **Rank-IC** at a 5-minute horizon kept as a secondary, reported statistic.
4. **Factor family capped at 5**, BH-FDR at q = 0.05 across the full family.
PBO, the economic criteria and the drawdown method are unchanged.

**Why per minute, not the recommended 5-minute horizon.** The 5-minute
recommendation came from comparing rank-IC horizons. Measuring the likelihood test
directly (`prereg/power_analysis.ipynb`, step 7) showed the per-minute version needs
78% of the matches that non-overlapping 5-minute blocks need (information per match
1.53 against 1.35), because it always uses the freshest factor value. Both had
correct false-positive rates (2.0% and 1.9% against a nominal 2.3%).

**MDE of the primary test** (phi 0.95, goal-rate shift per SD, base / adjusted for
BH across 5 factors, a 30% holdout, and factor reliability 0.7):

| Matches | 100 | 200 | 300 | 500 | 1,000 | 2,000 |
|---|---|---|---|---|---|---|
| Base | 20.4% | 14.0% | 11.3% | 8.7% | 6.1% | 4.2% |
| Adjusted | 37.6% | 25.3% | 20.2% | 15.3% | 10.6% | 7.4% |

**Still open:**
- Holdout confirmation rule. A 30% holdout of 500 matches is 150 matches, where a
  confirmation test at t of 2 or more only has power for large effects. The rule
  must be pre-registered before any factor test.
- The number of matches, which depends on the data-source strategy. Options weighed
  on 2026-10-05 (archival video, forward capture, an event-data signal check first,
  the Track 2 market study as a main result); recommendation was cheap tests first
  (Track 2 plus an event-data check) before the video source and Stages 2 to 4.
  Not decided.
- The MDE assumes the simplified goal model in the notebook; broadcast minutes lost
  to replays and close-ups would raise it.

---

## 2026-10-08: Jurisdiction check - real-money trading is not legal from Singapore `[RESULT]`

Rohan is legally resident in Singapore. Checked on 2026-10-08 (web sources; not legal
advice, no lawyer consulted):
- **Gambling Control Act 2022, s.20(3):** gambling with an unlawful (unlicensed)
  gambling service provider is an offence, fine up to S$10,000, prison up to 6 months,
  or both. Singapore Pools is the only licensed remote gambling operator.
- **Polymarket:** blocked by the Gambling Regulatory Authority (GRA) since
  2025-01-12 as an "illegal gambling site". Polymarket's own geographic-restrictions
  page lists Singapore as close-only (no new positions) and its Terms of Use ban VPN
  circumvention. Confirmed in practice: fetching `polymarket.com/tos` from this
  machine was redirected to an IMDA block page.
- **Kalshi:** Singapore is on the restricted-jurisdiction list in the member
  agreement. The restriction applies to trading event contracts only and "do[es] not,
  in and of [itself], prohibit membership on, or non-trading access to, the
  Platform". Public market-data endpoints need no API key.
- **Betfair Exchange:** not offered in Singapore (unlicensed there).
- **Singapore Pools** offers in-play football betting, but it is a fixed-odds
  bookmaker that sets and varies its own odds at its discretion, not an exchange.
  Not a fit for this research design.
- **Gambling Control Act 2022, s.85:** advertising unlawful gambling is a
  strict-liability offence (fine up to S$20,000). "Advertising" includes informing
  the public of any online location where unlawful gambling takes place. The public
  GitHub repo already names Kalshi and Polymarket as trading venues in `CLAUDE.md`,
  this log and the memo. Whether research text naming a venue counts is not settled
  here.
- **Kalshi fees** (from Kalshi's help page, full PDF schedule not read): taker fee
  `round_up(0.07 x C x P x (1 - P))` per trade, `0.0175` instead of `0.07` on some
  markets; resting (maker) orders are free unless a market lists maker fees; no
  settlement fee.

**Consequence:** the 2026-09-21 goal (live real-money in-play trading) cannot be
pursued while Rohan lives in Singapore. Kalshi is usable as a research data source.
See the next entry.

---

## 2026-10-08: Scope change - backtest plus paper trading replaces live trading as the primary goal `[DECIDED]`

**Decision (Rohan, 2026-10-08):** the primary goal is a rigorous backtest plus paper
trading (simulated orders against live public Kalshi prices, no real money). This
reverses the 2026-09-21 scope change, which is left as written above. No real-money
trading, and no VPN or other workaround of geographic restrictions.

**Rationale:** (1) the jurisdiction check above; (2) Rohan stated the purpose of the
project on 2026-10-08: it is a portfolio and learning project towards becoming a quant
trader. Methodology, honest validation and a finished result serve that goal; real
P&L adds little and is not legally available.

**What this does not change:** the evaluation discipline, pre-registration, the
revised statistical criterion (2026-10-05), PBO and the economic criterion (net edge
after fees and latency, now measured in backtest and paper trading only).

**What this changes:**
- Live-system requirements from 2026-09-21 drop in priority. A latency estimate is
  still needed for the latency-injected backtest and for paper trading, but order
  execution, live video licensing and real-money risk limits are out of scope.
- The frame-by-frame tracker design (2026-09-22) stays: it costs nothing and suits
  paper trading.
- Market platform: Kalshi as data source only. Polymarket dropped (blocked in
  Singapore, VPN use banned by its terms).

**Recorded with this decision (Rohan's answers, 2026-10-08):**
- Compute budget: **free Colab tier only** `[DECIDED]`. This caps how much video can
  be processed and therefore the number of matches and the MDE. Rough estimate, not
  measured: a full match at 25 fps is about 20 min of T4 time; slow state factors may
  not need 25 fps (about 4 min per match at 5 fps).
- Maximum drawdown tolerance: **deferred** until backtest P&L exists. With no real
  money at risk, the planned bootstrap halt-rule method is enough for paper trading.

**Proposed on 2026-10-08, not yet decided:** decouple (option 1) for the temporal
mismatch; data-source order of Track 2 Kalshi event study, then Track 3 outcome
baseline, then an event-data signal check, then video at scale; send the SoccerNet NDA
request now.

---

## 2026-10-09: Stage 1 carry-overs - SoccerNet-Tracking as fresh footage, and acceptance rules set before measuring `[DECIDED]` `[PRE-REGISTERED]`

Rohan asked to finish every open Stage 1 item. Rules below were written by Claude
under that instruction, before any tracker output on SoccerNet existed (only the
sequence list and one ground-truth file format had been looked at).

**Data source: SN-Tracking-2023 on Hugging Face, not NDA-gated.** Checked
2026-10-09: the repo is public and ungated (`gated: False` from the HF API), unlike
`SoccerNet-Tracking-RAW-Video`, which is gated behind the NDA. No licence file on the
dataset or the `sn-tracking` GitHub repo. The SoccerNet-Tracking paper (arXiv
2204.06918) says test ground truth is published "so that researchers can benchmark
their results locally". Used only for that: private local evaluation, frames
gitignored in `experiments/soccernet/`, no redistribution, aggregate scores only.
So the HOTA check is no longer blocked on the NDA. The NDA question remains for the
main SoccerNet broadcast corpus.

**Fresh footage:** sequences picked by seed 20261009 before any frame was viewed
(`experiments/soccernet/selection.json`):
- Tuning: 6 train sequences (075, 101, 104, 112, 114, 165), plus the spent DFL frames
  300 to 749 with their hand labels.
- Validation, run once per pre-listed variant: 12 test sequences (123, 127, 130,
  131, 140, 144, 149, 150, 187, 189, 191, 196). Different league (Swiss Super League),
  1080p, 25 fps, 30 s each. The test split has 49 sequences; this subset keeps CPU
  time manageable.

**Variants (fixed now):** V0 baseline (frozen settings from the 2026-10-05
spot-check); V1 ball fixes (upper-body rejection, confident-detection override, and
a recalibrated `accel_std`, all chosen on the tuning set); V2 team-split tracking
(per-frame shirt colour, one ByteTrack per team plus one for goalkeepers and
referees); V3 = V1 + V2.

**Acceptance rules on the validation sequences, each against V0 on the same
sequences:**
1. **Ball fixes (V1) are adopted if all three hold:** (a) wrong-ball frames (a
   reported ball, detected or predicted, more than 50 px from the ground-truth ball
   while it is annotated) fall by at least 25% relative; (b) the share of annotated
   ball frames with a reported ball within 10 px drops by no more than 2 percentage
   points; (c) the share of Kalman-predicted frames with the truth inside 2 sigma is
   between 75% and 95%.
2. **Note on (c):** the 2026-10-05 spot-check expected "about 95%" inside 2 sigma.
   That was wrong for a 2D position: for a 2D Gaussian with per-axis sigma, the
   2-sigma circle holds 1 - exp(-2) = 86.5%. The spot-check's 51% is still
   overconfident.
3. **Team-split tracking (V2) becomes the default if** people-only HOTA improves by
   at least 1.0 point and people-only DetA falls by no more than 0.5 point.
   Otherwise it stays in the code as an option, off by default.
4. **HOTA architecture check:** class-agnostic HOTA including the ball (the
   benchmark's own convention) of at least 47.2, the published off-the-shelf
   ByteTrack baseline on the full test set (paper Table 3; FairMOT fine-tuned on
   SoccerNet scored 57.9; ByteTrack with ground-truth detections 71.5). Below 47.2,
   Stage 1 is reopened. Comparison caveat: 12 of 49 sequences, and our detector was
   fine-tuned on Bundesliga images, not SoccerNet.

**ID errors by type:** each ground-truth track's matched prediction ID (IoU of at
least 0.5, per-frame Hungarian matching) is followed; a change is a cross-team swap,
a teammate swap (the new ID previously sat on another ground-truth track of the
other or same team), or a new ID (fragmentation). Reported, no pass rule.

---

## Still open (not decided as of 2026-09-22; last updated 2026-10-08)

- Primary video/CV source for the full research build, following loss of DFL Kaggle
  access (see above).
- Resolution of the video/market temporal mismatch (decouple vs. period-matched odds
  vs. forward-looking live capture).
- SoccerNet NDA text: not yet requested/reviewed.
- Market data redistribution terms: Kalshi is now the only market source (2026-10-08);
  its Developer Agreement is not yet read. football-data.co.uk not yet reviewed.
- Regulatory exposure: real-money trading ruled out (2026-10-08). Publication
  boundary narrowed but open: s.85 of the Gambling Control Act (advertising unlawful
  gambling, strict liability) versus a public repo that names Kalshi and Polymarket.
- Numeric maximum drawdown tolerance: deferred until backtest P&L exists (2026-10-08).
- Compute budget: free Colab tier only (decided 2026-10-08).
- Git repository is initialized (initial commit `35cff10`). `prereg/`,
  `factor_log.md`, and `data_dictionary.md` not yet created.
- Trained detector was run on the cached DFL clip on 2026-09-21 (counts and visual
  review only). Ball precision and referee confusion measured by the 2026-10-05
  spot-check (see that entry).
- Out-of-sample holdout set not yet locked.
- Manual spot-check on the DFL clip (Stage 1 quality gate): done 2026-10-05. Stage 1
  closed as validated with known limits (see those entries).
- Tracker bugs from the 2026-09-22 review: fixed as of 2026-09-25, including the
  NMS test and the Kalman ball filter (see the 2026-09-25 entries).
- Ball filter captured by a persistent false positive, and overconfident
  `ball_sigma` (2026-10-05 spot-check): open; a fix needs fresh footage to validate.
- ID errors on overlap: open; team-split tracking is the candidate fix.
- Power analysis: done 2026-10-05; statistical success criterion revised the same
  day (see those entries). Holdout confirmation rule and the data-source strategy
  are still open.
- Kalshi fee formula taken from Kalshi's help page (2026-10-08); the full PDF fee
  schedule is not yet read. Polymarket fees no longer needed.
