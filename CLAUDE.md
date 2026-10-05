# Quant Sports Trading Project

## Vision
Institutional-grade quant research project applying equities-style factor methodology
(point-in-time discipline, cross-sectional z-scoring, IC / Newey-West / BH-FDR / PBO
evaluation) to sports, using computer-vision-derived features as the data-generation
layer in place of fundamentals data. Primary goal (changed 2026-09-21): live in-play
trading. The locked success criteria and validation discipline below still gate any
real-money trading. Earlier framing was "backtest first, live is a stretch goal";
see `decision_log.md`.

## Status
Phase 0 (scoping) — in progress, not yet closed. Solo project.

## Decisions locked so far
- Sport: soccer.
- Video/CV source: DFL Bundesliga Data Shootout (Kaggle), 2022 Bundesliga season —
  **access lost 2026-09-18** (Kaggle competition closed, download unavailable).
  Written Kaggle Data Team usability confirmation is now practically moot since the
  data it covers is no longer reachable through that channel. A locally-cached clip
  from before the listing closed is in use for private smoke-testing only (see
  `experiments/`), not as a settled primary source. Primary video/CV source for the
  actual research build is open again. SoccerNet's main corpus (500 games, 6 leagues
  incl. Bundesliga) is the leading alternative under consideration, but it covers
  2014-2017 — older than DFL's 2022 footage, not closer to 2024-onward market data —
  so it does not resolve the temporal mismatch below, and its licensing is not
  confirmed cleaner (NDA text still unread, see Data licensing status). Full
  investigation and options considered are in `decision_log.md`.
- DFL dataset's own annotations are event-spotting only (play / challenge / throw-in
  timestamps) — no bounding boxes, player tracking, or ball tracking ground truth.
  Any player/ball tracker (YOLO + ByteTrack/DeepSORT + homography calibration) has to
  be built from scratch on the raw video, with no ground truth on this specific
  footage to validate against. A smoke test of a stock pretrained detector (YOLO26n,
  COCO weights) confirmed this is a real gap in practice (unreliable ball detection,
  a `tv` false positive on broadcast graphics, sideline personnel misclassified as
  players) — decision made 2026-09-18 to fine-tune on the Roboflow Universe "Football
  Players Detection" dataset (CC BY 4.0) instead of relying on generic weights.
  Executed 2026-09-20 on Colab (T4, about 32 minutes, 100 epochs): test-split mAP50
  0.772 / mAP50-95 0.495, strong for player/goalkeeper/referee but weak for the ball
  (mAP50 about 0.31, recall about 0.36). Measured on Roboflow's images only. Run on
  the cached DFL clip 2026-09-21 (counts and visual review only), then hand-labelled
  on 2026-10-05 (see the Stage 1 bullet below). Detail in `decision_log.md`.
- Tracker architecture (2026-09-22, revised 2026-09-25): frame-by-frame `Tracker`
  class in `pipeline/detection/tracker.py`, chosen for live-stream compatibility.
  People: YOLO, class-aware NMS at 0.3, a second cross-class NMS at 0.7 (removes
  referee/player double boxes), then `supervision` ByteTrack. Ball: kept out of
  ByteTrack (which was discarding over half the ball detections), picked from
  detector output and tracked by a forward-only Kalman filter
  (`pipeline/detection/ball_filter.py`) with a gate and a `ball_sigma` uncertainty.
  imgsz 640. About 43 to 54 ms/frame mean on local CPU (YOLO about 84%), over the
  40 ms budget at 25 fps; GPU needed for live.
- **Stage 1 closed 2026-10-05 as "validated with known limits"** (no pass threshold
  was pre-set; future stages should set one before measuring). Hand-label spot-check
  on held-out frames 300 to 749: people precision 99.8%, recall 97.2%, class correct
  98.2%; ball detector precision 91.2% but recall only 54.2%; a persistent false
  positive (a green-kit player's body) captured the ball filter for about 4.5 s, and
  `ball_sigma` is overconfident (truth inside 2 sigma 51% instead of 95%). Frames 300
  to 749 are now spent for tuning; fixes need fresh footage. SoccerNet-Tracking HOTA
  still not done (NDA unread).
- Research plan and edge hypothesis (2026-09-22, full detail in `decision_log.md`):
  in-play prices absorb goals swiftly and fully (Croxson & Reade 2014), and broadcast
  video lags the live event, so any edge must come from **slow state estimation**
  (sustained pressure, dominance, shape changes, fatigue), not from detecting events
  before the market. Tested via a pre-registered H0 to H4 ladder: tracking adds
  nothing (H0), adds short-horizon information over a score/time/team-strength
  baseline (H1), only in some latent states (H2), survives realistic latency (H3),
  holds across seasons/competitions (H4). EGARCH/Markov-switching are demoted to
  descriptive use; Markowitz, technical indicators, deep order-book models, RL
  execution and market making are rejected; features needing all 22 players (pitch
  control, Voronoi) are deferred because broadcast video shows only part of the pitch.
- Tracking-accuracy validation: SoccerNet-Tracking (Swiss Super League, 12 games,
  labeled bounding boxes + tracklet IDs), scored via HOTA (not MOTA — HOTA balances
  detection and identity-association quality, which is where trackers usually fail
  across camera cuts). This is a proxy/architecture check only — different league,
  different footage from DFL, so it does not directly validate DFL-derived tracking
  accuracy. A manual spot-check (hand-label a small sample of actual DFL clips) is
  still required to get real error bars on the data actually being used.
- Market platform (tentative): Kalshi and/or Polymarket. Both confirmed (2026) to have
  genuine continuous in-play soccer markets and historical-data APIs.

## Critical open blocker: video/market temporal mismatch
DFL video is from the 2022 Bundesliga season. Polymarket price history only goes back
to 2024; Kalshi's soccer/sports market activity is a recent (post-2022) expansion.
There is likely no historical odds data on either platform for the specific 2022
matches in the DFL dataset — a data-availability problem independent of which
platform is chosen. Not yet decided among:
1. Decouple: validate the CV-derived signal against realized match outcomes only
   (no market/edge calc) using the 2022 video; treat market-fit as a separate,
   later, forward-looking phase using fresh video capture of current matches.
2. Source period-matching historical odds for 2022 Bundesliga from an older-style
   provider (Betfair Historical Data, football-data.co.uk) instead of Kalshi/Polymarket.
3. Abandon the archived DFL video and build the pipeline forward-looking against
   live/recent matches (reopens broadcast-copyright/scraping risk for video sourcing).

Update: since losing DFL Kaggle access, the leading replacement (SoccerNet's main
corpus) is *older* than DFL, not closer to present-day market data. Any archival
video source makes option 1 (decouple) the practically favored path by default —
still not formally locked.

## Data licensing status
- DFL Kaggle dataset: usable for this personal project per written confirmation from
  Kaggle's Data Team. Keep the confirmation email on file.
- SoccerNet: raw video requires signing an NDA (prevents redistribution of copyrighted
  broadcast material). Actual NDA text is not public — only sent via email after
  requesting access. Not yet verified — do not assume "open source" terms; that claim
  was raised in conversation and is inconsistent with the NDA-gated access process.
  Unresolved until the real NDA text has been read.
- Market data (Kalshi/Polymarket/Betfair/etc.): redistribution ToS not yet checked
  for whichever source is ultimately used.
- Gambling-advice / publication regulatory exposure: flagged, not resolved for the
  user's specific jurisdiction. Any public output should be framed as academic/
  research backtesting, never as betting advice.

## Target variable (draft, not finalized)
Two parallel targets, not one:
- Calibration/IC target: edge_t = P_model(outcome | data through t) −
  P_market(outcome | tradeable price at t), scored via Brier score / log loss and
  rank-IC of edge vs. realized outcome.
- Economic target: simulated P&L using actual tradeable back/lay prices (not mid),
  net of commission, with a deliberately injected latency lag to simulate real
  CV + inference delay.
Market price baseline must be the price actually tradeable at time t, never a
revised/settled price — the sports-analytics equivalent of lookahead bias.

## Success criteria (draft — must be formally pre-registered before any factor testing)
- **Statistical criterion (revised 2026-10-05, replaces "rank-IC ≥ 0.02–0.03, t ≥ 2"):**
  - Primary test: per-minute Poisson goal-rate model. Factor measured at minute t-1,
    goals in minute t, baseline expected rate (score, time, team strength) as an
    offset; score/Wald test of the factor coefficient beta with standard errors
    clustered by match.
  - Success is stated as a **minimum detectable effect (MDE)**: before any factor
    test, fix the number of matches and report the smallest goal-rate shift per SD of
    the factor that the test detects with 80% power (from `prereg/power_analysis.ipynb`,
    step 7). A null result only rules out effects larger than the MDE. Reference
    points (phi 0.95, base / adjusted for BH, holdout, CV error): 500 matches 8.7% /
    15.3%; 1,000 matches 6.1% / 10.6%.
  - Secondary, reported but not the pass criterion: rank-IC at a 5-minute horizon
    with match-clustered errors (effective N is about matches x 18, not minutes).
- Factor family capped at **5** factors; each must survive BH-FDR at q = 0.05 across
  the full family tested. Every factor tried must be logged, including failures, or
  the correction is invalid.
- PBO ≤ 20%.
- Positive net edge after commission + slippage, with a minimum Sharpe-like ratio on
  match-level P&L, validated across multiple seasons/competitions out-of-sample.
- Max drawdown tolerance: not yet numerically set. Planned method: a halt level from a
  match-block bootstrap of backtest P&L (e.g. its 99th-percentile drawdown).
- Power analysis done 2026-10-05 (`prereg/power_analysis.ipynb`): the old bar,
  rank-IC 0.02 at t ≥ 2 with 80% power, needed about 1,100 to 5,600 matches best
  case (about 3.4 times more adjusted), against about 500 in the largest archive.
  That is why the statistical criterion above was revised.
- Still open: holdout confirmation rule. A 30% holdout of 500 matches is 150 matches,
  where a confirmation test at t ≥ 2 only has power for large effects; the rule (for
  example, same sign plus a one-sided p below 0.10, or pooled testing) must be
  pre-registered before any factor test. Number of matches depends on the open
  video-source decision.

## Roles
Solo project. "Independent validation" enforced procedurally via a locked holdout set
(season/competition TBD) not touched until every factor has already passed
IC/FDR/PBO on the development set.

## Process discipline (applies to all future work on this project)
- Pre-register every factor hypothesis (definition, expected sign, success criteria)
  in writing before testing it — never after.
- Log every factor tested, including negative results.
- Log structural/scoping decisions with dated rationale as they're made, not
  retroactively.
- Point-in-time correctness is mandatory for both CV-derived tracking data and market
  price data — no factor may use data not observable as of its own timestamp.
- Public repo output may contain code and aggregated statistical results only — no
  raw video, no derived video frames, given DFL/SoccerNet redistribution restrictions.

## Pipeline architecture (conceptual stages)
Code layout: real pipeline code lives in `pipeline/<stage>/` (Stage 1 is
`pipeline/detection/`: `train.ipynb`, `evaluation.ipynb`, `tracker.py`,
`ball_filter.py`; shared drawing helpers in `pipeline/common/drawing.py`).
`experiments/` is reserved for diagnostic spikes (including `spotcheck.py`, the
hand-label tool; its output folder `experiments/spotcheck/` is gitignored). Package
versions are pinned in `requirements.txt` (`sv.ByteTrack` is removed in supervision
0.31). This machine's OpenCV has no window support (`opencv-python-headless` is also
installed), so interactive tools use matplotlib with the TkAgg backend.

1. Detection & tracking (player/ball/referee) — validated via HOTA against
   SoccerNet-Tracking, plus manual spot-check on actual DFL footage.
2. Calibration (pitch homography) — pixel space → real-world coordinates; required
   before any speed/distance figure is meaningful.
3. Identity & context layer — team assignment, re-ID across camera cuts, possession
   attribution.
4. PIT feature extraction — timestamped rows into the raw data store, strict as-of
   discipline.
5. Factor construction — pre-registered before testing.
6. Market/target alignment — factor timestamps aligned to market price timestamps.
7. Evaluation — IC, Newey-West, BH-FDR, PBO, net-of-cost backtest against locked
   success criteria.

## Open next steps
- Resolve the video/market temporal mismatch (highest priority — blocks finalizing
  the target variable and evaluation design).
- Resolve primary video/CV source for the research build, following DFL Kaggle
  access loss (see Decisions locked so far).
- Request and read SoccerNet's actual NDA text.
- Stand up `prereg/`, `factor_log.md`, `data_dictionary.md`. (`decision_log.md` is
  done and the repo is git-initialized.)
- Lock the out-of-sample holdout set.
- Check redistribution ToS for whichever market data source is chosen.
- Set a numeric max-drawdown tolerance.
- Confirm gambling-advice/publication boundary for the user's jurisdiction.
- Set a numeric compute budget/cost ceiling for cloud GPU usage. Where training runs
  is now decided (external/cloud GPU, e.g. Google Colab, not sustained local
  training — a local training attempt caused a hardware failure on 2026-09-18; local
  hardware remains fine for short inference-only smoke tests). See `decision_log.md`.
  Data point: the first 100-epoch fine-tuning run took about 0.53 hours on a Colab T4.
- Define live-system requirements now that live trading is the primary goal (live
  video source and its terms, frame-to-order latency budget, market execution,
  regulatory exposure). See the 2026-09-21 entry in `decision_log.md`.

Research plan (2026-09-22, see `decision_log.md`): four parallel tracks feeding one
final test. Tracks 2 and 3 do not need the unresolved video source.

- **Track 1: Stage 1. Done 2026-10-05** (review bugs fixed, ball out of ByteTrack,
  Kalman ball filter, cross-class NMS, visual check, hand-label spot-check; see the
  Stage 1 bullet in Decisions and `decision_log.md`). Carried forward, none blocking:
  - Ball filter capture by a persistent false positive, and overconfident
    `ball_sigma`. Candidate fixes: reject ball detections on a player's upper body;
    let a confident detection outside the gate override weak ones inside it. Must be
    validated on fresh footage, not frames 300 to 749.
  - ID errors on overlap: count by type first; candidate fix is team-split tracking
    (per-frame team label from shirt colour, one tracker per team, vote per ID).
  - SoccerNet-Tracking HOTA proxy check (blocked on the NDA).
  - Annotation polish (low priority): ring width from box height; per-ID smoothing.
- **Track 2: market event study.** Check Kalshi/Polymarket data ToS and official fee
  schedules first. Then, on 2024+ in-play soccer prices: speed and completeness of
  price reaction to goals and red cards (Croxson & Reade method), which also measures
  broadcast delay; market-price calibration including favorite-longshot bias;
  spread, depth and fee profile by minute and price level.
- **Track 3: outcome baseline.** Dixon-Coles pre-match strength with partial-pooling
  shrinkage, Dixon-Robinson in-play goal hazard, Monte Carlo of the remaining match.
  Check licensing of the historical results source first.
- **Track 4: pre-registration (current focus).** Power analysis done 2026-10-05;
  statistical success criterion revised the same day (MDE, per-minute goal-rate test,
  5 factors). Next: the data-source strategy (cheap tests first: Track 2 and an
  event-data signal check, weighed 2026-10-05, not decided). Then write the H0 to H4
  ladder and a small factor family (at most 5 slow state factors, including the
  forward-only HMM state) into `prereg/`, lock the holdout, and fix the latency grid
  (2, 5, 10, 30 seconds).
- **Stages 2 to 4 before the final test:** homography calibration, team assignment
  from jersey colour (per-frame label with a running vote per track ID, not one fit
  per ID then reuse, which would carry an ID switch forward; see the 2026-09-25
  visual-review entry), re-ID, possession, PIT features.
- **Final test:** factors neutralized against the baseline probability; IC and IC
  decay by horizon, match-clustered SEs, Diebold-Mariano clustered by match, BH,
  combinatorial purged CV, PBO, Deflated Sharpe Ratio; CV-error perturbation and
  latency robustness. Only if the signal survives: latency-injected replay on
  tradeable order-book prices, edge threshold of fee plus half-spread plus buffer,
  markouts, shrunk Kelly with per-match caps, bootstrap drawdown halt rule.

## Maintenance notes (read before making updates to this project)
- `decision_log.md` (project root) is the append-only source of truth for what
  happened and why. When something changes — a dataset goes away, a decision gets
  reversed, a problem gets solved — log it there with a date, don't just silently
  edit the past. Never rewrite past entries to look like the current decision was
  the only one ever considered.
- `research/phase0_scoping_memo.md` is a point-in-time formal snapshot, not a live
  document. When asked to "update" the project, check whether its `[DECIDED]` tags
  still match `decision_log.md` — if not, revise the memo's affected sections, bump
  its version number, and add a Revision History entry, rather than leaving it
  silently stale (this happened once already: Section 2.3 kept saying DFL was
  cleanly `[DECIDED]` for a while after Kaggle access was actually lost).
- This CLAUDE.md file's own "Decisions locked so far" section can go stale the same
  way. Treat it as needing the same check-and-sync pass as the memo, not as
  permanently authoritative just because it's the instructions file.
- This project has a repeated pattern: assumed data availability/licensing turns out
  to be wrong (DFL Kaggle delisting; an unverified "SoccerNet is open source" claim
  that contradicted its actual NDA-gated access). Treat every data-source
  availability/licensing claim as unverified until independently confirmed — every
  time a new one comes up, not just the ones already flagged here.
- Diagnostic/spike work (e.g. `experiments/`) doesn't need pre-registration, but
  material findings from it still belong in `decision_log.md` if they drive a
  structural decision (e.g. the YOLO smoke test → Roboflow retrain decision) — don't
  let it stay buried only in a notebook.
