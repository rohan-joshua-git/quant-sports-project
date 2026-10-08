# Phase 0 Scoping Memo: Quant Sports Trading Project

**Document ID:** QSP-P0-001
**Version:** 1.6
**Date:** 2026-10-08 (originally issued 2026-09-18; revised 2026-09-18,
2026-09-20, 2026-09-21, twice on 2026-10-05 and on 2026-10-08, see Revision History)
**Author:** Rohan
**Distribution:** Internal
**Status:** Draft. Phase 0 is not yet closed. Sections marked `[DECIDED]` are locked
and govern downstream work. Sections marked `[OPEN]` are unresolved and block
downstream work until closed.

**Revision History:**
- v1.0 (2026-09-18): Initial Phase 0 scoping memo.
- v1.1 (2026-09-18): DFL Kaggle dataset access lost (competition closed, download
  unavailable) same day as v1.0. Section 2.3 and 5.1 updated accordingly. Detailed
  investigation, interim workaround, and a related detector-retraining decision are
  logged in `decision_log.md`, which this memo now references rather than
  duplicates. This memo remains the point-in-time formal snapshot; `decision_log.md`
  is the append-only source of truth for what happened and when.
- v1.2 (2026-09-20): Synced with `decision_log.md`. Section 5.5 (compute) was stale:
  it still said no hardware plan existed, although the 2026-09-18 log entry had
  already moved training to cloud GPU. Now updated, with a first runtime data point
  from the executed Roboflow fine-tuning run. Section 6 (Stage 1) notes that the
  run has been executed and summarizes the result. Section 8 updated for completed
  items (git initialized, `decision_log.md` exists). No `[DECIDED]` scope decision
  changed.
- v1.3 (2026-09-21): Scope change logged in `decision_log.md`. Live in-play trading
  is now the primary goal instead of a stretch goal (Executive Summary and Section
  1 updated). The evaluation discipline and Section 4 success criteria are
  unchanged. Live-system requirements (video source, latency, execution,
  regulatory exposure) are open and tracked in the log, not yet added to Section 5.
- v1.4 (2026-10-05): Synced Section 6 (Stage 1) with `decision_log.md`. The memo
  still said the detector was "not yet checked on DFL footage"; since then the
  tracker architecture was fixed (2026-09-22, revised 2026-09-25) and a hand-label
  spot-check on held-out DFL frames closed Stage 1 as validated with known limits
  (2026-10-05). The 2026-09-22 research plan and edge hypothesis (H0 to H4 ladder,
  slow state estimation rather than event detection) are recorded in the log and
  `CLAUDE.md` and are not duplicated here. No `[DECIDED]` scope decision changed.
- v1.5 (2026-10-05): Section 4 statistical criterion revised after the power
  analysis: a minimum detectable effect for a per-minute goal-rate test replaces the
  fixed rank-IC bar, and the factor family is capped at 5. Section 4 remains
  `[DRAFT]` until pre-registration; the holdout confirmation rule and the number of
  matches (which depends on the open data-source question) are not yet fixed.
- v1.6 (2026-10-08): Scope change logged in `decision_log.md`. The primary goal is
  now a rigorous backtest plus paper trading, reversing v1.3: real-money trading is
  not legal from Singapore, where the author lives, and the project's purpose is a
  portfolio towards a quant career. Executive Summary, Section 1, Section 2.5 (Kalshi
  as data source only, Polymarket dropped), Section 4 (drawdown deferred), Sections
  5.3 to 5.5 and Section 8 updated. Section 5.5 compute budget decided (free tier).
  Section 5.4 narrowed but still open (publication boundary).

---

## Executive Summary

This memo documents the Phase 0 scoping process for a quantitative sports trading
research project. The project applies established equities-style factor research
methodology, specifically point-in-time discipline, cross-sectional standardization,
and a formal evaluation stack (Information Coefficient, Newey-West significance
testing, Benjamini-Hochberg false discovery rate control, and Probability of Backtest
Overfitting), to a sports domain, using computer-vision-derived tracking and event
data in place of fundamentals data.

Scope is fixed at soccer, with a team-level match-market framing. The primary goal
is a rigorous backtest plus paper trading against live public prices, with no real
money (revised 2026-10-08; live in-play trading was the primary goal from 2026-09-21,
and the project was originally backtest-first with live as a stretch goal). One
material risk remains open and gates
finalization of the target variable and success criteria: a temporal mismatch between
the available video data (2022 season) and the available market history on the
candidate trading venues (2024 onward). This risk, and four smaller open items, are
detailed in Section 5.

---

## 1. Objective

This project applies equities-style factor research methodology, point-in-time (PIT)
discipline, cross-sectional standardization, and the IC / Newey-West / BH-FDR / PBO
evaluation stack, to sports, using computer-vision-derived tracking and event data as
the data-generation layer in place of fundamentals data. In-play sports markets are
treated as a cross-sectional asset universe: a set of entities (teams), each carrying
a market-implied price (odds), evaluated for mispricing using signals computed across
that cross-section.

The primary goal is a rigorous backtest plus paper trading (revised 2026-10-08).
Real-money trading is out of scope: it is not legal from the author's jurisdiction
(Singapore, Section 5.4), and the project's purpose is a portfolio towards a quant
career. From 2026-09-21 to 2026-10-08 the goal was live in-play trading; before that,
a backtest first phase with live execution as a stretch goal. The evaluation bar is set at
an institutional research standard: every factor tested is pre-registered before its
result is known, every test, whether it passes or fails, is logged, and statistical
validity (FDR-corrected significance, PBO-checked robustness) is required before any
factor is treated as real.

---

## 2. Scope Decisions

### 2.1 Sport `[DECIDED]`

**Soccer.** Selected over alternative sports primarily on the strength of publicly
available broadcast video and computer-vision research infrastructure, in particular
SoccerNet, which no other sport currently matches at comparable scale and
documentation quality.

### 2.2 Market Universe `[DECIDED, with an open sub-question]`

**Team-level match markets** (win probability, in-play repricing), rather than
player-level proposition markets. Player-level cross-sectional factors were
considered as a closer match to the original equities framing, being a
position-neutralized panel of players, but were rejected for the first version of
this project on three grounds: match markets carry materially better liquidity and
odds-data coverage than player props; SoccerNet's annotation coverage is strongest at
the match and event level; and point-in-time alignment is simpler against a single
match clock than against a many-player, many-game panel. Player-level factors remain
a candidate second phase once the core pipeline discipline is proven against the
simpler case.

### 2.3 Video and Computer-Vision Data Source `[DECIDED, then blocked same day; see
decision_log.md]`

Primary analysis footage was drawn from the DFL Bundesliga Data Shootout dataset
(2022 Bundesliga season broadcast video). This dataset's own annotations are limited
to event-spotting labels (play, challenge, and throw-in timestamps) and provide no
bounding boxes, player tracking, or ball tracking ground truth. Any player or ball
tracker built on this footage (detector, tracker, and homography calibration) is
therefore self-supervised, with no ground truth available on this specific footage
against which to check accuracy.

**Update, same day:** the Kaggle competition's data download is no longer available
(competition closed, download disabled). The written usability confirmation from
Kaggle's Data Team is now practically moot, since the data it covers is no longer
accessible through that channel. A local copy of at least one clip, downloaded
before the listing closed, is being used for private technical smoke-testing only
(see `experiments/yolo_smoke_test.ipynb`), not as a settled primary source for the
research build. Full investigation and options considered are in `decision_log.md`.
Primary video/CV source for the actual research run is `[OPEN]` again as of this
revision.

SoccerNet's broader corpus (500 broadcast games across six leagues, including
Bundesliga, spanning the 2014-2017 seasons) remains under consideration as an
alternative, but it does **not** close the temporal gap to 2024-onward market data
discussed in Section 5.1 — it is older than the DFL footage it would replace, not
closer to the present. Its NDA terms also remain unread (Section 5.2), so it should
not be assumed to be a lower-friction substitute on licensing grounds either.

### 2.4 Tracking-Accuracy Validation `[DECIDED]`

SoccerNet-Tracking (Swiss Super League, 12 full games, labeled bounding boxes and
tracklet identifiers) is used to validate the tracking pipeline's architecture, scored
using HOTA rather than MOTA. HOTA weights detection quality and identity-association
quality more evenly than MOTA, and association is the point at which trackers
typically fail across camera cuts.

This is explicitly a proxy validation, not a direct one. SoccerNet-Tracking's footage
is drawn from a different league and different matches than the DFL analysis footage,
so a satisfactory HOTA score confirms that the pipeline design performs acceptably on
video of this general character; it does not measure actual error on the DFL footage
used for research. A manual spot-check, consisting of hand-labeling a small sample of
actual DFL clips, is required separately to establish real error bars on the data
in use. Both figures should be reported side by side; the SoccerNet figure should not
be allowed to stand in for tracking accuracy on the project's actual data.

### 2.5 Market Platform `[DECIDED: Kalshi as data source only; see Section 5.1]`

**Revised 2026-10-08:** Kalshi is the market data source, for research and paper
trading only. Its member agreement restricts Singapore for trading but allows
non-trading access, and its market-data endpoints are public. Polymarket is dropped:
the Singapore Gambling Regulatory Authority has blocked it since January 2025, it is
close-only for Singapore, and its terms ban VPN circumvention. The original text
follows.

Kalshi and Polymarket were selected initially for API accessibility. Both are
confirmed, as of 2026, to offer continuous in-play soccer markets, not merely
pre-game settlement, and historical-data APIs suitable for backtesting. This decision
should not yet be treated as final; see the temporal mismatch documented in Section
5.1. Platform liquidity and API ergonomics were not, in the end, the binding
constraint on this choice.

---

## 3. Target Variable `[DRAFT; pre-registration pending final lock]`

Two parallel targets are tracked, not one, because they answer distinct questions.

**Calibration / Information Coefficient target:**

```
edge_t = P_model(outcome | data through t) - P_market(outcome | tradeable price at t)
```

with the forward outcome realized after t. This target is scored using Brier score
and log loss for calibration, and rank-IC of the edge against the realized outcome
for predictive power.

**Economic target:** simulated profit and loss using actual tradeable back and lay
prices, not the mid-price, net of exchange commission, with a deliberately injected
latency lag between the moment data becomes observable and the moment an order is
placed, to simulate realistic computer-vision inference delay. A model that achieves
acceptable IC but no economically survivable edge after latency and commission is
treated as a null result for this project, not a partial success. This is stated
explicitly so that it cannot be reinterpreted favorably at a later stage.

**Critical constraint:** the market price baseline must be the price actually
tradeable at time t, never a revised or settled price. This is the sports-analytics
equivalent of lookahead bias and applies with equal weight to market data and to
computer-vision-derived tracking data; no factor may use tracking data that was not
yet observable as of its own timestamp.

---

## 4. Success Criteria `[DRAFT; must be formally pre-registered, in writing, before
the first factor test is run]`

- **Statistical criterion (revised 2026-10-05).** The original criterion, a rank-IC
  of at least 0.02 to 0.03 with a Newey-West t-statistic of at least 2, was replaced
  after a power analysis (`prereg/power_analysis.ipynb`) showed it needs about 1,100
  to 5,600 matches in the best case, against about 500 in the largest archive
  considered. The primary test is now a per-minute Poisson goal-rate model (factor at
  minute t-1, goals in minute t, the baseline expected rate as an offset, standard
  errors clustered by match). Success is stated as a minimum detectable effect: the
  number of matches is fixed before any factor test, together with the goal-rate
  shift per standard deviation of the factor that the test detects with 80% power,
  and a null result only rules out effects larger than that (for example, about 8.7%
  at 500 matches in the base case, 15.3% after multiple-testing, holdout and
  measurement-error adjustments). Rank-IC at a 5-minute horizon, with match-clustered
  errors, is reported as a secondary statistic. Minute-level observations within a
  match are highly autocorrelated: a match carries only about 18 effectively
  independent observations for a 5-minute target.
- The factor family is capped at 5. Factors must survive Benjamini-Hochberg false
  discovery rate correction at q = 0.05, applied across the complete family of
  factors tested. Every factor tried must be logged, including negative results; the
  correction is statistically meaningless without the complete family.
- The holdout confirmation rule is not yet set: with a 30% holdout of about 500
  matches, a confirmation test at t of 2 or more only has power for large effects.
  It must be fixed before the first factor test.
- Probability of Backtest Overfitting (PBO) of 20% or lower. A threshold below 50%,
  per Bailey and Lopez de Prado, indicates only that a result is not obviously
  overfit, not that it is robust; 20% is set as the working bar for this project.
- Positive net-of-cost economic edge (commission plus simulated slippage), with a
  minimum Sharpe-like ratio on the match-level profit and loss series, validated
  across multiple seasons and competitions out of sample. Single-season validation is
  not sufficient.
- Maximum drawdown tolerance: deferred until backtest P&L exists (decided
  2026-10-08). With paper trading only, no money is at risk; the halt level will come
  from a match-block bootstrap of backtest P&L.

---

## 5. Known Risks and Open Items

### 5.1 Video/Market Temporal Mismatch `[OPEN; highest priority]`

The DFL analysis footage is drawn from the 2022 Bundesliga season. Polymarket's price
history extends back only to 2024. Kalshi's soccer and sports market activity is a
recent, post-2022 expansion. It is likely that no historical odds data exists on
either platform for the specific 2022 matches contained in the DFL dataset. This is a
data-availability problem, independent of which platform is ultimately chosen. Three
options remain on the table, and none has yet been decided:

1. **Decouple.** Validate the computer-vision-derived signal against realized match
   outcomes only, with no market and no edge calculation, using the 2022 video now.
   Treat market-fit, meaning trading against a live price, as a separate, later,
   forward-looking phase using freshly captured video of current matches.
2. **Source period-matching historical odds.** Obtain 2022 Bundesliga season odds
   from an established historical-data provider, such as Betfair Historical Data or
   football-data.co.uk, in place of Kalshi or Polymarket, trading platform-API
   convenience for temporal coverage.
3. **Abandon the archived DFL video.** Build the pipeline against live or recent
   matches from the outset. This reopens the video-sourcing problem, including
   broadcast copyright and scraping risk, that using an existing archival dataset was
   intended to avoid.

**Update, same day as v1.0:** DFL Kaggle access was lost (Section 2.3), and its most
likely archival replacement, SoccerNet's main corpus, is *older* (2014-2017) than
the DFL footage it would replace, not closer to 2024-onward market data. This does
not change which of the three options above is correct, but it does mean any
archival-video path makes the case for option 1 (decouple) stronger by default,
since no archival broadcast source currently available closes this gap. Still not
formally locked.

### 5.2 SoccerNet NDA Terms `[OPEN]`

Raw SoccerNet video access requires execution of a non-disclosure agreement intended
to prevent redistribution of copyrighted broadcast material. The full text of this
agreement is not publicly available; it is provided by email only after access is
requested, and it has not yet been reviewed. Permissive, unrestricted terms should
not be assumed. Such a claim surfaced during scoping discussion and is inconsistent
with the NDA-gated access process observed on the SoccerNet data portal. This item
remains open until the executed text is reviewed directly.

### 5.3 Market Data Redistribution Terms `[OPEN]`

Kalshi is now the only market source (Section 2.5). Its Developer Agreement, which
governs API data use, has not yet been read. Until it is, only aggregate statistics
are published. football-data.co.uk terms (for the outcome baseline) are also unread.

### 5.4 Regulatory and Publication Exposure `[OPEN, narrowed 2026-10-08]`

The author is resident in Singapore. Checked 2026-10-08 from public sources, not
legal advice: gambling with an unlicensed operator is an offence under the Gambling
Control Act 2022, s.20(3), so real-money trading is ruled out. Still open: s.85 of
the same Act makes advertising unlawful gambling a strict-liability offence, and
advertising includes informing the public of an online location where unlawful
gambling takes place. The project repository is public and names Kalshi and
Polymarket. Any public writeup is framed as academic research, never as betting
advice, with no links or invitations to any betting venue.

### 5.5 Compute Budget `[DECIDED 2026-10-08: free tier only]`

**Revised 2026-10-08:** the budget is the free Google Colab tier only. This caps how
much video can be processed, and so the number of matches and the minimum detectable
effect in Section 4. The original text follows.

Computer-vision training and inference on broadcast video is GPU-intensive. The
hardware plan is decided: training runs on external cloud GPU (Google Colab), not
on local hardware, after a local training attempt caused a hardware fault on
2026-09-18. Local hardware is used only for short inference-only checks. What remains
open is a numeric cost ceiling for cloud usage. One data point now exists: the first
100-epoch fine-tuning run (`yolo26n`, about 0.53 hours on a T4). This will still
determine how much footage can realistically be processed.

---

## 6. Pipeline Architecture (Conceptual)

1. **Detection and tracking.** Player, ball, and referee detection and tracking
   across frames, with handling for camera cuts. Validated via HOTA against
   SoccerNet-Tracking, together with a manual spot-check on actual DFL footage
   (Section 2.4). A smoke test of a stock pretrained detector (YOLO26n, COCO
   weights) surfaced three problems (unreliable ball detection, a false-positive
   `tv` classification on broadcast graphics, and sideline personnel misclassified
   as players); the resulting decision to fine-tune on a Roboflow dataset instead of
   generic weights is logged in `decision_log.md`, not repeated here. That run was
   executed on 2026-09-20: strong player, goalkeeper, and referee detection on the
   Roboflow test split, but weak ball detection. A hand-label spot-check on held-out
   frames of the cached DFL clip (2026-10-05) measured people detection at about 97
   to 100% (precision 99.8%, recall 97.2%, class correct 98.2%) and the ball at 91.2%
   precision but 54.2% recall, with the ball filter vulnerable to capture by a
   persistent false positive. Stage 1 is closed as validated with known limits; the
   SoccerNet-Tracking HOTA proxy check is still outstanding. Full results are in
   `decision_log.md`.
2. **Calibration (pitch homography).** Pixel-space coordinates are mapped to
   real-world pitch coordinates. This step is required before any speed or distance
   figure is meaningful.
3. **Identity and context layer.** Team assignment via jersey clustering,
   re-identification across camera cuts, and possession attribution.
4. **Point-in-time feature extraction.** Tracking and event streams are converted
   into timestamped, as-of-correct rows in the raw data store.
5. **Factor construction.** Candidate factors, including a fatigue proxy, momentum,
   event-rate intensity, and territorial dominance, are constructed from Stage 4
   output. Each factor is pre-registered before testing.
6. **Market and target alignment.** Factor timestamps are aligned to market price
   timestamps; edge and forward outcome are defined per Section 3.
7. **Evaluation.** IC, Newey-West, BH-FDR, PBO, and net-of-cost economic backtest,
   assessed against the locked success criteria in Section 4.

---

## 7. Governance and Process Discipline

- Every factor hypothesis is pre-registered, including its definition, expected sign,
  and success criteria, in writing, before it is tested. It is never registered after
  the fact.
- Every factor tested is logged, including negative results. BH-FDR correction is
  invalid without the complete tested family.
- Structural and scoping decisions, including those in this memo, are dated and
  logged at the time they are made, not reconstructed retroactively.
- Point-in-time correctness is mandatory for both computer-vision-derived tracking
  data and market price data. No factor may reference data that was not observable as
  of its own timestamp.
- This is a solo project. Independent validation is enforced procedurally through a
  locked out-of-sample holdout set, with the season or competition to be fixed, which
  is not examined until every factor has already passed IC, FDR, and PBO testing on
  the development set.
- Public-facing output, including the repository and any writeups, may contain code
  and aggregated statistical results only. Raw video and derived video frames are
  excluded, given SoccerNet and DFL redistribution restrictions.

---

## 8. Next Steps

1. Resolve the video/market temporal mismatch (Section 5.1). This blocks final lock
   of Sections 3 and 4.
2. Request and review SoccerNet's executed NDA text (Section 5.2).
3. Read Kalshi's Developer Agreement (Section 5.3).
4. Decide the publication boundary under s.85 of the Gambling Control Act (Section
   5.4).
5. Maximum drawdown tolerance: deferred until backtest P&L exists (Section 4).
6. Compute budget: decided 2026-10-08, free tier only (Section 5.5).
7. Establish `prereg/`, `factor_log.md`, and `data_dictionary.md` alongside this
   memo. (Version control is initialized and `decision_log.md` exists.)
8. Lock the out-of-sample holdout set (Section 7).
