# Phase 0 Scoping Memo: Quant Sports Trading Project

**Document ID:** QSP-P0-001
**Version:** 1.2
**Date:** 2026-09-20 (originally issued 2026-09-18; revised 2026-09-18 and
2026-09-20, see Revision History)
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

---

## Executive Summary

This memo documents the Phase 0 scoping process for a quantitative sports trading
research project. The project applies established equities-style factor research
methodology, specifically point-in-time discipline, cross-sectional standardization,
and a formal evaluation stack (Information Coefficient, Newey-West significance
testing, Benjamini-Hochberg false discovery rate control, and Probability of Backtest
Overfitting), to a sports domain, using computer-vision-derived tracking and event
data in place of fundamentals data.

Scope is fixed at soccer, with a team-level match-market framing. The project is
structured as a backtest in its first phase; live execution is a stated stretch goal
and out of scope for the current phase. One material risk remains open and gates
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

The project is scoped as a backtest in its first phase. Live in-play trading and
execution is a stretch goal, not the initial deliverable. The evaluation bar is set at
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

### 2.5 Market Platform `[DECIDED in principle; blocked, see Section 5.1]`

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

- Rank-IC of at least 0.02 to 0.03, with a Newey-West t-statistic of at least 2.
  Minute-level observations within a single match are highly autocorrelated, so the
  effective sample size for significance purposes is closer to the number of
  independent matches than to the number of minute-level observations.
- Factors must survive Benjamini-Hochberg false discovery rate correction at q = 0.05,
  applied across the complete family of factors tested. Every factor tried must be
  logged, including negative results; the correction is statistically meaningless
  without the complete family.
- Probability of Backtest Overfitting (PBO) of 20% or lower. A threshold below 50%,
  per Bailey and Lopez de Prado, indicates only that a result is not obviously
  overfit, not that it is robust; 20% is set as the working bar for this project.
- Positive net-of-cost economic edge (commission plus simulated slippage), with a
  minimum Sharpe-like ratio on the match-level profit and loss series, validated
  across multiple seasons and competitions out of sample. Single-season validation is
  not sufficient.
- Maximum drawdown tolerance: not yet numerically set. This figure is to be fixed
  before any backtest is run, while no result yet exists that could bias the number.

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

Redistribution and publication terms for the eventual historical odds source, whether
Kalshi, Polymarket, Betfair, or football-data.co.uk, have not yet been reviewed.

### 5.4 Regulatory and Publication Exposure `[OPEN]`

Gambling-advice regulatory exposure for any public output has been identified but not
resolved for the applicable jurisdiction. Any public writeup should be framed as
academic or research backtesting, and never as betting advice, pending resolution of
this item.

### 5.5 Compute Budget `[OPEN, hardware plan decided]`

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
   Roboflow test split, but weak ball detection, and not yet checked on DFL footage.
   Full results are in `decision_log.md`.
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
3. Review redistribution terms for the eventual market-data source (Section 5.3).
4. Confirm the gambling-advice and publication boundary for the applicable
   jurisdiction (Section 5.4).
5. Set a numeric maximum drawdown tolerance (Section 4).
6. Set a numeric compute budget for cloud computer-vision training and inference
   (Section 5.5). The hardware plan is already decided.
7. Establish `prereg/`, `factor_log.md`, and `data_dictionary.md` alongside this
   memo. (Version control is initialized and `decision_log.md` exists.)
8. Lock the out-of-sample holdout set (Section 7).
