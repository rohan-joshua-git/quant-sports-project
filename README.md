# Quant Sports Trading Project

Academic research project applying equities-style quantitative factor methodology to
soccer, using computer-vision-derived tracking features as the data-generation layer in
place of fundamentals data.

**Status: Phase 0 (scoping), in progress.** No factor has been tested. No result in this
repository has been validated. Nothing here has been traded with real money.

---

## Disclaimer

**This repository is research and educational material. It is not betting advice, not
gambling advice, not investment advice, and not a recommendation to place any wager or
trade any market.**

- Nothing here is an offer, solicitation, or inducement to gamble or to trade.
- No representation is made that any method described here is profitable. The explicit
  working expectation is that most tested hypotheses will fail, and a negative result is
  treated as a valid outcome of the research.
- Sports betting and prediction-market trading carry a substantial risk of financial
  loss. Legality varies by jurisdiction, and it is the reader's responsibility to know
  the rules that apply to them.
- Any figures, backtests, or simulated results are historical or hypothetical.
  Hypothetical performance has inherent limitations and does not indicate future
  results.
- The author accepts no liability for any decision taken on the basis of this material.
- If gambling is causing you harm, seek support from a service in your country.

## No data is redistributed here

This repository contains **code and aggregated statistical results only**. It
deliberately excludes, and will continue to exclude:

- Raw broadcast video, and any frames, crops, or images derived from it.
- Third-party datasets, including the Roboflow detection dataset used for training.
- Trained model weights.
- Market or odds data obtained from any platform.
- Credentials of any kind.

Reasons: broadcast footage is copyrighted, soccer video corpora restrict
redistribution (SoccerNet's raw video is NDA-gated, and the DFL Bundesliga Data
Shootout listing is closed), and market-data terms of service have not been cleared for
redistribution. As a result, the notebooks here are **not directly reproducible** by
cloning this repository. Anyone wanting to rerun them has to obtain the underlying data
themselves, under that data's own terms.

Detector training used the Roboflow Universe "Football Players Detection" dataset,
licensed CC BY 4.0. The dataset itself is not included here.

## Research design

The pipeline runs in seven conceptual stages: detection and tracking, pitch
calibration, identity and context, point-in-time feature extraction, factor
construction, market alignment, and evaluation.

The central hypothesis is about **state estimation, not event detection**. Published
work shows in-play betting prices absorb discrete events such as goals swiftly and
fully (Croxson & Reade 2014, *The Economic Journal*), and broadcast video reaches a
viewer later than the data feeds market makers use. So the question is not whether a
camera can see a goal first. It is whether slowly evolving tactical and physical state
(sustained pressure, territorial dominance, shape changes, fatigue proxies), which is
absent from discrete event feeds, carries information that prices do not already
reflect.

That is tested as a ladder of increasingly demanding hypotheses, from "tracking data
adds nothing" through "the effect survives realistic video latency" to "the effect
holds out of sample across seasons and competitions."

## Methodological discipline

The point of the project is the method, so the rules are fixed in advance:

- Every factor hypothesis is pre-registered in writing, with its definition and
  expected sign, before it is tested.
- Every factor tested is logged, including failures, because a multiple-testing
  correction computed over only the survivors is invalid.
- Point-in-time correctness is mandatory. No factor may use data that was not
  observable at its own timestamp, and market baselines must use the price actually
  tradeable at that moment, never a settled or revised price.
- Evaluation uses rank-IC with Newey-West standard errors, Benjamini-Hochberg FDR
  control across the full family of factors tried, the Probability of Backtest
  Overfitting, and the Deflated Sharpe Ratio.
- Effective sample size is treated as the number of matches, not the number of frames,
  since observations within a match are heavily autocorrelated.
- A holdout set is locked and left untouched until everything has already passed on
  development data.
- Structural decisions are logged with dated rationale as they are made, not
  retroactively, in an append-only log.

## Repository layout

```
CLAUDE.md          Project charter: decisions, constraints, open blockers
decision_log.md    Append-only record of what happened and why, with dates
research/          Point-in-time formal write-ups (Phase 0 scoping memo)
pipeline/
  detection/       Stage 1: detector training, evaluation, tracker
  common/          Shared helpers (annotation drawing)
experiments/       Diagnostic spikes, not pipeline code
learning.ipynb     Personal log of concepts and gotchas while building this
```

`decision_log.md` is the source of truth for project history. `CLAUDE.md` holds the
current constraints. Where they disagree, the log wins.

## Current state and known limitations

Stage 1 (detection and tracking) is **reopened**: it failed a pre-registered proxy
check on an independent tracking benchmark.

- A YOLO26n detector fine-tuned for player, goalkeeper, referee and ball, with
  ByteTrack for people and a separate Kalman filter for the ball, in a frame-by-frame
  design.
- A blind hand-labelling check on held-out frames of one clip measured people
  detection at about 97 to 100% and the ball at 91% precision but only 54% recall.
  These figures come from one clip of one match, labelled by the author.
- On 12 SoccerNet-Tracking test sequences (a different league), HOTA was 40.5 against
  a bar of 47.2 set before measuring. Detection is the main loss: boxes often miss or
  fit loosely, especially for small, distant players. Two candidate fixes (ball-filter
  changes, team-split tracking) were tested there and not adopted.
- Open problems: detection on unfamiliar footage, weak ball detection, identity
  errors (mostly track fragmentation), and per-frame latency above the real-time
  budget on CPU.

Major open blockers are listed in `CLAUDE.md`, including the choice of video source and
the mismatch between the era of available video and the era of available market price
history.

## License

No license is granted. All rights reserved by the author. The code is published for
reading and review, not for reuse. Open an issue if you want to discuss use.
