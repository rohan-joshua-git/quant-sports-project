"""Hand-label spot-check for the Stage 1 tracker (Track 1 step 8).

Measures real error rates on frames 300 to 749 of the cached DFL clip, which were
held back from all tuning (decision_log.md, 2026-09-25). Run from the project root,
in this order:

    python experiments/spotcheck.py extract        # frozen tracker over the clip, saves frames + predictions
    python experiments/spotcheck.py label-ball     # ~100 frames: click the ball, or n = not visible
    python experiments/spotcheck.py label-people   # ~20 frames: true class of each box, then click misses
    python experiments/spotcheck.py switches       # watch the ID video, mark every ID swap
    python experiments/spotcheck.py score          # error rates with 95% intervals

Label the ball and people before watching the ID video, so its drawn ball and
classes can't anchor your labels. Labelling is blind: the ball labeller never shows
the tracker's ball, and the people labeller shows boxes but not their predicted
class. Labels save after every action; quit any time and rerun to resume.

Everything goes to experiments/spotcheck/, which is gitignored (derived frames).
"""
import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SRC = ROOT / "Tester video" / "08fd33_4.mp4"
WEIGHTS = ROOT / "pipeline" / "detection" / "football_yolo26n_best.pt"
OUT = ROOT / "experiments" / "spotcheck"
FRAMES = OUT / "frames"          # full-resolution sampled frames
IDS = OUT / "ids"                # annotated frames 300-749 for the ID-switch review
PRED = OUT / "predictions.json"
BALL_LABELS = OUT / "labels_ball.json"
PEOPLE_LABELS = OUT / "labels_people.json"
SWITCH_LABELS = OUT / "labels_switches.json"
RESULTS = OUT / "results.json"

FIRST, LAST = 300, 749           # held-out range
N_BALL, N_PEOPLE = 100, 20
ID_VIEW_W = 1280                 # width of the ID-review frames
BALL_RADII = (10, 20)            # px at full resolution; a ball is about 15 px across
ZOOM_HALF = 60                   # ball zoom shows a 120 x 120 px crop
REAL = ("player", "goalkeeper", "referee")
PEOPLE_KEYS = {"p": "player", "g": "goalkeeper", "r": "referee",
               "x": "not_a_person", "d": "duplicate"}
SWITCH_KEYS = {"c": "cross_team", "t": "teammate", "o": "other"}


def load(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def save(path, data):
    path.write_text(json.dumps(data, indent=1))


# --------------------------------------------------------------------------- extract

def extract():
    import copy
    from pipeline.common.drawing import annotate_tracks
    from pipeline.detection.tracker import Tracker

    FRAMES.mkdir(parents=True, exist_ok=True)
    IDS.mkdir(parents=True, exist_ok=True)
    ball_frames = set(np.linspace(FIRST, LAST, N_BALL).round().astype(int).tolist())
    people_frames = set(np.linspace(FIRST + 5, LAST - 4, N_PEOPLE).round().astype(int).tolist())

    tracker = Tracker(str(WEIGHTS))  # frozen defaults, see decision_log.md 2026-09-25
    names = tracker.model.names
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    preds = {
        "settings": {k: getattr(tracker, k) for k in (
            "conf", "imgsz", "nms_threshold", "cross_class_nms_threshold", "ball_conf",
            "max_ball_miss", "ball_gate", "ball_reacquire_after", "device")},
        "commit": commit,
        "ball": {},
        "people": {},
    }

    cap = cv2.VideoCapture(str(SRC))
    assert cap.isOpened(), f"Failed to open {SRC}"
    # Start at frame 0 so ByteTrack and the Kalman filter reach frame 300 warmed up,
    # as they would mid-match. Nothing before frame 300 is saved or scored.
    for i in range(LAST + 1):
        ok, frame = cap.read()
        if not ok:
            break
        tracks, _ = tracker.track_frame(frame)
        if i < FIRST:
            continue

        is_ball = tracks.class_id == tracker.ball_class_id
        if i in ball_frames or i in people_frames:
            cv2.imwrite(str(FRAMES / f"{i}.png"), frame)
        if i in ball_frames:
            ball = None
            if is_ball.any():
                k = int(np.flatnonzero(is_ball)[0])
                x1, y1, x2, y2 = tracks.xyxy[k].tolist()
                ball = {"x": (x1 + x2) / 2, "y": (y1 + y2) / 2,
                        "interpolated": bool(tracks.data["interpolated"][k]),
                        "sigma": float(tracks.data["ball_sigma"][k]),
                        "confidence": float(tracks.confidence[k])}
            preds["ball"][str(i)] = ball
        if i in people_frames:
            preds["people"][str(i)] = [
                {"xyxy": tracks.xyxy[k].tolist(), "class": names[int(tracks.class_id[k])],
                 "tracker_id": int(tracks.tracker_id[k]),
                 "confidence": float(tracks.confidence[k])}
                for k in np.flatnonzero(~is_ball)]

        # Draw on a downscaled copy with scaled boxes, so ID labels stay readable.
        s = ID_VIEW_W / frame.shape[1]
        small = cv2.resize(frame, (ID_VIEW_W, int(frame.shape[0] * s)))
        scaled = copy.deepcopy(tracks)
        scaled.xyxy = scaled.xyxy * s
        annotate_tracks(small, scaled, names)
        cv2.imwrite(str(IDS / f"{i}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cap.release()

    save(PRED, preds)
    print(f"Saved {len(preds['ball'])} ball frames, {len(preds['people'])} people frames, "
          f"{len(list(IDS.glob('*.jpg')))} ID-review frames to {OUT}")


# --------------------------------------------------------------------------- GUI

def gui():
    # This OpenCV build has no window support (opencv-python-headless wins), so the
    # labelling windows use matplotlib with the Tk backend.
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    for key in list(plt.rcParams):
        if key.startswith("keymap."):
            plt.rcParams[key] = []   # free every key for labelling ('s' = save, 'q' = quit, ...)
    return plt


def rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def require_predictions():
    preds = load(PRED, None)
    if preds is None:
        sys.exit("No predictions yet: run `python experiments/spotcheck.py extract` first.")
    return preds


class BallLabeller:
    """Click near the ball to zoom in, then click its exact centre."""

    def __init__(self, plt, preds):
        self.plt = plt
        self.frames = sorted(int(f) for f in preds["ball"])
        self.labels = load(BALL_LABELS, {})
        self.i = next((j for j, f in enumerate(self.frames) if str(f) not in self.labels),
                      len(self.frames))
        self.zoom = None   # top-left corner of the zoom crop while zoomed in
        self.fig, self.ax = plt.subplots(figsize=(15, 9))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.show()

    def show(self):
        self.ax.clear()
        self.ax.set_axis_off()
        if self.i >= len(self.frames):
            self.ax.set_title("All ball frames labelled. Close this window.")
            self.fig.canvas.draw_idle()
            return
        f = self.frames[self.i]
        img = rgb(FRAMES / f"{f}.png")
        self.size = img.shape[1], img.shape[0]
        if self.zoom is None:
            self.ax.imshow(img)
            title = (f"BALL  frame {f}  ({self.i + 1}/{len(self.frames)}, "
                     f"{len(self.labels)} labelled)\n"
                     "click near the ball to zoom in   |   n = not visible   "
                     "u = unsure   b = back")
        else:
            x0, y0 = self.zoom
            side = 2 * ZOOM_HALF
            # extent keeps click coordinates in full-frame pixels while zoomed
            self.ax.imshow(img[y0:y0 + side, x0:x0 + side],
                           extent=(x0 - 0.5, x0 + side - 0.5, y0 + side - 0.5, y0 - 0.5))
            title = (f"BALL  frame {f}  zoomed\n"
                     "click the exact ball centre   |   esc = back to full frame")
        self.ax.set_title(title)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None or self.i >= len(self.frames):
            return
        x, y = event.xdata, event.ydata
        if self.zoom is None:
            w, h = self.size
            x0 = int(min(max(x - ZOOM_HALF, 0), w - 2 * ZOOM_HALF))
            y0 = int(min(max(y - ZOOM_HALF, 0), h - 2 * ZOOM_HALF))
            self.zoom = (x0, y0)
        else:
            self.labels[str(self.frames[self.i])] = {"status": "visible", "x": x, "y": y}
            save(BALL_LABELS, self.labels)
            self.zoom = None
            self.i += 1
        self.show()

    def on_key(self, event):
        if event.key == "escape":
            self.zoom = None
        elif event.key in ("n", "u") and self.i < len(self.frames):
            status = "not_visible" if event.key == "n" else "unsure"
            self.labels[str(self.frames[self.i])] = {"status": status}
            save(BALL_LABELS, self.labels)
            self.zoom = None
            self.i += 1
        elif event.key == "b" and self.i > 0:
            self.i -= 1
            self.labels.pop(str(self.frames[self.i]), None)
            save(BALL_LABELS, self.labels)
            self.zoom = None
        else:
            return
        self.show()


class PeopleLabeller:
    """For each box: its true class. Then click any real person with no box."""

    def __init__(self, plt, preds):
        self.plt = plt
        self.preds = preds["people"]
        self.frames = sorted(int(f) for f in self.preds)
        self.labels = load(PEOPLE_LABELS, {})
        for f in self.frames:
            self.labels.setdefault(str(f), {"boxes": [None] * len(self.preds[str(f)]),
                                            "misses": [], "done": False})
        self.fi = next((k for k, f in enumerate(self.frames)
                        if not self.labels[str(f)]["done"]), len(self.frames))
        self.fig, (self.ax, self.ax_crop) = plt.subplots(
            1, 2, figsize=(17, 8), gridspec_kw={"width_ratios": [4, 1]})
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.load_frame()

    def load_frame(self):
        if self.fi < len(self.frames):
            f = self.frames[self.fi]
            self.img = rgb(FRAMES / f"{f}.png")
            self.entry = self.labels[str(f)]
            self.boxes = self.preds[str(f)]
        self.show()

    @property
    def j(self):
        """Index of the first unlabelled box, or len(boxes) once all are done."""
        return next((k for k, lab in enumerate(self.entry["boxes"]) if lab is None),
                    len(self.boxes))

    def show(self):
        from matplotlib.patches import Rectangle
        for ax in (self.ax, self.ax_crop):
            ax.clear()
            ax.set_axis_off()
        if self.fi >= len(self.frames):
            self.ax.set_title("All people frames labelled. Close this window.")
            self.fig.canvas.draw_idle()
            return

        f, j = self.frames[self.fi], self.j
        self.ax.imshow(self.img)
        for k, b in enumerate(self.boxes):
            x1, y1, x2, y2 = b["xyxy"]
            current = k == j
            color = "cyan" if current else ("lime" if j >= len(self.boxes) else "0.6")
            self.ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                        edgecolor=color, linewidth=2.5 if current else 0.8))
        for mx, my in self.entry["misses"]:
            self.ax.plot(mx, my, "rx", markersize=12, markeredgewidth=3)

        head = f"PEOPLE  frame {f}  ({self.fi + 1}/{len(self.frames)})"
        if j < len(self.boxes):
            title = (f"{head}  box {j + 1}/{len(self.boxes)}\n"
                     "true class of the CYAN box:  p player  g goalkeeper  r referee  "
                     "x not a person  d duplicate (2nd box on someone)  |  b = back")
            x1, y1, x2, y2 = map(int, self.boxes[j]["xyxy"])
            pad = 40
            cx0, cy0 = max(x1 - pad, 0), max(y1 - pad, 0)
            crop = self.img[cy0:y2 + pad, cx0:x2 + pad]
            if crop.size:
                self.ax_crop.imshow(crop)
                self.ax_crop.add_patch(Rectangle((x1 - cx0, y1 - cy0), x2 - x1, y2 - y1,
                                                 fill=False, edgecolor="cyan", linewidth=1.5))
        else:
            title = (f"{head}  all boxes done\n"
                     "click every player, goalkeeper or referee WITHOUT a box (red x)  |  "
                     "z = undo  enter = next frame  b = back to last box")
        self.ax.set_title(title, fontsize=10)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if (self.fi >= len(self.frames) or self.j < len(self.boxes)
                or event.inaxes is not self.ax or event.xdata is None):
            return
        self.entry["misses"].append([event.xdata, event.ydata])
        save(PEOPLE_LABELS, self.labels)
        self.show()

    def on_key(self, event):
        if self.fi >= len(self.frames):
            return
        j = self.j
        if j < len(self.boxes) and event.key in PEOPLE_KEYS:
            self.entry["boxes"][j] = PEOPLE_KEYS[event.key]
        elif event.key == "b" and j > 0:
            self.entry["boxes"][j - 1] = None
        elif j >= len(self.boxes) and event.key == "z" and self.entry["misses"]:
            self.entry["misses"].pop()
        elif j >= len(self.boxes) and event.key == "enter":
            self.entry["done"] = True
            save(PEOPLE_LABELS, self.labels)
            self.fi += 1
            self.load_frame()
            return
        else:
            return
        save(PEOPLE_LABELS, self.labels)
        self.show()


class SwitchReviewer:
    """Play the ID video and mark every moment two IDs swap people."""

    def __init__(self, plt):
        self.files = sorted(IDS.glob("*.jpg"), key=lambda p: int(p.stem))
        if not self.files:
            sys.exit("No ID frames yet: run `python experiments/spotcheck.py extract` first.")
        self.events = load(SWITCH_LABELS, [])
        self.i, self.playing = 0, False
        self.fig, self.ax = plt.subplots(figsize=(15, 9))
        self.ax.set_axis_off()
        self.im = self.ax.imshow(rgb(self.files[0]))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.timer = self.fig.canvas.new_timer(interval=40)   # 25 fps
        self.timer.add_callback(self.tick)
        self.show()

    def frame_no(self):
        return int(self.files[self.i].stem)

    def show(self):
        self.im.set_data(rgb(self.files[self.i]))
        here = [e["type"] for e in self.events if e["frame"] == self.frame_no()]
        counts = Counter(e["type"] for e in self.events)
        self.ax.set_title(
            f"ID SWITCHES  frame {self.frame_no()}  {'playing' if self.playing else 'paused'}"
            f"   marked: cross-team {counts['cross_team']}, teammate {counts['teammate']}, "
            f"other {counts['other']}" + (f"   <- {', '.join(here)} here" if here else "") +
            "\nspace play/pause  a/d -1/+1 frame  s/w -10/+10  |  "
            "c cross-team swap  t teammate swap  o other ID error  z undo", fontsize=10)
        self.fig.canvas.draw_idle()

    def step(self, n):
        self.i = min(max(self.i + n, 0), len(self.files) - 1)

    def tick(self):
        if self.playing:
            if self.i >= len(self.files) - 1:
                self.playing = False
                self.timer.stop()
            else:
                self.step(1)
            self.show()

    def on_key(self, event):
        key = event.key
        if key == " ":
            self.playing = not self.playing
            (self.timer.start if self.playing else self.timer.stop)()
        elif key in ("a", "d", "s", "w"):
            self.playing = False
            self.timer.stop()
            self.step({"a": -1, "d": 1, "s": -10, "w": 10}[key])
        elif key in SWITCH_KEYS:
            self.events.append({"frame": self.frame_no(), "type": SWITCH_KEYS[key]})
            save(SWITCH_LABELS, self.events)
        elif key == "z" and self.events:
            self.events.pop()
            save(SWITCH_LABELS, self.events)
        else:
            return
        self.show()


def run_gui(kind):
    plt = gui()
    if kind == "switches":
        app = SwitchReviewer(plt)
    else:
        preds = require_predictions()
        app = (BallLabeller if kind == "ball" else PeopleLabeller)(plt, preds)
    plt.show()
    return app


# --------------------------------------------------------------------------- score

def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion k/n (better than +-1.96 SE at small n)."""
    if n == 0:
        return math.nan, math.nan
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def rate(k, n):
    if n == 0:
        return "n/a (0 cases)"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.1%}  (95% CI {lo:.1%} to {hi:.1%})"


def score_ball(preds, labels):
    out = {}
    usable = {f: lab for f, lab in labels.items() if lab["status"] != "unsure"}
    visible = {f: lab for f, lab in usable.items() if lab["status"] == "visible"}
    hidden = [f for f, lab in usable.items() if lab["status"] == "not_visible"]
    dist = lambda p, lab: math.hypot(p["x"] - lab["x"], p["y"] - lab["y"])

    print(f"\nBALL  {len(labels)} frames labelled: {len(visible)} visible, {len(hidden)} not "
          f"visible, {len(labels) - len(usable)} unsure (excluded)")
    for r in BALL_RADII:
        tp = fp = 0
        errors = []
        for f, lab in usable.items():
            p = preds["ball"].get(f)
            if p is None or p["interpolated"]:
                continue
            if lab["status"] == "visible" and dist(p, lab) <= r:
                tp += 1
                errors.append(dist(p, lab))
            else:
                fp += 1
        out[f"detector_r{r}"] = {"tp": tp, "fp": fp, "visible": len(visible)}
        print(f"  Detector, correct = within {r} px of the labelled centre")
        print(f"    precision {rate(tp, tp + fp)}")
        print(f"    recall    {rate(tp, len(visible))}")
        if errors:
            print(f"    median error of correct detections {np.median(errors):.1f} px")

        reported = sum(1 for f, lab in visible.items()
                       if preds["ball"].get(f) is not None and dist(preds["ball"][f], lab) <= r)
        print(f"    tracker incl. Kalman predictions, ball found within {r} px: "
              f"{rate(reported, len(visible))}")
        out[f"tracker_r{r}"] = {"found": reported, "visible": len(visible)}

    pred_frames = [(f, lab) for f, lab in usable.items()
                   if preds["ball"].get(f) is not None and preds["ball"][f]["interpolated"]]
    in_view = [(f, lab) for f, lab in pred_frames if lab["status"] == "visible"]
    within_2s = sum(1 for f, lab in in_view
                    if dist(preds["ball"][f], lab) <= 2 * preds["ball"][f]["sigma"])
    print(f"  Kalman predictions: {len(pred_frames)} frames, ball visible in {len(in_view)}")
    print(f"    truth inside the 2-sigma circle {rate(within_2s, len(in_view))}  "
          "(about 95% if ball_sigma is honest)")
    ghosts = sum(1 for f in hidden if preds["ball"].get(f) is not None)
    print(f"  ball reported while not visible {rate(ghosts, len(hidden))}")
    out["kalman"] = {"predicted": len(pred_frames), "visible": len(in_view), "within_2sigma": within_2s}
    out["ghosts"] = {"reported": ghosts, "not_visible": len(hidden)}
    return out


def score_people(preds, labels):
    done = {f: e for f, e in labels.items() if e["done"]}
    confusion = Counter()
    misses = 0
    for f, e in done.items():
        for box, lab in zip(preds["people"][f], e["boxes"]):
            confusion[(box["class"], lab)] += 1
        misses += len(e["misses"])
    n_boxes = sum(confusion.values())
    n_real = sum(v for (_, t), v in confusion.items() if t in REAL)
    n_right = sum(v for (p, t), v in confusion.items() if p == t)

    print(f"\nPEOPLE  {len(done)} frames, {n_boxes} boxes, {misses} people missed")
    print(f"  precision (box is a real person, not junk or a duplicate) {rate(n_real, n_boxes)}")
    print(f"  recall (real people that got a box)                      {rate(n_real, n_real + misses)}")
    print(f"  class correct, among real people                         {rate(n_right, n_real)}")
    cols = list(REAL) + ["not_a_person", "duplicate"]
    print("  rows = predicted class, columns = true label")
    print("  " + " " * 12 + "".join(f"{c[:10]:>12}" for c in cols))
    for p in REAL:
        print(f"  {p:<12}" + "".join(f"{confusion[(p, c)]:>12}" for c in cols))
    ref_as_player = confusion[("player", "referee")]
    player_as_ref = confusion[("referee", "player")]
    print(f"  referee boxed as player: {ref_as_player},  player boxed as referee: {player_as_ref}")
    return {"frames": len(done), "boxes": n_boxes, "real": n_real, "class_correct": n_right,
            "misses": misses, "confusion": {f"{p}->{t}": v for (p, t), v in confusion.items()}}


def score_switches(events, n_frames):
    counts = Counter(e["type"] for e in events)
    seconds = n_frames / 25
    print(f"\nID SWITCHES  over {n_frames} frames ({seconds:.0f} s)")
    for t in SWITCH_KEYS.values():
        print(f"  {t:<11} {counts[t]:>3}   ({counts[t] / seconds * 60:.1f} per minute)")
    return dict(counts)


def score():
    preds = require_predictions()
    results = {"commit": preds.get("commit"), "settings": preds["settings"]}
    print(f"Tracker commit {preds.get('commit')}, settings {preds['settings']}")
    ball = load(BALL_LABELS, {})
    people = load(PEOPLE_LABELS, {})
    events = load(SWITCH_LABELS, None)
    if ball:
        results["ball"] = score_ball(preds, ball)
    if any(e["done"] for e in people.values()):
        results["people"] = score_people(preds, people)
    if events is not None:
        results["switches"] = score_switches(events, len(list(IDS.glob("*.jpg"))))
    print("\nCaveats: one clip, one match, one labeller (who also built the tracker). "
          "Neighbouring frames are correlated, so the intervals are narrower than the "
          "truth. Ball radius and duplicate rules are definitions, not ground truth.")
    save(RESULTS, results)
    print(f"Saved {RESULTS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["extract", "label-ball", "label-people",
                                         "switches", "score"])
    step = parser.parse_args().step
    if step == "extract":
        extract()
    elif step == "score":
        score()
    else:
        run_gui({"label-ball": "ball", "label-people": "people", "switches": "switches"}[step])
