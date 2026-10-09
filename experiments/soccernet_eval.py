"""SoccerNet-Tracking proxy check for the Stage 1 tracker.

Scores the tracker against SoccerNet-Tracking (SN-Tracking-2023 on Hugging Face,
ungated; the paper publishes test ground truth "so that researchers can benchmark
their results locally"). Different league and footage from DFL, so this is a proxy
for the design, not a measure of error on the project's own clip. Run from the
project root:

    python experiments/soccernet_eval.py fetch                      # pick sequences (fixed seed), download only those
    python experiments/soccernet_eval.py detect                     # YOLO once per frame, cached (SoccerNet + DFL clip)
    python experiments/soccernet_eval.py tune                       # ball settings grid, train + DFL only
    python experiments/soccernet_eval.py track V0 V1 V2 V3 --split train
    python experiments/soccernet_eval.py score V0 V1 V2 V3 --split train

Train sequences (and the spent DFL frames 300 to 749) are for tuning; test sequences
are for validation only, scored once against acceptance rules written into
decision_log.md (2026-10-09) beforehand. Everything goes to experiments/soccernet/,
which is gitignored (derived frames, no redistribution).
"""
import argparse
import configparser
import io
import json
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "experiments" / "soccernet"
DATA = OUT / "data"
DET = OUT / "detections"
TRACKS = OUT / "tracks"
SCORES = OUT / "scores"
SELECTION = OUT / "selection.json"
URL = "https://huggingface.co/datasets/SoccerNet/SN-Tracking-2023/resolve/main/{}.zip"
SEED = 20261009
N_SEQ = {"train": 6, "test": 12}

WEIGHTS = ROOT / "pipeline" / "detection" / "football_yolo26n_best.pt"
DFL_CLIP = ROOT / "Tester video" / "08fd33_4.mp4"
DFL_BALL_LABELS = ROOT / "experiments" / "spotcheck" / "labels_ball.json"
TEAM_FIT_FRAMES = 25      # kit colours fitted on each clip's first second
FOUND_PX, WRONG_PX = 10, 50
EPISODE_FRAMES = 25       # a wrong ball held this long (1 s) counts as a capture

# Variants fixed before the validation run (decision_log.md 2026-10-09). BALL_FIX is
# the setting chosen by `tune` on train + DFL only. tune's accel grid (4 to 16) ended
# with coverage still rising at its top, so it was widened to 24, 32, 48, 64 on the
# same tuning data; 64 is closest to 86.5% (58.6%).
BALL_FIX = {"ball_body_frac": 0.65, "ball_override_conf": 0.5, "ball_accel_std": 64.0}
VARIANTS = {
    "V0": {},
    "V1": BALL_FIX,
    "V2": {"team_split": True},
    "V3": {**BALL_FIX, "team_split": True},
}


def load(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def save(path, data):
    path.write_text(json.dumps(data, indent=1))


# --------------------------------------------------------------------------- fetch

class HttpFile(io.RawIOBase):
    """Read-only, seekable view of a remote file through HTTP range requests.

    zipfile only needs the central directory plus the members asked for, so a few
    sequences come out of an 8.7 GB archive without downloading the rest.
    """

    def __init__(self, url, block=1 << 20):
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.url = r.geturl()          # follow the CDN redirect once
            self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.block = block
        self.cache = {}

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def _get(self, start, end):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return r.read()
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if n > self.block:                  # big member data: one direct request
            data = self._get(self.pos, self.pos + n)
        else:                               # small reads (headers): cached blocks
            data = b""
            while len(data) < n:
                b0 = (self.pos + len(data)) // self.block
                if b0 not in self.cache:
                    s = b0 * self.block
                    self.cache[b0] = self._get(s, min(s + self.block, self.size))
                    if len(self.cache) > 64:          # keep memory bounded
                        self.cache.pop(next(iter(self.cache)))
                off = self.pos + len(data) - b0 * self.block
                data += self.cache[b0][off:off + n - len(data)]
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def fetch():
    DATA.mkdir(parents=True, exist_ok=True)
    selection = load(SELECTION, None)
    zips = {split: zipfile.ZipFile(HttpFile(URL.format(split))) for split in N_SEQ}

    if selection is None:
        # Chosen by seed from the full list before any frame is looked at.
        rng = np.random.default_rng(SEED)
        selection = {}
        for split, z in zips.items():
            seqs = sorted({n.split("/")[1] for n in z.namelist()
                           if n.count("/") >= 2 and n.split("/")[1]})
            pick = rng.choice(len(seqs), size=N_SEQ[split], replace=False)
            selection[split] = sorted(seqs[i] for i in pick)
            print(f"{split}: {len(seqs)} sequences, picked {selection[split]}")
        save(SELECTION, selection)

    for split, seqs in selection.items():
        z = zips[split]
        http = z.fp
        for seq in seqs:
            infos = [i for i in z.infolist()
                     if i.filename.startswith(f"{split}/{seq}/") and not i.is_dir()]
            todo = [i for i in infos if not (DATA / i.filename).exists()]
            if not todo:
                print(f"{split}/{seq}: complete", flush=True)
                continue
            # A sequence's members sit next to each other in the archive: fetch the
            # whole span in one request (many small requests ran ~10x slower).
            t0 = time.time()
            start = min(i.header_offset for i in todo)
            end = max(i.header_offset + 30 + len(i.orig_filename.encode()) + 1024 + i.compress_size
                      for i in todo)
            buf = http._get(start, min(end, http.size))
            for i in todo:
                h = i.header_offset - start
                assert buf[h:h + 4] == b"PK\x03\x04", f"bad local header for {i.filename}"
                name_len, extra_len = struct.unpack("<HH", buf[h + 26:h + 30])
                d = h + 30 + name_len + extra_len
                raw = buf[d:d + i.compress_size]
                if i.compress_type == zipfile.ZIP_DEFLATED:
                    data = zlib.decompress(raw, -15)
                elif i.compress_type == zipfile.ZIP_STORED:
                    data = raw
                else:
                    raise ValueError(f"unsupported compression {i.compress_type}")
                assert zlib.crc32(data) == i.CRC, f"CRC mismatch for {i.filename}"
                dest = DATA / i.filename
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Write then rename, so an interrupted run never leaves a truncated
                # file that the resume check would count as done.
                tmp = dest.with_name(dest.name + ".part")
                tmp.write_bytes(data)
                tmp.replace(dest)
            print(f"{split}/{seq}: {len(todo)} files, {len(buf) / 1e6:.0f} MB "
                  f"in {time.time() - t0:.0f} s", flush=True)


# --------------------------------------------------------------------------- clips

def clips(split):
    if split == "dfl":
        return ["08fd33_4"]
    return load(SELECTION, {})[split]


def frames(split, name):
    """BGR frames in order, frame index 0 first (SoccerNet's 000001.jpg is index 0)."""
    import cv2
    if split == "dfl":
        cap = cv2.VideoCapture(str(DFL_CLIP))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
        cap.release()
    else:
        for p in sorted((DATA / split / name / "img1").glob("*.jpg")):
            yield cv2.imread(str(p))


# --------------------------------------------------------------------------- detect

def detect():
    import supervision as sv
    from pipeline.detection.tracker import Tracker

    DET.mkdir(parents=True, exist_ok=True)
    tracker = Tracker(str(WEIGHTS))          # same model and predict settings as live
    for split in ("dfl", "train", "test"):
        for name in clips(split):
            path = DET / f"{split}_{name}.npz"
            if path.exists():
                continue
            if split != "dfl":
                seqinfo = configparser.ConfigParser()
                seqinfo.read(DATA / split / name / "seqinfo.ini")
                n_jpg = len(list((DATA / split / name / "img1").glob("*.jpg")))
                if not seqinfo.has_section("Sequence") or n_jpg != seqinfo.getint("Sequence", "seqLength"):
                    print(f"{split}/{name}: incomplete download ({n_jpg} frames), skipped", flush=True)
                    continue
            t0 = time.time()
            parts = []
            n = 0
            for i, frame in enumerate(frames(split, name)):
                raw = tracker.model.predict(frame, conf=tracker.conf, imgsz=tracker.imgsz,
                                            device=tracker.device, verbose=False)[0]
                d = sv.Detections.from_ultralytics(raw)
                parts.append((np.full(len(d), i), d.xyxy, d.confidence, d.class_id))
                n = i + 1
            np.savez_compressed(path, frame=np.concatenate([p[0] for p in parts]),
                                xyxy=np.concatenate([p[1] for p in parts]),
                                conf=np.concatenate([p[2] for p in parts]),
                                cls=np.concatenate([p[3] for p in parts]), n_frames=n)
            print(f"{split}/{name}: {n} frames in {time.time() - t0:.0f} s", flush=True)


def cached_detections(split, name, names):
    """Per-frame supervision Detections, rebuilt exactly as from_ultralytics makes them."""
    import supervision as sv
    z = np.load(DET / f"{split}_{name}.npz")
    out = []
    for i in range(int(z["n_frames"])):
        m = z["frame"] == i
        cls = z["cls"][m]
        out.append(sv.Detections(xyxy=z["xyxy"][m], confidence=z["conf"][m], class_id=cls,
                                 data={"class_name": np.array([names[int(c)] for c in cls])}))
    return out


# --------------------------------------------------------------------------- track

def run_tracker(split, name, settings, tracker_cache={}):
    """Replay one clip's cached detections through a Tracker with `settings`."""
    from pipeline.detection.tracker import Tracker

    key = json.dumps(settings, sort_keys=True)
    if key not in tracker_cache:
        tracker_cache.clear()                 # one model in memory at a time
        tracker_cache[key] = Tracker(str(WEIGHTS), **settings)
    tracker = tracker_cache[key]
    tracker.reset()
    dets = cached_detections(split, name, tracker.model.names)

    frame_iter = None
    if settings.get("team_split"):
        # Kit colours fitted on the first second, then tracking starts at frame 0:
        # a 1 s lookahead on a per-match constant (live use fits before kick-off).
        first = []
        for i, frame in enumerate(frames(split, name)):
            if i >= TEAM_FIT_FRAMES:
                break
            first.append((frame, dets[i]))
        tracker.fit_teams(first)
        frame_iter = frames(split, name)

    rows = []
    for i, det in enumerate(dets):
        frame = next(frame_iter) if frame_iter is not None else None
        t = tracker.track_detections(frame, det)
        rows.append((np.full(len(t), i), t.xyxy, t.tracker_id, t.class_id, t.data["team"],
                     t.data["interpolated"], t.data["ball_sigma"], t.confidence))
    keys = ("frame", "xyxy", "tid", "cls", "team", "interp", "sigma", "conf")
    out = {k: np.concatenate([r[j] for r in rows]) for j, k in enumerate(keys)}
    out["n_frames"] = len(dets)
    out["ball_class_id"] = tracker.ball_class_id
    return out


def track(variants, split):
    if split == "test" and not BALL_FIX:
        sys.exit("BALL_FIX is empty: run `tune` and write its result into BALL_FIX before "
                 "touching the validation sequences.")
    for v in variants:
        (TRACKS / v).mkdir(parents=True, exist_ok=True)
        for name in clips(split):
            t0 = time.time()
            np.savez_compressed(TRACKS / v / f"{split}_{name}.npz",
                                **run_tracker(split, name, VARIANTS[v]))
            print(f"{v} {split}/{name}: {time.time() - t0:.0f} s", flush=True)


# --------------------------------------------------------------------------- ground truth

def load_gt(split, name):
    """SoccerNet ground truth: rows (frame, id, xyxy) and per-id (kind, side)."""
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read(DATA / split / name / "gameinfo.ini")
    info = {}
    for k, v in cfg["Sequence"].items():
        if k.startswith("trackletid_"):
            desc = v.split(";")[0].strip()
            kind = next((c for c in ("player", "goalkeeper", "referee", "ball") if desc.startswith(c)),
                        "other")
            side = "left" if "left" in desc else ("right" if "right" in desc else None)
            info[int(k.split("_")[1])] = (kind, side)
    g = np.loadtxt(DATA / split / name / "gt" / "gt.txt", delimiter=",", ndmin=2)
    xyxy = np.column_stack([g[:, 2], g[:, 3], g[:, 2] + g[:, 4], g[:, 3] + g[:, 5]])
    return {"frame": g[:, 0].astype(int) - 1, "id": g[:, 1].astype(int), "xyxy": xyxy}, info


def gt_ball_centres(split, name):
    """{frame: (k, 2) array of true ball centres} for frames where the ball is known."""
    if split == "dfl":
        labels = load(DFL_BALL_LABELS, {})
        return {int(f): np.array([[lab["x"], lab["y"]]]) for f, lab in labels.items()
                if lab["status"] == "visible"}
    gt, info = load_gt(split, name)
    is_ball = np.array([info.get(i, ("other",))[0] == "ball" for i in gt["id"]], dtype=bool)
    out = {}
    for f, b in zip(gt["frame"][is_ball], gt["xyxy"][is_ball]):
        c = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]])
        out[f] = np.vstack([out[f], c]) if f in out else c
    return out


# --------------------------------------------------------------------------- metrics

def iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def trackeval_metrics():
    # TrackEval (pinned commit 12c8791) still uses np.float / np.int, removed in
    # NumPy 2. np.bool must not be aliased: NumPy 2 has it again.
    np.float, np.int = float, int
    from trackeval.metrics import CLEAR, HOTA, Identity
    return HOTA(), CLEAR({"PRINT_CONFIG": False}), Identity({"PRINT_CONFIG": False})


def mot_data(n_frames, gt_frame, gt_id, gt_box, pr_frame, pr_id, pr_box):
    """TrackEval's per-sequence input: per-frame IDs (0-based, contiguous) and IoUs."""
    gmap = {g: i for i, g in enumerate(np.unique(gt_id))}
    pmap = {p: i for i, p in enumerate(np.unique(pr_id))}
    data = {"num_timesteps": n_frames, "num_gt_ids": len(gmap), "num_tracker_ids": len(pmap),
            "num_gt_dets": len(gt_id), "num_tracker_dets": len(pr_id),
            "gt_ids": [], "tracker_ids": [], "similarity_scores": []}
    for t in range(n_frames):
        gm, pm = gt_frame == t, pr_frame == t
        data["gt_ids"].append(np.array([gmap[x] for x in gt_id[gm]], dtype=int))
        data["tracker_ids"].append(np.array([pmap[x] for x in pr_id[pm]], dtype=int))
        data["similarity_scores"].append(iou(gt_box[gm], pr_box[pm]))
    return data


def ball_metrics(pred, gt_balls, n_frames, consecutive=True):
    """Ball accuracy against true centres. `gt_balls` only holds frames where the ball
    is known, so every count is over those frames."""
    is_ball = pred["cls"] == pred["ball_class_id"]
    pb = {int(f): (x, i, s) for f, x, i, s in zip(pred["frame"][is_ball], pred["xyxy"][is_ball],
                                                     pred["interp"][is_ball], pred["sigma"][is_ball])}
    m = dict(gt_frames=0, found=0, det_tp=0, det_fp=0, wrong=0,
             predicted=0, within_2sigma=0, episodes=0, episode_frames=0)
    run = 0
    for f in range(n_frames):
        wrong_here = False
        if f in gt_balls:
            m["gt_frames"] += 1
            if f in pb:
                box, interp, sigma = pb[f]
                c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                d = np.min(np.hypot(*(gt_balls[f] - c).T))
                m["found"] += d <= FOUND_PX
                if interp:
                    m["predicted"] += 1
                    m["within_2sigma"] += d <= 2 * sigma
                else:
                    m["det_tp" if d <= FOUND_PX else "det_fp"] += 1
                if d > WRONG_PX:
                    m["wrong"] += 1
                    wrong_here = True
        if consecutive:
            if wrong_here:
                run += 1
            else:
                if run >= EPISODE_FRAMES:
                    m["episodes"] += 1
                    m["episode_frames"] += run
                run = 0
    if consecutive and run >= EPISODE_FRAMES:
        m["episodes"] += 1
        m["episode_frames"] += run
    return {k: int(v) for k, v in m.items()}


def id_errors(n_frames, gt, info, pred):
    """Follow each true person's matched prediction ID (IoU >= 0.5, per-frame Hungarian
    matching) and classify every change. Also scores per-frame team labels."""
    from scipy.optimize import linear_sum_assignment
    side = {i: s for i, (k, s) in info.items()}
    last, owners = {}, {}
    counts = dict(cross_team=0, teammate=0, other=0, new_id=0)
    team_hits = {0: 0, 1: 0}                 # agreement if label 0 means left / right
    team_n = 0
    for t in range(n_frames):
        gm, pm = gt["frame"] == t, pred["frame"] == t
        ious = iou(gt["xyxy"][gm], pred["xyxy"][pm])
        if ious.size == 0:
            continue
        r, c = linear_sum_assignment(-ious)
        keep = ious[r, c] >= 0.5
        for g, p, team in zip(gt["id"][gm][r[keep]], pred["tid"][pm][c[keep]],
                              pred["team"][pm][c[keep]]):
            g, p = int(g), int(p)
            if g in last and last[g] != p:
                prev = owners.get(p, set()) - {g}
                if not prev:
                    counts["new_id"] += 1
                elif any(side.get(o) and side.get(g) and side[o] != side[g] for o in prev):
                    counts["cross_team"] += 1
                elif any(side.get(o) and side[o] == side.get(g) for o in prev):
                    counts["teammate"] += 1
                else:
                    counts["other"] += 1
            last[g] = p
            owners.setdefault(p, set()).add(g)
            if info.get(g, ("",))[0] == "player" and side.get(g) and team in (0, 1):
                team_n += 1
                team_hits[0] += (team == 0) == (side[g] == "left")
                team_hits[1] += (team == 0) == (side[g] == "right")
    counts["team_labels"] = team_n
    counts["team_correct"] = max(team_hits.values())   # labels 0/1 are arbitrary per clip
    return {k: int(v) for k, v in counts.items()}


# --------------------------------------------------------------------------- tune

TUNE_GRID = {"ball_body_frac": (None, 0.5, 0.65), "ball_override_conf": (None, 0.3, 0.5)}
TUNE_ACCEL = (4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0)


def tune():
    """Choose BALL_FIX on tuning data only (SoccerNet train + spent DFL frames).

    Rule fixed in advance: fewest wrong-ball frames on SoccerNet train with the found
    rate no more than 2 points below V0; then the accel_std whose 2-sigma coverage is
    closest to 86.5% (the 2D-Gaussian value)."""
    def evaluate(settings):
        tot = {"train": {}, "dfl": {}}
        for split in ("train", "dfl"):
            for name in clips(split):
                pred = run_tracker(split, name, settings)
                m = ball_metrics(pred, gt_ball_centres(split, name), pred["n_frames"],
                                 consecutive=(split != "dfl"))
                for k, v in m.items():
                    tot[split][k] = tot[split].get(k, 0) + v
        return tot

    def show(label, tot):
        s, d = tot["train"], tot["dfl"]
        cov = s["within_2sigma"] / max(s["predicted"], 1)
        print(f"{label:<44} SN found {s['found'] / s['gt_frames']:.1%} wrong {s['wrong']:>5} "
              f"episodes {s['episodes']:>3} cov2s {cov:.1%} | DFL found {d['found']}/{d['gt_frames']} "
              f"wrong {d['wrong']}", flush=True)

    results = {}
    for body in TUNE_GRID["ball_body_frac"]:
        for over in TUNE_GRID["ball_override_conf"]:
            settings = {"ball_body_frac": body, "ball_override_conf": over}
            results[(body, over)] = evaluate(settings)
            show(f"body {body} override {over}", results[(body, over)])
    base = results[(None, None)]["train"]
    base_found = base["found"] / base["gt_frames"]
    ok = {k: r for k, r in results.items()
          if r["train"]["found"] / r["train"]["gt_frames"] >= base_found - 0.02}
    body, over = min(ok, key=lambda k: (ok[k]["train"]["wrong"], k != (None, None)))
    print(f"\nchosen: ball_body_frac {body}, ball_override_conf {over}")

    cov = {}
    for a in TUNE_ACCEL:
        settings = {"ball_body_frac": body, "ball_override_conf": over, "ball_accel_std": a}
        r = evaluate(settings)
        show(f"  accel_std {a}", r)
        cov[a] = r["train"]["within_2sigma"] / max(r["train"]["predicted"], 1)
    a = min(cov, key=lambda k: abs(cov[k] - 0.865))
    fix = {"ball_body_frac": body, "ball_override_conf": over, "ball_accel_std": a}
    print(f"\nBALL_FIX = {fix}")
    save(OUT / "tuned.json", fix)


# --------------------------------------------------------------------------- score

def score(variants, split):
    SCORES.mkdir(parents=True, exist_ok=True)
    hota, clear, ident = trackeval_metrics()
    for v in variants:
        per_seq = {"people": {}, "all": {}}
        ball, ids = {}, {}
        for name in clips(split):
            pred = dict(np.load(TRACKS / v / f"{split}_{name}.npz"))
            gt, info = load_gt(split, name)
            n = int(pred["n_frames"])
            gt_ball = np.array([info.get(i, ("other",))[0] == "ball" for i in gt["id"]], dtype=bool)
            pr_ball = pred["cls"] == pred["ball_class_id"]
            for scope, gmask, pmask in (("people", ~gt_ball, ~pr_ball),
                                        ("all", np.ones_like(gt_ball), np.ones_like(pr_ball))):
                d = mot_data(n, gt["frame"][gmask], gt["id"][gmask], gt["xyxy"][gmask],
                             pred["frame"][pmask], pred["tid"][pmask], pred["xyxy"][pmask])
                per_seq[scope][name] = {"HOTA": hota.eval_sequence(d), "CLEAR": clear.eval_sequence(d),
                                        "Identity": ident.eval_sequence(d)}
            for k, val in ball_metrics(pred, gt_ball_centres(split, name), n).items():
                ball[k] = ball.get(k, 0) + val
            people_gt = {k: x[~gt_ball] for k, x in gt.items()}
            people_pr = {k: (x[~pr_ball] if k in ("frame", "xyxy", "tid", "team") else x)
                         for k, x in pred.items()}
            for k, val in id_errors(n, people_gt, info, people_pr).items():
                ids[k] = ids.get(k, 0) + val

        out = {"variant": v, "settings": VARIANTS[v], "split": split, "clips": clips(split),
               "ball": ball, "id_errors": ids}
        for scope in ("people", "all"):
            seqs = per_seq[scope]
            h = hota.combine_sequences({s: r["HOTA"] for s, r in seqs.items()})
            c = clear.combine_sequences({s: r["CLEAR"] for s, r in seqs.items()})
            i = ident.combine_sequences({s: r["Identity"] for s, r in seqs.items()})
            out[scope] = {"HOTA": float(h["HOTA"].mean() * 100), "DetA": float(h["DetA"].mean() * 100),
                          "AssA": float(h["AssA"].mean() * 100), "MOTA": float(c["MOTA"] * 100),
                          "IDSW": int(c["IDSW"]), "IDF1": float(i["IDF1"] * 100),
                          "per_clip_HOTA": {s: float(r["HOTA"]["HOTA"].mean() * 100)
                                            for s, r in seqs.items()}}
        save(SCORES / f"{split}_{v}.json", out)

        p, a = out["people"], out["all"]
        b = ball
        print(f"\n{v} on {split} ({len(clips(split))} clips) {VARIANTS[v]}")
        print(f"  people   HOTA {p['HOTA']:.1f}  DetA {p['DetA']:.1f}  AssA {p['AssA']:.1f}  "
              f"MOTA {p['MOTA']:.1f}  IDF1 {p['IDF1']:.1f}  IDSW {p['IDSW']}")
        print(f"  all+ball HOTA {a['HOTA']:.1f}  DetA {a['DetA']:.1f}  AssA {a['AssA']:.1f}")
        print(f"  ball     found<=10px {b['found']}/{b['gt_frames']} = {b['found'] / b['gt_frames']:.1%}  "
              f"wrong>50px {b['wrong']}  capture episodes {b['episodes']} ({b['episode_frames']} frames)  "
              f"detector precision {b['det_tp'] / max(b['det_tp'] + b['det_fp'], 1):.1%}  "
              f"2-sigma coverage {b['within_2sigma']}/{b['predicted']} = "
              f"{b['within_2sigma'] / max(b['predicted'], 1):.1%}")
        team = (f"  team labels correct {ids['team_correct']}/{ids['team_labels']} = "
                f"{ids['team_correct'] / ids['team_labels']:.1%}" if ids["team_labels"] else "")
        print(f"  ID changes: cross-team {ids['cross_team']}, teammate {ids['teammate']}, "
              f"other {ids['other']}, new ID {ids['new_id']}{team}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["fetch", "detect", "tune", "track", "score"])
    parser.add_argument("variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--split", choices=["train", "test"], default="train")
    args = parser.parse_args()
    if args.step == "fetch":
        fetch()
    elif args.step == "detect":
        detect()
    elif args.step == "tune":
        tune()
    elif args.step == "track":
        track(args.variants, args.split)
    else:
        score(args.variants, args.split)
