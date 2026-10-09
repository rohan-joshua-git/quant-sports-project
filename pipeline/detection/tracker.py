import time

import supervision as sv
from ultralytics import YOLO
import numpy as np

from pipeline.detection.ball_filter import BallKalman
from pipeline.detection.team import TeamColours, shirt_colours


class Tracker:
    def __init__(self, model_path, conf=0.1, imgsz=640, max_ball_miss=10,
                 nms_threshold=0.3, cross_class_nms_threshold=0.7, ball_conf=0.1,
                 device="cpu", ball_accel_std=4.0, ball_meas_std=2.0,
                 ball_gate=9.21, ball_reacquire_after=3,
                 ball_body_frac=None, ball_override_conf=None, team_split=False):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.nms_threshold = nms_threshold
        self.cross_class_nms_threshold = cross_class_nms_threshold
        self.ball_conf = ball_conf
        self.device = device
        self.max_ball_miss = max_ball_miss
        # 9.21 is the 99% point of a chi-square with 2 degrees of freedom: a real
        # ball lands inside the gate 99% of the time if the filter is right.
        self.ball_gate = ball_gate
        self.ball_reacquire_after = ball_reacquire_after
        # Ball detections centred on the upper `ball_body_frac` of a person box are
        # dropped (None = off): a shirt or shoulder was the false ball that captured
        # the filter in the 2026-10-05 spot-check.
        self.ball_body_frac = ball_body_frac
        # A detection outside the gate at this confidence or more, and above every
        # detection inside it, restarts the filter there (None = off).
        self.ball_override_conf = ball_override_conf
        # One ByteTrack per team (plus one for goalkeepers and referees), so an ID
        # can't jump between opponents. Needs fit_teams() first.
        self.team_split = team_split
        self.ball_filter = BallKalman(accel_std=ball_accel_std, meas_std=ball_meas_std)
        self.ball_class_id = next(i for i, name in self.model.names.items() if name == 'ball')
        self.player_class_id = next(i for i, name in self.model.names.items() if name == 'player')
        self.ball_track_id = -1
        self.reset()

    def reset(self):
        """Clear all per-video state, team colours included. Call before a new clip,
        then fit_teams() if team labels are wanted."""
        self.tracker = sv.ByteTrack()
        self.group_trackers = {}       # team-split mode: group -> ByteTrack
        self.id_map = {}               # team-split mode: (group, local id) -> global id
        self.teams = TeamColours()
        self.ball_filter.reset()
        self.ball_size = None
        self.ball_miss_count = 0

    def fit_teams(self, samples, min_conf=0.5):
        """Fit the two kit colours from (frame, detections) pairs, using confident
        player boxes only. Kit colours are fixed for a match, so in live use this
        runs once on frames from before kick-off or the first seconds of play."""
        colours = []
        for frame, det in samples:
            players = det[(det.class_id == self.player_class_id) & (det.confidence >= min_conf)]
            if len(players):
                colours.append(shirt_colours(frame, players.xyxy))
        self.teams.fit(np.concatenate(colours))

    @staticmethod
    def get_center_bbox(bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return cx, cy

    @staticmethod
    def get_foot_position(bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        return cx, y2

    def track_frame(self, frame):
        t0 = time.perf_counter()
        detection_raw = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                           device=self.device, verbose=False)[0]
        t1 = time.perf_counter()
        tracks = self.track_detections(frame, sv.Detections.from_ultralytics(detection_raw))
        self.last_timings['yolo'] = (t1 - t0) * 1000
        return tracks, detection_raw

    def track_detections(self, frame, detection_sv):
        """Everything after the detector, for one frame. Split out of track_frame so
        cached detections can be replayed through different tracker settings.
        `frame` is only read when team colours are fitted; it may be None otherwise.
        """
        t1 = time.perf_counter()
        # The ball stays out of ByteTrack: its tracks start only at conf >= 0.35
        # and follow by box overlap, which a small fast ball keeps failing.
        is_ball = detection_sv.class_id == self.ball_class_id
        balls = detection_sv[is_ball]
        people = detection_sv[~is_ball]

        people = people.with_nms(threshold=self.nms_threshold, class_agnostic=False)
        # One person boxed twice under different classes (e.g. a referee also
        # labelled player) overlaps at IoU 0.76 or more; two real people standing
        # close together stayed at 0.56 or less on the tuning frames.
        if self.cross_class_nms_threshold is not None:
            people = people.with_nms(threshold=self.cross_class_nms_threshold,
                                     class_agnostic=True)

        # Per-frame team label: 0 or 1 for players, -1 for everyone else or when no
        # colours are fitted.
        team = np.full(len(people), -1, dtype=int)
        if self.teams.fitted:
            is_player = people.class_id == self.player_class_id
            if is_player.any():
                team[is_player] = self.teams.predict(shirt_colours(frame, people.xyxy[is_player]))
        people.data['team'] = team
        t2 = time.perf_counter()

        if self.team_split:
            tracks = self._track_split(people)
        else:
            tracks = self.tracker.update_with_detections(people)
        t3 = time.perf_counter()

        balls = balls[balls.confidence >= self.ball_conf]
        if self.ball_body_frac is not None:
            balls = balls[~self._on_body(balls, people)]
        ball = self._update_ball(balls)

        # Columns must exist before _append_ball, which only pads existing keys.
        # ball_sigma is NaN for people: it only means something for the ball.
        if 'team' not in tracks.data:          # ByteTrack drops columns when empty
            tracks.data['team'] = np.full(len(tracks), -1, dtype=int)
        tracks.data['interpolated'] = np.zeros(len(tracks), dtype=bool)
        tracks.data['ball_sigma'] = np.full(len(tracks), np.nan, dtype=np.float32)

        if ball is not None:
            tracks = self._append_ball(tracks, *ball)

        t4 = time.perf_counter()
        # Per-stage milliseconds for the last frame, for latency breakdowns.
        self.last_timings = {
            'yolo': 0.0,
            'split_nms': (t2 - t1) * 1000,
            'bytetrack': (t3 - t2) * 1000,
            'ball': (t4 - t3) * 1000,
        }
        return tracks

    def _track_split(self, people):
        """Team-split tracking: each group gets its own ByteTrack, so a box can only
        take over an ID from the same group. Group 0/1 = team, 2 = goalkeepers and
        referees (their kits differ from both teams)."""
        if not self.teams.fitted:
            raise RuntimeError("team_split needs fit_teams() after reset()")
        group = np.where(people.class_id == self.player_class_id, people.data['team'], 2)
        out = []
        for g in (0, 1, 2):
            # Update every group every frame, even with no boxes, so lost tracks age.
            trk = self.group_trackers.setdefault(g, sv.ByteTrack())
            t = trk.update_with_detections(people[group == g])
            if len(t):
                t.tracker_id = np.array([self._global_id(g, int(i)) for i in t.tracker_id])
                out.append(t)
        if not out:
            empty = people[np.zeros(len(people), dtype=bool)]
            empty.tracker_id = np.zeros(0, dtype=int)
            return empty
        return sv.Detections.merge(out)

    def _global_id(self, group, local_id):
        # Each ByteTrack counts IDs from 1, so map (group, local) to one sequence.
        key = (group, local_id)
        if key not in self.id_map:
            self.id_map[key] = len(self.id_map) + 1
        return self.id_map[key]

    def _on_body(self, balls, people):
        """True for each ball detection centred on a person's upper body (central
        70% of the box width, top `ball_body_frac` of its height)."""
        if len(balls) == 0 or len(people) == 0:
            return np.zeros(len(balls), dtype=bool)
        cx = (balls.xyxy[:, 0] + balls.xyxy[:, 2]) / 2
        cy = (balls.xyxy[:, 1] + balls.xyxy[:, 3]) / 2
        x1, y1, x2, y2 = (people.xyxy[:, k] for k in range(4))
        w, h = x2 - x1, y2 - y1
        inside = ((cx[:, None] >= x1 + 0.15 * w) & (cx[:, None] <= x2 - 0.15 * w) &
                  (cy[:, None] >= y1) & (cy[:, None] <= y1 + self.ball_body_frac * h))
        return inside.any(axis=1)

    def _update_ball(self, balls):
        """Pick this frame's ball from the detections and advance the Kalman filter.

        Returns (bbox, confidence, interpolated, sigma_px), or None when there is
        no ball to report.
        """
        kf = self.ball_filter
        if kf.initialized:
            kf.predict()

        chosen = None
        restart = False
        if len(balls) > 0:
            centers = np.column_stack([(balls.xyxy[:, 0] + balls.xyxy[:, 2]) / 2,
                                       (balls.xyxy[:, 1] + balls.xyxy[:, 3]) / 2])
            if not kf.initialized:
                chosen, restart = int(np.argmax(balls.confidence)), True
            else:
                # Gate: only detections plausibly where the ball should be.
                d2 = np.array([kf.distance_sq(c) for c in centers])
                inside = np.flatnonzero(d2 <= self.ball_gate)
                outside = np.flatnonzero(d2 > self.ball_gate)
                best_out = (int(outside[np.argmax(balls.confidence[outside])])
                            if len(outside) else None)
                if (self.ball_override_conf is not None and best_out is not None
                        and balls.confidence[best_out] >= self.ball_override_conf
                        and (len(inside) == 0
                             or balls.confidence[best_out] > balls.confidence[inside].max())):
                    # A confident ball elsewhere beats weak ones near the prediction:
                    # stops a repeated low-confidence false ball holding the filter.
                    chosen, restart = best_out, True
                elif len(inside) > 0:
                    chosen = int(inside[np.argmax(balls.confidence[inside])])
                elif self.ball_miss_count >= self.ball_reacquire_after:
                    # Lost for a few frames while something ball-like shows up
                    # elsewhere: most likely a kick the filter could not follow.
                    chosen, restart = int(np.argmax(balls.confidence)), True

        if chosen is not None:
            if restart:
                kf.initiate(centers[chosen])
            else:
                kf.update(centers[chosen])
            self.ball_miss_count = 0
            bbox = np.array(balls.xyxy[chosen], copy=True)
            self.ball_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            return bbox, float(balls.confidence[chosen]), False, 0.0

        if not kf.initialized:
            return None
        self.ball_miss_count += 1
        if self.ball_miss_count > self.max_ball_miss:
            # Unseen too long for the prediction to mean anything: report no
            # ball rather than a ghost.
            kf.reset()
            return None

        cx, cy = kf.position
        w, h = self.ball_size
        bbox = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        return bbox, 0.5, True, kf.position_std

    def _append_ball(self, tracks, bbox, confidence, interpolated, sigma):
        ball_xyxy = np.array([bbox], dtype=np.float32)
        ball_confidence = np.array([confidence], dtype=np.float32)
        ball_class = np.array([self.ball_class_id], dtype=tracks.class_id.dtype)
        ball_tracker_id = np.array([self.ball_track_id], dtype=tracks.tracker_id.dtype)

        tracks.xyxy = np.vstack([tracks.xyxy, ball_xyxy])
        tracks.confidence = np.concatenate([tracks.confidence, ball_confidence])
        tracks.class_id = np.concatenate([tracks.class_id, ball_class])
        tracks.tracker_id = np.concatenate([tracks.tracker_id, ball_tracker_id])

        # Every array in .data must stay the same length as the boxes, or
        # indexing a Detections object later silently misaligns.
        for key, values in tracks.data.items():
            if key == 'interpolated':
                filler = np.array([interpolated])
            elif key == 'ball_sigma':
                filler = np.array([sigma], dtype=values.dtype)
            elif key == 'team':
                filler = np.array([-1], dtype=values.dtype)   # the ball has no team
            elif key == 'class_name':
                filler = np.array([self.model.names[self.ball_class_id]])
            elif values.dtype.kind in 'US':
                filler = np.array([''])
            else:
                filler = np.zeros((1,) + values.shape[1:], dtype=values.dtype)
            tracks.data[key] = np.concatenate([values, filler])

        return tracks
