import cv2
import numpy as np

# BGR, not RGB
COLORS = {
    "player": (0, 0, 255),
    "goalkeeper": (0, 165, 255),
    "referee": (0, 255, 255),
    "ball": (0, 255, 0),
}


# Radius of the 95% circle of a 2D Gaussian, in per-axis standard deviations:
# sqrt of the chi-square(2) 95% point. A 2-sigma circle only holds 86.5%.
SIGMA_95 = 2.448


def draw_ellipse(frame, bbox, color, thickness=2):
    x1, y1, x2, y2 = map(int, bbox)
    cx = (x1 + x2) // 2
    # Sized from box height, not width: width jumps when arms swing or a leg kicks
    # out, height barely changes. 0.4 x height matches the old width-based ring
    # for a typical standing player box.
    a = max(1, int(0.4 * (y2 - y1)))
    b = max(1, int(0.35 * a))
    cv2.ellipse(frame, (cx, y2), (a, b), 0, 0, 360, color, thickness)
    return frame


def draw_triangle(frame, bbox, color, filled=True):
    x1, y1, x2, _ = map(int, bbox)
    cx = (x1 + x2) // 2
    size = 14
    tip_y = y1 - 2
    points = np.array([
        [cx, tip_y],
        [cx - size, tip_y - 2 * size],
        [cx + size, tip_y - 2 * size],
    ], dtype=np.int32)
    if filled:
        cv2.drawContours(frame, [points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [points], 0, (0, 0, 0), 2)
    else:
        cv2.drawContours(frame, [points], 0, color, 2)
    return frame


def draw_label(frame, bbox, text, color):
    x1, _, x2, y2 = map(int, bbox)
    cx = (x1 + x2) // 2
    cv2.putText(frame, text, (cx - 10, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (cx - 10, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA)
    return frame


def annotate_frame(frame, result):
    """Draw raw YOLO output (an ultralytics Result). No tracking, no IDs."""
    for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
        name = result.names[int(cls)]
        color = COLORS.get(name, (255, 255, 255))
        bbox = box.tolist()
        if name == "ball":
            draw_triangle(frame, bbox, color)
        else:
            draw_ellipse(frame, bbox, color)
    return frame


class BoxSmoother:
    """Per-ID exponential smoothing of boxes, for drawing only.

    Detector boxes jitter by a few pixels frame to frame, which makes rings and
    labels shimmer. Forward-only (each frame mixes in past frames, never future
    ones). Never feed smoothed boxes back into tracking or features: they lag.
    """

    def __init__(self, alpha=0.5, max_age=5):
        self.alpha = alpha          # weight on the new box; 1 = no smoothing
        self.max_age = max_age      # frames an unseen ID is remembered
        self.state = {}             # id -> (box, frames since last seen)

    def __call__(self, tracks):
        out = tracks.xyxy.astype(float).copy()
        seen = set()
        for i, tid in enumerate(tracks.tracker_id):
            tid = int(tid)
            if tid < 0:             # the ball (-1) moves too fast to smooth
                continue
            if tid in self.state:
                out[i] = self.alpha * out[i] + (1 - self.alpha) * self.state[tid][0]
            self.state[tid] = (out[i], 0)
            seen.add(tid)
        for tid in list(self.state):
            if tid not in seen:
                box, age = self.state[tid]
                if age + 1 > self.max_age:
                    del self.state[tid]
                else:
                    self.state[tid] = (box, age + 1)
        return out


def annotate_tracks(frame, tracks, names, show_ids=True, smoother=None):
    """Draw tracker output (a supervision Detections) with persistent track IDs.

    Predicted balls (Kalman estimate, not detected) are drawn as an outline so a
    guess never looks like an observation, with the 95% circle (2.45 sigma) showing
    how unsure the guess is. Pass a BoxSmoother, one per video, to steady the rings.
    """
    interpolated = tracks.data.get("interpolated")
    sigma = tracks.data.get("ball_sigma")
    boxes = smoother(tracks) if smoother is not None else tracks.xyxy
    for i in range(len(tracks)):
        bbox = boxes[i]
        name = names[int(tracks.class_id[i])]
        color = COLORS.get(name, (255, 255, 255))
        is_interp = bool(interpolated[i]) if interpolated is not None else False

        if name == "ball":
            if is_interp:
                draw_triangle(frame, bbox, color, filled=False)
                if sigma is not None and np.isfinite(sigma[i]):
                    x1, y1, x2, y2 = map(int, bbox)
                    radius = max(1, int(SIGMA_95 * sigma[i]))
                    cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), radius, color, 1)
            else:
                draw_triangle(frame, bbox, color)
        else:
            draw_ellipse(frame, bbox, color)
            if show_ids and tracks.tracker_id is not None:
                draw_label(frame, bbox, str(int(tracks.tracker_id[i])), color)
    return frame
