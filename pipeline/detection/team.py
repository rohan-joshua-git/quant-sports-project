import cv2
import numpy as np


def shirt_colours(frame, xyxy):
    """Median shirt colour (CIE Lab) of each person box, with grass pixels removed.

    Looks at the torso only (central half of the width, 15% to 50% of the height):
    the head, shorts and the pitch around the legs vary more than the shirt does.
    Returns an (N, 3) float array.
    """
    h_img, w_img = frame.shape[:2]
    out = np.zeros((len(xyxy), 3), dtype=np.float32)
    for i, (x1, y1, x2, y2) in enumerate(np.asarray(xyxy, dtype=float)):
        w, h = x2 - x1, y2 - y1
        cx1 = int(np.clip(x1 + 0.25 * w, 0, w_img - 1))
        cx2 = int(np.clip(x2 - 0.25 * w, cx1 + 1, w_img))
        cy1 = int(np.clip(y1 + 0.15 * h, 0, h_img - 1))
        cy2 = int(np.clip(y1 + 0.50 * h, cy1 + 1, h_img))
        crop = frame[cy1:cy2, cx1:cx2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        # OpenCV hue runs 0 to 180; pitch green sits around 35 to 85.
        grass = (hsv[:, 0] >= 35) & (hsv[:, 0] <= 85) & (hsv[:, 1] >= 50) & (hsv[:, 2] >= 40)
        keep = lab[~grass] if (~grass).sum() >= 5 else lab
        out[i] = np.median(keep, axis=0)
    return out


class TeamColours:
    """Two kit colours, fitted once per match, then a team label for any box.

    Fitting is a two-cluster k-means on shirt colours of outfield players. Labels 0
    and 1 are arbitrary: which team is "0" is not fixed across matches.
    """

    def __init__(self, iters=20):
        self.iters = iters
        self.centres = None

    @property
    def fitted(self):
        return self.centres is not None

    def fit(self, colours):
        x = np.asarray(colours, dtype=np.float32)
        if len(x) < 4:
            raise ValueError(f"need at least 4 player colours to fit, got {len(x)}")
        # Deterministic start: the colour farthest from the mean, then the colour
        # farthest from that one.
        a = x[np.argmax(((x - x.mean(0)) ** 2).sum(1))]
        b = x[np.argmax(((x - a) ** 2).sum(1))]
        c = np.stack([a, b])
        for _ in range(self.iters):
            lab = np.argmin(((x[:, None] - c[None]) ** 2).sum(2), axis=1)
            new = np.stack([x[lab == k].mean(0) if (lab == k).any() else c[k] for k in (0, 1)])
            if np.allclose(new, c):
                break
            c = new
        self.centres = c
        return self

    def predict(self, colours):
        x = np.asarray(colours, dtype=np.float32).reshape(-1, 3)
        return np.argmin(((x[:, None] - self.centres[None]) ** 2).sum(2), axis=1)
