import cv2
import numpy as np

# BGR, not RGB
COLORS = {
    "player": (0, 0, 255),
    "goalkeeper": (0, 165, 255),
    "referee": (0, 255, 255),
    "ball": (0, 255, 0),
}


def draw_ellipse(frame, bbox, color, thickness=2):
    x1,y1,x2,y2 = map(int,bbox)
    cx = (x1+x2) // 2
    width = x2-x1
    a = width
    b = max(1, int(0.35 * width))
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


def annotate_tracks(frame, tracks, names, show_ids=True):
    """Draw tracker output (a supervision Detections) with persistent track IDs.

    Predicted balls (Kalman estimate, not detected) are drawn as an outline so a
    guess never looks like an observation, with a circle two standard deviations
    wide showing how unsure the guess is.
    """
    interpolated = tracks.data.get("interpolated")
    sigma = tracks.data.get("ball_sigma")
    for i in range(len(tracks)):
        bbox = tracks.xyxy[i]
        name = names[int(tracks.class_id[i])]
        color = COLORS.get(name, (255, 255, 255))
        is_interp = bool(interpolated[i]) if interpolated is not None else False

        if name == "ball":
            if is_interp:
                draw_triangle(frame, bbox, color, filled=False)
                if sigma is not None and np.isfinite(sigma[i]):
                    x1, y1, x2, y2 = map(int, bbox)
                    radius = max(1, int(2 * sigma[i]))
                    cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), radius, color, 1)
            else:
                draw_triangle(frame, bbox, color)
        else:
            draw_ellipse(frame, bbox, color)
            if show_ids and tracks.tracker_id is not None:
                draw_label(frame, bbox, str(int(tracks.tracker_id[i])), color)
    return frame
