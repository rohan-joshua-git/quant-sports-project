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


def draw_triangle(frame, bbox, color):
    x1, y1, x2, _ = map(int, bbox)
    cx = (x1 + x2) // 2
    size = 14
    tip_y = y1 - 2
    points = np.array([
        [cx, tip_y],
        [cx - size, tip_y - 2 * size],
        [cx + size, tip_y - 2 * size],
    ], dtype=np.int32)
    cv2.drawContours(frame, [points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [points], 0, (0, 0, 0), 2)
    return frame


def annotate_frame(frame, result):
    for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
        name = result.names[int(cls)]
        color = COLORS.get(name, (255, 255, 255))
        bbox = box.tolist()
        if name == "ball":
            draw_triangle(frame, bbox, color)
        else:
            draw_ellipse(frame, bbox, color)
    return frame
