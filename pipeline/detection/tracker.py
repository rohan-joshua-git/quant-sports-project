import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np


class Tracker:
    def __init__(self, model_path, conf=0.1):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.conf = conf
        self.last_ball_bbox = None

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
        detection_raw = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        detection_sv = sv.Detections.from_ultralytics(detection_raw)
        detection_sv = detection_sv.with_nms(threshold=0.3)
        tracks = self.tracker.update_with_detections(detection_sv)

        ball_detected = False
        if len(tracks) > 0:
            for i, cls_id in enumerate(tracks.class_id):
                if detection_raw.names[int(cls_id)] == 'ball':
                    bbox = tracks.xyxy[i]
                    self.last_ball_bbox = bbox
                    ball_detected = True
                    break

        if not ball_detected and self.last_ball_bbox is not None:
            tracks = self._add_interpolated_ball(tracks)

        return tracks, detection_raw

    def _add_interpolated_ball(self, tracks):
        if self.last_ball_bbox is None:
            return tracks

        ball_xyxy = np.array([self.last_ball_bbox], dtype=np.float32)
        ball_confidence = np.array([0.5], dtype=np.float32)
        ball_class = np.array([2])
        ball_tracker_id = np.array([1])

        tracks.xyxy = np.vstack([tracks.xyxy, ball_xyxy])
        tracks.confidence = np.concatenate([tracks.confidence, ball_confidence])
        tracks.class_id = np.concatenate([tracks.class_id, ball_class])
        tracks.tracker_id = np.concatenate([tracks.tracker_id, ball_tracker_id])

        return tracks
