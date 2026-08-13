"""Mediapipe compatibility module for legacy solutions API support."""
import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp


class DrawingSpec:
    def __init__(self, color=(0, 255, 0), thickness=2, circle_radius=2):
        self.color = color
        self.thickness = thickness
        self.circle_radius = circle_radius


class DummyDrawingUtils:
    DrawingSpec = DrawingSpec

    @staticmethod
    def draw_landmarks(image, landmark_list, connections=None, landmark_drawing_spec=None, connection_drawing_spec=None):
        if not landmark_list:
            return
        h, w, _ = image.shape
        pts = landmark_list.landmark if hasattr(landmark_list, 'landmark') else landmark_list
        if connections:
            for c in connections:
                s_idx = c[0] if isinstance(c, (tuple, list)) else getattr(c, 'start', c[0])
                e_idx = c[1] if isinstance(c, (tuple, list)) else getattr(c, 'end', c[1])
                if s_idx < len(pts) and e_idx < len(pts):
                    p1, p2 = pts[s_idx], pts[e_idx]
                    x1, y1 = int(p1.x * w), int(p1.y * h)
                    x2, y2 = int(p2.x * w), int(p2.y * h)
                    spec = connection_drawing_spec
                    if isinstance(spec, dict):
                        spec = spec.get(c, None)
                    color = getattr(spec, 'color', (255, 200, 0)) if spec else (255, 200, 0)
                    thickness = getattr(spec, 'thickness', 1) if spec else 1
                    cv2.line(image, (x1, y1), (x2, y2), color, thickness)
        for idx, lm in enumerate(pts):
            cx, cy = int(lm.x * w), int(lm.y * h)
            spec = landmark_drawing_spec
            if isinstance(spec, dict):
                spec = spec.get(idx, None)
            color = getattr(spec, 'color', (0, 255, 255)) if spec else (0, 255, 255)
            radius = getattr(spec, 'circle_radius', 2) if spec else 2
            thickness = getattr(spec, 'thickness', -1) if spec else -1
            cv2.circle(image, (cx, cy), radius, color, thickness)


class HandLandmarkWrapper:
    def __init__(self, landmark_list):
        self.landmark = landmark_list


class ProcessResult:
    def __init__(self, multi_hand_landmarks):
        self.multi_hand_landmarks = multi_hand_landmarks


class HandsWrapper:
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
    ]

    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
        if not os.path.exists(model_path):
            url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            try:
                urllib.request.urlretrieve(url, model_path)
            except Exception as e:
                raise RuntimeError(f"Failed to download MediaPipe hand landmarker model: {e}") from e

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process(self, image_rgb):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        res = self.detector.detect(mp_image)
        if res.hand_landmarks:
            wrapped = [HandLandmarkWrapper(h) for h in res.hand_landmarks]
            return ProcessResult(wrapped)
        return ProcessResult(None)

    def close(self):
        if hasattr(self.detector, 'close'):
            self.detector.close()


class HandsModule:
    Hands = HandsWrapper
    HAND_CONNECTIONS = HandsWrapper.HAND_CONNECTIONS

    def __call__(self, *args, **kwargs):
        return HandsWrapper(*args, **kwargs)


class DummySolutions:
    hands = HandsModule()
    drawing_utils = DummyDrawingUtils
    drawing_styles = type('DrawingStyles', (), {
        'get_default_hand_landmarks_style': lambda: {},
        'get_default_hand_connections_style': lambda: {}
    })


def ensure_mediapipe_solutions():
    """Ensure mp.solutions exists, falling back to MediaPipe Tasks if needed."""
    if not hasattr(mp, 'solutions') or not hasattr(mp.solutions, 'hands'):
        mp.solutions = DummySolutions


ensure_mediapipe_solutions()
