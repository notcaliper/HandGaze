"""HandGaze AI - Real-time hand gesture to text recognition system."""
import os
import pickle
import time
from collections import deque
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np

from offline_dictionary import OfflineDictionary

# --- UI CONSTANTS ---
COLOR_ACCENT = (255, 200, 0)   # Gold
COLOR_BG = (20, 20, 20)        # Dark background
COLOR_SUCCESS = (0, 255, 100)  # Green
COLOR_ERROR = (100, 100, 255)  # Blue-red
COLOR_NEON = (0, 255, 255)     # Cyan
FONT = cv2.FONT_HERSHEY_SIMPLEX

FINGER_JOINTS = [
    [0, 1, 2], [1, 2, 3], [2, 3, 4],
    [0, 5, 6], [5, 6, 7], [6, 7, 8],
    [0, 9, 10], [9, 10, 11], [10, 11, 12],
    [0, 13, 14], [13, 14, 15], [14, 15, 16],
    [0, 17, 18], [17, 18, 19], [18, 19, 20],
]


class CustomHandGestureRecognizer:  # pylint: disable=too-many-instance-attributes
    """Recognize hand gestures and convert them to text input.

    Organized into: Core Logic, Text Processing, and UI Drawing.
    """

    def __init__(self):
        # 1. Hardware
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera.")

        # 2. AI Model
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # 3. Gesture Data
        self.gesture_data = self.load_latest_gesture_data()
        if not self.gesture_data:
            raise FileNotFoundError("No gesture data found! Run gesture_trainer.py first.")
        self.missing_gestures = [
            g for g in ["SPACE", "BACKSPACE"] if g not in self.gesture_data
        ]
        self.precomputed = self.precompute_gesture_features()

        # 4. State & Performance
        self.fps_buffer: deque = deque(maxlen=10)
        self.prev_frame_time = 0.0
        self.process_every_n_frames = 2
        self.frame_count = 0
        self.last_gesture = "Unknown"
        self.current_gesture = "Unknown"
        self.gesture_confirmed = False
        self.last_gesture_time = time.time()
        self.gesture_delay = 1.5
        self.start_time = time.time()

        # 5. Text Input
        self.dictionary_helper = OfflineDictionary()
        self.current_word = ""
        self.sentence = ""
        self.word_suggestions: List[str] = []

    # --- CORE RECOGNITION LOGIC ---

    def load_latest_gesture_data(self) -> Optional[Dict]:
        """Load the most recent gesture data pickle file."""
        data_dir = 'gesture_data'
        if not os.path.exists(data_dir):
            return None
        primary = os.path.join(data_dir, 'gesture_data.pkl')
        if os.path.exists(primary):
            with open(primary, 'rb') as f:
                return pickle.load(f)
        files = [
            fn for fn in os.listdir(data_dir)
            if fn.startswith('gesture_data_') and fn.endswith('.pkl')
        ]
        if not files:
            return None
        latest = max(files, key=lambda x: os.path.getctime(os.path.join(data_dir, x)))
        with open(os.path.join(data_dir, latest), 'rb') as f:
            return pickle.load(f)

    def precompute_gesture_features(self) -> Dict:
        """Pre-compute angle and distance features for all gesture samples."""
        result = {}
        for name, samples in self.gesture_data.items():
            result[name] = []
            for sample in samples:
                sample_2d = [[p[0], p[1]] for p in sample]
                result[name].append({
                    'angles': self.calculate_angles(sample_2d),
                    'rel_distances': self.get_relative_distances(sample_2d),
                })
        return result

    def get_relative_distances(self, landmarks: List[List[float]]) -> np.ndarray:
        """Calculate pairwise distances between fingertip keypoints."""
        key_points = [4, 8, 12, 16, 20]
        n_dist = (len(key_points) * (len(key_points) - 1)) // 2
        distances = np.zeros(n_dist, dtype=np.float32)
        idx = 0
        for i, kp_i in enumerate(key_points):
            p1 = np.array(landmarks[kp_i], dtype=np.float32)
            for kp_j in key_points[i + 1:]:
                p2 = np.array(landmarks[kp_j], dtype=np.float32)
                distances[idx] = np.linalg.norm(p1 - p2)
                idx += 1
        return distances

    def calculate_angles(self, landmarks: List[List[float]]) -> List[float]:
        """Calculate joint angles for each finger segment."""
        angles = []
        for p1, p2, p3 in FINGER_JOINTS:
            v1 = np.array(landmarks[p1]) - np.array(landmarks[p2])
            v2 = np.array(landmarks[p3]) - np.array(landmarks[p2])
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 == 0 or n2 == 0:
                angles.append(0.0)
                continue
            dot = np.dot(v1 / n1, v2 / n2)
            angles.append(float(np.arccos(np.clip(dot, -1.0, 1.0))))
        return angles

    def recognize_gesture(self, landmarks) -> str:
        """Match landmarks against precomputed gesture features."""
        if not landmarks:
            return "Unknown"
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            return self.last_gesture

        curr_lms = [[lm.x, lm.y] for lm in landmarks.landmark]
        curr_angles = np.array(self.calculate_angles(curr_lms))
        curr_dists = self.get_relative_distances(curr_lms)

        best_score, best_gesture = float('inf'), "Unknown"
        for name, samples in self.precomputed.items():
            scores = []
            for s in samples[:3]:
                a_diff = np.mean(np.abs(curr_angles - s['angles']))
                d_diff = np.mean(np.abs(curr_dists - s['rel_distances']))
                scores.append(a_diff * 0.6 + d_diff * 0.4)
            avg = float(np.mean(scores))
            if avg < best_score:
                best_score, best_gesture = avg, name

        self.last_gesture = best_gesture if best_score <= 0.25 else "Unknown"
        return self.last_gesture

    # --- TEXT PROCESSING LOGIC ---

    def _handle_text_logic(self, gesture: str):
        """Apply text mutation based on the confirmed gesture."""
        if gesture == "SPACE":
            if self.current_word:
                if not self.dictionary_helper.is_valid_word(self.current_word):
                    sugg = self.dictionary_helper.get_suggestions(self.current_word)
                    if sugg:
                        self.current_word = sugg[0]
                self.sentence += self.current_word + " "
                self.current_word = ""
                self.word_suggestions = []
        elif gesture == "BACKSPACE":
            if self.current_word:
                self.current_word = self.current_word[:-1]
                if self.current_word:
                    self.word_suggestions = (
                        self.dictionary_helper.get_suggestions(self.current_word)[:3]
                    )
            elif self.sentence:
                self.sentence = self.sentence[:-1]
                if self.sentence and self.sentence[-1] == " ":
                    words = self.sentence.strip().split()
                    if words:
                        self.current_word = words[-1]
                        self.sentence = (
                            " ".join(words[:-1]) + " " if len(words) > 1 else ""
                        )
                        self.word_suggestions = (
                            self.dictionary_helper.get_suggestions(self.current_word)[:3]
                        )
        else:
            self.current_word += gesture
            if len(self.current_word) >= 2:
                self.word_suggestions = (
                    self.dictionary_helper.get_suggestions(self.current_word)[:3]
                )

    def add_gesture_to_text(self, gesture: str):
        """Confirm a held gesture and commit it to text."""
        if gesture == "Unknown":
            return
        now = time.time()
        if gesture != self.current_gesture:
            self.current_gesture = gesture
            self.gesture_confirmed = False
            self.last_gesture_time = now
        elif (not self.gesture_confirmed
              and (now - self.last_gesture_time) >= self.gesture_delay):
            self._handle_text_logic(gesture)
            self.gesture_confirmed = True

    # --- UI DRAWING HELPERS ---

    def _draw_rounded_rect(self, img, pt1, pt2, color, thickness, r):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Simulate a rounded rectangle with circles at corners."""
        x1, y1 = pt1
        x2, y2 = pt2
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
        for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r),
                       (x1 + r, y2 - r), (x2 - r, y2 - r)]:
            cv2.circle(img, (cx, cy), r, color, thickness)

    def _draw_hand_box(self, frame, landmarks):  # pylint: disable=too-many-locals
        """Draw stylized corner brackets around the detected hand."""
        h, w, _ = frame.shape
        xs = [int(lm.x * w) for lm in landmarks.landmark]
        ys = [int(lm.y * h) for lm in landmarks.landmark]
        pad = 20
        x_min = max(0, min(xs) - pad)
        x_max = min(w, max(xs) + pad)
        y_min = max(0, min(ys) - pad)
        y_max = min(h, max(ys) + pad)
        ln = 30
        for (cx, cy) in [(x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)]:
            sx = 1 if cx == x_min else -1
            sy = 1 if cy == y_min else -1
            cv2.line(frame, (cx, cy), (cx + sx * ln, cy), COLOR_NEON, 2)
            cv2.line(frame, (cx, cy), (cx, cy + sy * ln), COLOR_NEON, 2)

    # --- MAIN UI DRAWING ---

    def _draw_text_display(self, frame):  # pylint: disable=too-many-locals
        """Render the text panel with typed output and word suggestions."""
        display_text = self.sentence + self.current_word
        px, py, pw, ph = 10, 45, 400, 240

        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + pw, py + ph), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        self._draw_rounded_rect(frame, (px, py), (px + pw, py + ph), COLOR_ACCENT, 1, 10)

        y_off = py + 30
        for i in range(0, len(display_text), 35):
            if y_off >= py + 140:
                break
            txt = display_text[i:i + 35]
            cv2.putText(frame, txt, (px + 15, y_off), FONT, 0.6, (255, 255, 255), 2)
            y_off += 30

        if self.word_suggestions:  # pylint: disable=too-many-locals
            y_s = py + 130
            cv2.putText(frame, "SUGGESTIONS", (px + 15, y_s), FONT, 0.5, COLOR_ACCENT, 2)
            for idx, word in enumerate(self.word_suggestions, 1):
                y_s += 30
                sz = cv2.getTextSize(f"{idx}. {word}", FONT, 0.5, 2)[0]
                pulse = int(abs(np.sin(time.time() * 5)) * 50) if idx == 1 else 0
                bg = (0, 80 + pulse, 0) if idx == 1 else (40, 40, 40)
                self._draw_rounded_rect(
                    frame,
                    (px + 30, y_s - 18),
                    (px + 30 + sz[0] + 20, y_s + 8),
                    bg, -1, 5,
                )
                cv2.putText(frame, f"{idx}. {word}", (px + 35, y_s), FONT, 0.5, (255, 255, 255), 1)

    def _draw_status_info(self, frame, gesture, hand_detected):  # pylint: disable=too-many-locals
        """Render the top status bar and gesture confirmation card."""
        # Status bar
        cv2.rectangle(frame, (0, 0), (640, 35), (15, 15, 15), -1)
        cv2.line(frame, (0, 35), (640, 35), COLOR_ACCENT, 1)
        cv2.putText(frame, "[CAM ACTIVE]", (10, 23), FONT, 0.4, COLOR_SUCCESS, 1)
        hand_label = "[HAND DETECTED]" if hand_detected else "[SEARCHING...]"
        hand_col = COLOR_SUCCESS if hand_detected else COLOR_ERROR
        cv2.putText(frame, hand_label, (110, 23), FONT, 0.4, hand_col, 1)
        cv2.putText(frame, f"FPS: {self._get_avg_fps()}", (550, 23), FONT, 0.4, COLOR_SUCCESS, 1)

        # Gesture card
        yc = 300
        g_color = COLOR_SUCCESS if gesture != "Unknown" else COLOR_ERROR
        self._draw_rounded_rect(frame, (10, yc), (280, yc + 70), (30, 30, 30), -1, 10)
        self._draw_rounded_rect(frame, (10, yc), (280, yc + 70), COLOR_ACCENT, 1, 10)
        cv2.putText(frame, "CURRENT GESTURE", (25, yc + 20), FONT, 0.4, COLOR_ACCENT, 1)
        cv2.putText(frame, gesture, (25, yc + 55), FONT, 1.0, g_color, 2)

        # Circular confirmation ring
        if self.current_gesture != "Unknown" and not self.gesture_confirmed:
            held = time.time() - self.last_gesture_time
            prog = min(1.0, held / self.gesture_delay)
            center = (320, yc + 35)
            cv2.circle(frame, center, 25, (50, 50, 50), 3)
            cv2.ellipse(frame, center, (25, 25), -90, 0, prog * 360, COLOR_NEON, 3)
            pct = f"{int(prog * 100)}%"
            cv2.putText(frame, pct, (center[0] - 12, center[1] + 5), FONT, 0.35, COLOR_NEON, 1)

        if self.missing_gestures:
            label = "FALLBACK: " + "/".join(self.missing_gestures)
            cv2.putText(frame, label, (10, 465), FONT, 0.4, (150, 150, 255), 1)

    def _get_avg_fps(self) -> int:
        """Compute a rolling average FPS."""
        now = time.time()
        fps = 1.0 / (now - self.prev_frame_time) if self.prev_frame_time else 0.0
        self.prev_frame_time = now
        self.fps_buffer.append(fps)
        return int(sum(self.fps_buffer) / len(self.fps_buffer))

    # --- PIPELINE & EXECUTION ---

    def process_frame(self, frame):
        """Run calibration splash or the full recognition pipeline."""
        elapsed = time.time() - self.start_time
        if elapsed < 3.0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (640, 480), (10, 10, 10), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            cv2.putText(frame, "HANDGAZE AI v2.0", (180, 220), FONT, 1, COLOR_ACCENT, 2)
            cv2.putText(
                frame, "INITIALIZING NEURAL PIPELINE...",
                (190, 250), FONT, 0.5, (255, 255, 255), 1,
            )
            prog_w = int(300 * (elapsed / 3.0))
            cv2.rectangle(frame, (170, 270), (470, 275), (40, 40, 40), -1)
            cv2.rectangle(frame, (170, 270), (170 + prog_w, 275), COLOR_NEON, -1)
            return frame

        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = self.hands.process(rgb)
        self._draw_text_display(frame)

        gesture, hand_detected = "Unknown", False
        if res.multi_hand_landmarks:
            hand_detected = True
            for lms in res.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, lms, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=COLOR_NEON, thickness=1, circle_radius=1),
                    self.mp_draw.DrawingSpec(color=COLOR_ACCENT, thickness=1),
                )
                self._draw_hand_box(frame, lms)
                gesture = self.recognize_gesture(lms)
                self.add_gesture_to_text(gesture)

        self._draw_status_info(frame, gesture, hand_detected)
        return frame

    def run(self):
        """Start the real-time recognition loop."""
        print(f"\nHandGaze Active | Gestures: {list(self.gesture_data.keys())}")
        if self.missing_gestures:
            print(f"Keyboard fallback for: {', '.join(self.missing_gestures)}")
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame = self.process_frame(cv2.flip(frame, 1))
                cv2.imshow('HandGaze AI v2.0', frame)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:
                    break
                if key == 32 and "SPACE" in self.missing_gestures:
                    self._handle_text_logic("SPACE")
                if key == 8 and "BACKSPACE" in self.missing_gestures:
                    self._handle_text_logic("BACKSPACE")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.hands.close()


if __name__ == "__main__":
    try:
        CustomHandGestureRecognizer().run()
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"\nStartup Error: {exc}")
