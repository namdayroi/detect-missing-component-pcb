# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import json
import math
import time
import threading
from queue import Queue, Empty
import torch
import os

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QPushButton, QLabel, QTextEdit, QStatusBar, QMenuBar, QAction, QFileDialog, QLineEdit)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer
from ultralytics import YOLO

# --- Configuration Class ---
class Config:
    # Paths (adjust defaults as needed)
    PATH_TO_YOLO_MODEL = r"C:\Users\Namdr\Downloads\best (1).pt"
    PATH_TO_REFERENCE_MAP = r"C:\Users\Namdr\Downloads\reference_map_generated.json"
    PATH_TO_CLASSES_TXT = r"C:\Users\Namdr\Downloads\dataset\classes.txt"
    PATH_TO_FIXED_ALIGNMENT_MATRIX = r"C:\Users\Namdr\Downloads\fixed_alignment_matrix.json"

    # Inference Settings
    INFERENCE_SIZE = (720, 720)  # Input size for YOLO model
    LOWERED_YOLO_CONFIDENCE_THRESHOLD = 0.3  # Initial detection confidence (for rescue)
    STRICT_YOLO_CONFIDENCE_THRESHOLD = 0.6  # High confidence detection threshold
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    USE_HALF = (DEVICE == 'cuda')

    # Inspection Controls
    FRAME_SKIP_INTERVAL = 3  # Process 1 out of N frames to reduce CPU load
    ALIGNMENT_RECALC_INTERVAL = 20  # Recalculate dynamic alignment periodically
    MATCHING_DISTANCE_THRESHOLD = 25  # Pixel distance matching threshold after alignment
    USE_FIXED_ALIGNMENT = True  # Toggle fixed camera vs dynamic alignment

    # Geometric Validation Thresholds
    AREA_SIMILARITY_THRESHOLD = 0.8
    ASPECT_RATIO_SIMILARITY_THRESHOLD = 0.7

    # Contextual Filtering Parameters
    CONTEXT_UNEXPECTED_NEARBY_THRESHOLD = 60  # pixel radius for expected near check
    CONTEXT_RESCUE_CONFIDENCE_THRESHOLD = 0.35

    # Visual Styles (BGR format for OpenCV drawing)
    COLOR_MISSING = (0, 0, 255)         # Red
    COLOR_MATCHED = (0, 255, 0)         # Green
    COLOR_MATCHED_LOW_CONF = (150, 255, 150) # Light Green
    COLOR_UNEXPECTED = (0, 165, 255)     # Orange
    COLOR_UNEXPECTED_LOW_CONF = (0, 200, 255) # Light Orange
    COLOR_INFO_TEXT = (255, 255, 255)
    COLOR_ALIGN_OK = (0, 255, 0)
    COLOR_ALIGN_FAIL = (0, 0, 255)
    COLOR_FPS = (0, 255, 0)
    STATUS_PANEL_HEIGHT = 80

    # Live Camera
    IP_WEBCAM_URL = "rtsp://192.168.0.101:8080/h264.sdp"
    WEBCAM_READ_TIMEOUT = 2

# --- Webcam Streaming Class ---
class WebcamStream:
    def __init__(self, src=0, name="WebcamStream", read_timeout=1):
        self.src = src
        self.read_timeout = read_timeout
        print(f"[INFO] Initializing stream: {self.src}...")
        self.stream = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        if not self.stream.isOpened():
            print(f"[WARNING] OpenCV FFMPEG backend failed to open: {self.src}")
            if isinstance(src, int):
                print("[INFO] Retrying with default backend for local camera...")
                self.stream = cv2.VideoCapture(self.src)
                if not self.stream.isOpened():
                    raise ValueError(f"Failed to open video source: {self.src}")
            else:
                raise ValueError(f"Failed to open camera stream: {self.src}")

        (self.grabbed, self.frame) = self.stream.read()
        if not self.grabbed or self.frame is None:
            print("[ERROR] Cannot read initial frame from camera source.")
            self.stream.release()
            raise ValueError("Failed reading initial frame")

        self.name = name
        self.Q = Queue(maxsize=1)
        self.Q.put(self.frame)
        self.stopped = False
        self.thread = threading.Thread(target=self.update, name=self.name, daemon=True)
        print(f"[INFO] Camera stream {self.src} initialized successfully.")

    def start(self):
        self.stopped = False
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            if not self.stream.isOpened():
                self.stopped = True
                break
            (grabbed, frame) = self.stream.read()
            if not grabbed:
                time.sleep(0.01)
                continue
            if frame is None:
                continue
            if self.Q.full():
                try:
                    self.Q.get_nowait()
                except Empty:
                    pass
            self.Q.put(frame)
        if self.stream.isOpened():
            self.stream.release()
            print(f"[INFO] Released camera stream: {self.src}")

    def read(self):
        try:
            return self.Q.get(timeout=self.read_timeout)
        except Empty:
            return None

    def stop(self):
        if not self.stopped:
            self.stopped = True
            if self.thread.is_alive():
                self.thread.join(timeout=2.0)
            if self.stream.isOpened():
                self.stream.release()

# --- Helper Logic Functions ---
def load_yolo_labels(model=None, classes_path=""):
    labels = {}
    if model and hasattr(model, 'names') and model.names:
        labels = model.names
        if isinstance(labels, list):
            labels = {i: name for i, name in enumerate(labels)}
    if not labels and classes_path and os.path.exists(classes_path):
        try:
            with open(classes_path, 'r', encoding='utf-8') as f:
                labels_list = [line.strip() for line in f if line.strip()]
                labels = {i: name for i, name in enumerate(labels_list)}
        except Exception as e:
            print(f"[ERROR] Failed loading classes.txt: {e}")
    return labels

def load_reference_map(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            ref_data = json.load(f)
        ref_dict = {}
        expected_list = []
        for i, item in enumerate(ref_data):
            label = item.get("label")
            bbox = item.get("bbox_ref")  # [x, y, w, h]
            if label and bbox and len(bbox) == 4:
                center = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
                comp_info = {
                    "id": i,
                    "label": label,
                    "bbox_ref": bbox,
                    "center_ref": center
                }
                expected_list.append(comp_info)
                if label not in ref_dict:
                    ref_dict[label] = comp_info
        print(f"[INFO] Loaded reference map with {len(expected_list)} expected components.")
        return ref_dict, expected_list
    except Exception as e:
        print(f"[ERROR] Failed to load reference JSON: {e}")
        return None, None

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def find_anchor_points(detected_list, ref_dict):
    ref_coords = []
    det_coords = []
    used_ids = set()
    for det in detected_list:
        label = det['label']
        if label in ref_dict and label not in used_ids:
            ref_coords.append(ref_dict[label]['center_ref'])
            det_coords.append(det['center_det'])
            used_ids.add(label)
    return ref_coords, det_coords

def calculate_affine_transform(ref_pts, det_pts):
    if len(ref_pts) < 3 or len(det_pts) < 3:
        return None, None
    np_ref = np.float32(ref_pts).reshape(-1, 1, 2)
    np_det = np.float32(det_pts).reshape(-1, 1, 2)
    try:
        M_det_to_ref, inliers = cv2.estimateAffine2D(np_det, np_ref, cv2.RANSAC, 5.0)
        if M_det_to_ref is None:
            return None, None
        M3x3 = np.vstack([M_det_to_ref, [0, 0, 1]])
        if abs(np.linalg.det(M3x3)) > 1e-6:
            M_inv = np.linalg.inv(M3x3)
            M_ref_to_det = M_inv[:2, :]
        else:
            M_ref_to_det = None
        return M_det_to_ref, M_ref_to_det
    except Exception:
        return None, None

def transform_points_affine(points, M):
    if M is None or not points:
        return []
    np_pts = np.float32(points).reshape(-1, 1, 2)
    try:
        t_pts = cv2.transform(np_pts, M)
        return [tuple(map(int, p[0])) for p in t_pts]
    except Exception:
        return []

def save_affine_matrix(M, filepath):
    try:
        with open(filepath, 'w') as f:
            json.dump(M.tolist(), f)
    except Exception as e:
        print(f"[ERROR] Failed to save fixed alignment matrix: {e}")

def load_fixed_alignment_matrix(filepath):
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return np.array(data, dtype=np.float32)
    except Exception as e:
        print(f"[ERROR] Failed loading fixed alignment matrix: {e}")
        return None


# --- PCB Component Checker Engine Class ---
class PCBComponentChecker:
    def __init__(self, config, processing_thread=None):
        print("[INFO] Checker: Initializing PCB Component Checker Engine...")
        self.config = config
        self.processing_thread = processing_thread
        self.model = None
        self.model_labels = {}
        self.reference_map_dict = None
        self.expected_components_list = None
        self.webcam_stream = None

        # States
        self.alignment_status = False
        self.current_M_det_to_ref = None
        self.current_M_ref_to_det = None
        self.is_paused = False

        self.last_fps = 0.0
        self.last_processed_components = []
        self.last_matched_indices = set()
        self.last_missing_components_info = []
        self.last_unmatched_indices = set()

        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.total_frame_count = 0
        self.processed_frame_count = 0

        self._load_resources()

    def _load_resources(self):
        print("[INFO] Checker: Loading YOLO model...")
        try:
            self.model = YOLO(self.config.PATH_TO_YOLO_MODEL)
            self.model.to(self.config.DEVICE)
            self.model_labels = load_yolo_labels(self.model, self.config.PATH_TO_CLASSES_TXT)
            print(f"[INFO] Checker: YOLO model loaded on {self.config.DEVICE}.")
        except Exception as e:
            print(f"[FATAL] Checker: Failed loading YOLO model: {e}")
            if self.processing_thread:
                self.processing_thread.stop_processing()
            raise SystemExit("YOLO model load error")

        print("[INFO] Checker: Loading Reference map...")
        self.reference_map_dict, self.expected_components_list = load_reference_map(self.config.PATH_TO_REFERENCE_MAP)
        if not self.expected_components_list:
            if self.processing_thread:
                self.processing_thread.stop_processing()
            raise SystemExit("Reference map load error")

        # Load fixed matrix if enabled
        if self.config.USE_FIXED_ALIGNMENT:
            loaded_M = load_fixed_alignment_matrix(self.config.PATH_TO_FIXED_ALIGNMENT_MATRIX)
            if loaded_M is not None:
                self.current_M_ref_to_det = loaded_M
                M3x3 = np.vstack([self.current_M_ref_to_det, [0, 0, 1]])
                if abs(np.linalg.det(M3x3)) > 1e-6:
                    self.current_M_det_to_ref = cv2.invertAffineTransform(self.current_M_ref_to_det)
                    self.alignment_status = True
                    print("[INFO] Checker: Loaded fixed alignment matrix successfully.")
                else:
                    print("[WARNING] Checker: Loaded fixed matrix is singular.")

    def _recalculate_and_save_fixed_alignment(self, det_comps):
        print("[INFO] Checker: Calculating fixed alignment matrix...")
        ref_coords, det_coords = find_anchor_points(det_comps, self.reference_map_dict)
        if len(ref_coords) >= 3:
            M_det_to_ref, M_ref_to_det = calculate_affine_transform(ref_coords, det_coords)
            if M_ref_to_det is not None and M_det_to_ref is not None:
                self.current_M_det_to_ref = M_det_to_ref
                self.current_M_ref_to_det = M_ref_to_det
                self.alignment_status = True
                save_affine_matrix(self.current_M_ref_to_det, self.config.PATH_TO_FIXED_ALIGNMENT_MATRIX)
                print("[INFO] Checker: Saved fixed matrix successfully.")
            else:
                self.alignment_status = False
        else:
            print(f"[WARNING] Checker: Not enough anchors ({len(ref_coords)}/3).")
            self.alignment_status = False

    def _process_frame(self, frame):
        self.processed_frame_count += 1
        h_frame, w_frame = frame.shape[:2]

        try:
            resized = cv2.resize(frame, self.config.INFERENCE_SIZE, interpolation=cv2.INTER_LINEAR)
        except Exception:
            return

        current_dets = []
        try:
            results = self.model.predict(resized, verbose=False,
                                         conf=self.config.LOWERED_YOLO_CONFIDENCE_THRESHOLD,
                                         device=self.config.DEVICE, half=self.config.USE_HALF,
                                         imgsz=self.config.INFERENCE_SIZE)
            if results and results[0].boxes is not None:
                scale_x = w_frame / self.config.INFERENCE_SIZE[0]
                scale_y = h_frame / self.config.INFERENCE_SIZE[1]
                for box in results[0].boxes.data.cpu().numpy():
                    if len(box) >= 6:
                        x1, y1, x2, y2, conf, label_id = box[:6]
                        rx1, ry1 = int(x1 * scale_x), int(y1 * scale_y)
                        rx2, ry2 = int(x2 * scale_x), int(y2 * scale_y)
                        dw, dh = rx2 - rx1, ry2 - ry1
                        if dw <= 0 or dh <= 0: continue
                        current_dets.append({
                            'label': self.model_labels.get(int(label_id), f"ID_{int(label_id)}"),
                            'bbox_det': [rx1, ry1, rx2, ry2],
                            'center_det': (rx1 + dw // 2, ry1 + dh // 2),
                            'confidence': conf,
                            'matched_to_ref_id': None,
                            'is_high_confidence_detection': conf >= self.config.STRICT_YOLO_CONFIDENCE_THRESHOLD,
                            'area_det': dw * dh, 'width_det': dw, 'height_det': dh,
                            'is_contextually_verified': False
                        })
        except Exception as e:
            print(f"[ERROR] Checker: YOLO Prediction error: {e}")

        # Alignment calculations
        if self.config.USE_FIXED_ALIGNMENT and not self.alignment_status:
            if current_dets:
                high_conf = [d for d in current_dets if d['is_high_confidence_detection']]
                self._recalculate_and_save_fixed_alignment(high_conf)
        elif not self.config.USE_FIXED_ALIGNMENT:
            if (self.processed_frame_count == 1 or not self.alignment_status or self.processed_frame_count % self.config.ALIGNMENT_RECALC_INTERVAL == 0):
                if current_dets:
                    high_conf = [d for d in current_dets if d['is_high_confidence_detection']]
                    ref_pts, det_pts = find_anchor_points(high_conf, self.reference_map_dict)
                    if len(ref_pts) >= 3:
                        M_det_to_ref, M_ref_to_det = calculate_affine_transform(ref_pts, det_pts)
                        if M_ref_to_det is not None and M_det_to_ref is not None:
                            self.current_M_det_to_ref = M_det_to_ref
                            self.current_M_ref_to_det = M_ref_to_det
                            self.alignment_status = True
                        else:
                            self.alignment_status = False
                    else:
                        self.alignment_status = False

        self.last_processed_components = current_dets

        # Components checking & matching
        current_matched_indices = set()
        current_missing_info = []
        current_unmatched_indices = set()

        if self.alignment_status and self.current_M_ref_to_det is not None:
            expected_centers_transformed = []
            for exp in self.expected_components_list:
                t_center = transform_points_affine([exp['center_ref']], self.current_M_ref_to_det)
                if t_center: expected_centers_transformed.append(t_center[0])

            matched_det_indices = set()
            found_ref_ids = set()

            for exp in self.expected_components_list:
                exp_id, exp_label = exp['id'], exp['label']
                exp_center, exp_bbox = exp['center_ref'], exp['bbox_ref']
                exp_w, exp_h = exp_bbox[2], exp_bbox[3]

                t_center_list = transform_points_affine([exp_center], self.current_M_ref_to_det)
                if not t_center_list: continue
                t_center = t_center_list[0]

                ref_corners = [(exp_bbox[0], exp_bbox[1]), (exp_bbox[0] + exp_w, exp_bbox[1]),
                               (exp_bbox[0] + exp_w, exp_bbox[1] + exp_h), (exp_bbox[0], exp_bbox[1] + exp_h)]
                t_corners = transform_points_affine(ref_corners, self.current_M_ref_to_det)

                t_area, t_w, t_h = 0, 0, 0
                if len(t_corners) == 4:
                    xs = [p[0] for p in t_corners]; ys = [p[1] for p in t_corners]
                    t_w = abs(max(xs) - min(xs)); t_h = abs(max(ys) - min(ys))
                    t_area = t_w * t_h

                best_score = -float('inf')
                best_idx = -1

                for idx, det in enumerate(current_dets):
                    if idx in matched_det_indices: continue
                    if det['label'] != exp_label: continue

                    dist = calculate_distance(t_center, det['center_det'])
                    if dist < self.config.MATCHING_DISTANCE_THRESHOLD:
                        det_area, det_w, det_h = det['area_det'], det['width_det'], det['height_det']
                        
                        area_ok = False
                        if t_area > 0 and det_area > 0:
                            ratio = det_area / t_area
                            area_ok = self.config.AREA_SIMILARITY_THRESHOLD < ratio < (1.0 / self.config.AREA_SIMILARITY_THRESHOLD)

                        aspect_ok = False
                        if t_h > 0 and det_h > 0 and t_w > 0 and det_w > 0:
                            ar_ref = t_w / t_h
                            ar_det = det_w / det_h
                            ratio_ar = ar_det / ar_ref
                            aspect_ok = self.config.ASPECT_RATIO_SIMILARITY_THRESHOLD < ratio_ar < (1.0 / self.config.ASPECT_RATIO_SIMILARITY_THRESHOLD)

                        geom_ok = area_ok and aspect_ok
                        confident = det['confidence'] >= self.config.STRICT_YOLO_CONFIDENCE_THRESHOLD
                        rescuable = self.config.CONTEXT_RESCUE_CONFIDENCE_THRESHOLD <= det['confidence'] < self.config.STRICT_YOLO_CONFIDENCE_THRESHOLD

                        if geom_ok and (confident or rescuable):
                            score = det['confidence'] - (dist / (self.config.MATCHING_DISTANCE_THRESHOLD * 10))
                            if score > best_score:
                                best_score = score
                                best_idx = idx

                if best_idx != -1:
                    current_matched_indices.add(best_idx)
                    matched_det_indices.add(best_idx)
                    current_dets[best_idx]['matched_to_ref_id'] = exp_id
                    current_dets[best_idx]['is_contextually_verified'] = True
                    found_ref_ids.add(exp_id)

            for exp in self.expected_components_list:
                if exp['id'] not in found_ref_ids:
                    current_missing_info.append(exp)

            # Context filtering
            for idx, det in enumerate(current_dets):
                if idx not in current_matched_indices:
                    is_near = False
                    if expected_centers_transformed:
                        for exp_c in expected_centers_transformed:
                            if calculate_distance(det['center_det'], exp_c) < self.config.CONTEXT_UNEXPECTED_NEARBY_THRESHOLD:
                                is_near = True
                                break
                    if det['is_high_confidence_detection'] or (det['confidence'] >= self.config.LOWERED_YOLO_CONFIDENCE_THRESHOLD and is_near):
                        current_unmatched_indices.add(idx)
                        current_dets[idx]['is_contextually_verified'] = True
                    else:
                        current_dets[idx]['is_contextually_verified'] = False

        else:
            current_unmatched_indices = set(range(len(current_dets)))
            for i in range(len(current_dets)):
                current_dets[i]['is_contextually_verified'] = True
            current_missing_info = list(self.expected_components_list)

        self.last_matched_indices = current_matched_indices
        self.last_missing_components_info = current_missing_info
        self.last_unmatched_indices = current_unmatched_indices

    def _update_visualization(self, image):
        # 1. Draw Matched Boxes
        for idx in self.last_matched_indices:
            if idx < len(self.last_processed_components):
                comp = self.last_processed_components[idx]
                bbox = comp['bbox_det']
                color = self.config.COLOR_MATCHED if comp['confidence'] >= self.config.STRICT_YOLO_CONFIDENCE_THRESHOLD else self.config.COLOR_MATCHED_LOW_CONF
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(image, f"{comp['label']} ({comp['confidence']:.2f})", (bbox[0], bbox[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # 2. Draw Unexpected / Extra Boxes
        for idx in self.last_unmatched_indices:
            if idx < len(self.last_processed_components):
                comp = self.last_processed_components[idx]
                if comp.get('is_contextually_verified', False):
                    bbox = comp['bbox_det']
                    color = self.config.COLOR_UNEXPECTED if comp['confidence'] >= self.config.STRICT_YOLO_CONFIDENCE_THRESHOLD else self.config.COLOR_UNEXPECTED_LOW_CONF
                    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    cv2.putText(image, f"Extra: {comp['label']} ({comp['confidence']:.2f})", (bbox[0], bbox[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # 3. Draw Missing Boxes
        if self.alignment_status and self.current_M_ref_to_det is not None:
            for missing in self.last_missing_components_info:
                bbox_ref = missing['bbox_ref']
                ref_corners = [(bbox_ref[0], bbox_ref[1]), (bbox_ref[0] + bbox_ref[2], bbox_ref[1]),
                               (bbox_ref[0] + bbox_ref[2], bbox_ref[1] + bbox_ref[3]), (bbox_ref[0], bbox_ref[1] + bbox_ref[3])]
                t_corners = transform_points_affine(ref_corners, self.current_M_ref_to_det)
                t_center_list = transform_points_affine([missing['center_ref']], self.current_M_ref_to_det)
                
                if len(t_corners) == 4:
                    pts = np.array(t_corners, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(image, [pts], isClosed=True, color=self.config.COLOR_MISSING, thickness=2)
                if t_center_list:
                    tc = t_center_list[0]
                    cv2.drawMarker(image, tc, self.config.COLOR_MISSING, cv2.MARKER_CROSS, 15, 2)
                    cv2.putText(image, f"Missing: {missing['label']}", (tc[0] + 8, tc[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.config.COLOR_MISSING, 1, cv2.LINE_AA)

        # 4. FPS calculations
        self.fps_frame_count += 1
        curr_time = time.time()
        time_diff = curr_time - self.fps_start_time
        if time_diff >= 1.0:
            self.last_fps = self.fps_frame_count / time_diff
            self.fps_start_time = curr_time
            self.fps_frame_count = 0

    def _cleanup_resources_only(self):
        print("[INFO] Checker: Releasing webcam resources...")
        if self.webcam_stream:
            self.webcam_stream.stop()


# --- PyQt5 Worker Processing Thread Class ---
class ProcessingThread(QThread):
    frame_processed = pyqtSignal(np.ndarray)
    stats_updated = pyqtSignal(dict)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.checker = None
        self._is_running = True
        self._is_paused = False

    def run(self):
        print("[INFO] ProcessingThread: Starting background loop...")
        try:
            self.checker = PCBComponentChecker(self.config, self)
            self.checker.webcam_stream = WebcamStream(src=self.config.IP_WEBCAM_URL,
                                                      read_timeout=self.config.WEBCAM_READ_TIMEOUT).start()
            
            # Stabilization wait
            time.sleep(1.5)
            self._is_running = True

            while self._is_running:
                frame = self.checker.webcam_stream.read()
                if frame is None:
                    if self.checker.webcam_stream.stopped:
                        print("[WARNING] Webcam stream stopped in worker thread.")
                        break
                    time.sleep(0.01)
                    continue

                self.checker.total_frame_count += 1
                display_image = frame.copy()

                if not self._is_paused:
                    if self.checker.total_frame_count % self.config.FRAME_SKIP_INTERVAL == 0:
                        self.checker._process_frame(frame)

                    self.checker._update_visualization(display_image)
                    self.frame_processed.emit(display_image)

                    # AOI Pass Rate logic
                    # Pass score is calculated based on matching all expected parts and having no missing/unexpected errors.
                    pass_rate = 100.0
                    num_expected = len(self.checker.expected_components_list) if self.checker.expected_components_list else 0
                    num_matched = len(self.checker.last_matched_indices)
                    missing_cnt = len(self.checker.last_missing_components_info)
                    unexpected_cnt = len(self.checker.last_unmatched_indices)

                    if num_expected > 0:
                        pass_rate = max(0.0, (num_matched - missing_cnt - unexpected_cnt) / num_expected) * 100.0

                    # Stats update packet
                    stats = {
                        "fps": self.checker.last_fps,
                        "alignment_status": self.checker.alignment_status,
                        "detected": len(self.checker.last_processed_components),
                        "matched": num_matched,
                        "missing": missing_cnt,
                        "unexpected": unexpected_cnt,
                        "pass_rate": pass_rate,
                        "produced_qty": self.checker.total_frame_count // self.config.FRAME_SKIP_INTERVAL,
                        "good_qty": num_matched,
                        "ng_qty": missing_cnt + unexpected_cnt
                    }
                    self.stats_updated.emit(stats)
                else:
                    time.sleep(0.1)

        except Exception as e:
            print(f"[FATAL] Error in ProcessingThread loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.checker:
                self.checker._cleanup_resources_only()
            print("[INFO] ProcessingThread: Thread finished.")

    def stop_processing(self):
        self._is_running = False

    def toggle_pause(self):
        self._is_paused = not self._is_paused
        return self._is_paused

    def trigger_realign(self):
        if self.checker and self.checker.webcam_stream:
            recalc_frame = self.checker.webcam_stream.read()
            if recalc_frame is not None:
                self.checker._trigger_fixed_realignment(recalc_frame)


# --- PyQt5 MainWindow ---
class MainWindow(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setWindowTitle("PCB Component AOI Inspector - Production App")
        self.setGeometry(100, 100, 1280, 780)

        # Setup main container
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Left Column Layout (Video Panel + Stats Grid Bar)
        self.left_layout = QVBoxLayout()
        self.main_layout.addLayout(self.left_layout, 3)  # Takes 3/4 space

        self.video_display_label = QLabel("Camera stream inactive. Click 'Start Inspection' to begin.")
        self.video_display_label.setAlignment(Qt.AlignCenter)
        self.video_display_label.setStyleSheet("border: 2px solid #222; background-color: #111; color: #fff; font-size: 14px;")
        self.left_layout.addWidget(self.video_display_label, 1)

        # Statistics panel grid layout
        self.info_panel_layout = QGridLayout()
        self.left_layout.addLayout(self.info_panel_layout)

        # Labels
        self.lbl_status_run = QLabel("Status: IDLE")
        self.lbl_fps = QLabel("FPS: --")
        self.lbl_alignment = QLabel("Alignment: N/A")
        self.lbl_detected = QLabel("Detected: --")
        self.lbl_matched = QLabel("Matched: --")
        self.lbl_missing = QLabel("Missing: --")
        self.lbl_unexpected = QLabel("Unexpected: --")
        self.lbl_pass_rate = QLabel("Pass Rate: 0.0%")
        self.lbl_produced_qty = QLabel("Produced Qty: 0")
        self.lbl_good_qty = QLabel("Good Qty: 0")
        self.lbl_ng_qty = QLabel("NG Qty: 0")

        # Fonts & Styling
        stat_font = QFont("Arial", 10, QFont.Bold)
        for lbl in [self.lbl_status_run, self.lbl_fps, self.lbl_alignment, self.lbl_detected, self.lbl_matched,
                    self.lbl_missing, self.lbl_unexpected, self.lbl_pass_rate, self.lbl_produced_qty, self.lbl_good_qty, self.lbl_ng_qty]:
            lbl.setFont(stat_font)
            lbl.setStyleSheet("padding: 3px; border: 1px solid #ddd; background-color: #fcfcfc;")

        # Place stats inside grid
        self.info_panel_layout.addWidget(self.lbl_status_run, 0, 0)
        self.info_panel_layout.addWidget(self.lbl_fps, 0, 1)
        self.info_panel_layout.addWidget(self.lbl_alignment, 0, 2)
        self.info_panel_layout.addWidget(self.lbl_pass_rate, 0, 3)

        self.info_panel_layout.addWidget(self.lbl_detected, 1, 0)
        self.info_panel_layout.addWidget(self.lbl_matched, 1, 1)
        self.info_panel_layout.addWidget(self.lbl_missing, 1, 2)
        self.info_panel_layout.addWidget(self.lbl_unexpected, 1, 3)

        self.info_panel_layout.addWidget(self.lbl_produced_qty, 2, 0)
        self.info_panel_layout.addWidget(self.lbl_good_qty, 2, 1)
        self.info_panel_layout.addWidget(self.lbl_ng_qty, 2, 2)

        # Right Column Layout (Control inputs panel)
        self.control_panel_layout = QVBoxLayout()
        self.main_layout.addLayout(self.control_panel_layout, 1)  # Takes 1/4 space

        # Config Inputs Sidebar
        self.control_panel_layout.addWidget(QLabel("<b>Configuration Parameters</b>"))

        self.txt_model_path = QLineEdit(self.config.PATH_TO_YOLO_MODEL)
        self.control_panel_layout.addWidget(QLabel("YOLO Model (.pt):"))
        self.control_panel_layout.addWidget(self.txt_model_path)
        btn_browse_model = QPushButton("Browse Model")
        btn_browse_model.clicked.connect(self.browse_model)
        self.control_panel_layout.addWidget(btn_browse_model)

        self.txt_ref_path = QLineEdit(self.config.PATH_TO_REFERENCE_MAP)
        self.control_panel_layout.addWidget(QLabel("Reference Map (JSON):"))
        self.control_panel_layout.addWidget(self.txt_ref_path)
        btn_browse_ref = QPushButton("Browse Reference")
        btn_browse_ref.clicked.connect(self.browse_reference)
        self.control_panel_layout.addWidget(btn_browse_ref)

        self.txt_classes_path = QLineEdit(self.config.PATH_TO_CLASSES_TXT)
        self.control_panel_layout.addWidget(QLabel("classes.txt File:"))
        self.control_panel_layout.addWidget(self.txt_classes_path)
        btn_browse_classes = QPushButton("Browse classes.txt")
        btn_browse_classes.clicked.connect(self.browse_classes)
        self.control_panel_layout.addWidget(btn_browse_classes)

        self.txt_camera_url = QLineEdit(self.config.IP_WEBCAM_URL)
        self.control_panel_layout.addWidget(QLabel("RTSP IP Camera Stream URL:"))
        self.control_panel_layout.addWidget(self.txt_camera_url)

        self.control_panel_layout.addSpacing(15)
        self.control_panel_layout.addWidget(QLabel("<b>Controls</b>"))

        self.btn_start = QPushButton("Start Inspection")
        self.btn_start.clicked.connect(self.start_inspection)
        self.btn_start.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; height: 35px;")
        self.control_panel_layout.addWidget(self.btn_start)

        self.btn_pause = QPushButton("Pause")
        self.btn_pause.clicked.connect(self.toggle_pause_inspection)
        self.btn_pause.setEnabled(False)
        self.control_panel_layout.addWidget(self.btn_pause)

        self.btn_realign = QPushButton("Recalculate Alignment (r)")
        self.btn_realign.clicked.connect(self.trigger_realign)
        self.btn_realign.setEnabled(False)
        self.control_panel_layout.addWidget(self.btn_realign)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_inspection)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; height: 30px;")
        self.control_panel_layout.addWidget(self.btn_stop)

        self.control_panel_layout.addStretch()

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        self.processing_thread = ProcessingThread(self.config)
        self.processing_thread.frame_processed.connect(self.update_video_display)
        self.processing_thread.stats_updated.connect(self.update_stats_display)
        self.processing_thread.finished.connect(self.on_processing_finished)

    def browse_model(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model", "", "Model (*.pt)")
        if f: self.txt_model_path.setText(f)

    def browse_reference(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Reference Map", "", "JSON (*.json)")
        if f: self.txt_ref_path.setText(f)

    def browse_classes(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select classes.txt", "", "Text File (*.txt)")
        if f: self.txt_classes_path.setText(f)

    def start_inspection(self):
        # Update config with GUI values
        self.config.PATH_TO_YOLO_MODEL = self.txt_model_path.text().strip()
        self.config.PATH_TO_REFERENCE_MAP = self.txt_ref_path.text().strip()
        self.config.PATH_TO_CLASSES_TXT = self.txt_classes_path.text().strip()
        
        # Parse stream URL (handle int for local webcam indices)
        cam_src = self.txt_camera_url.text().strip()
        if cam_src.isdigit():
            self.config.IP_WEBCAM_URL = int(cam_src)
        else:
            self.config.IP_WEBCAM_URL = cam_src

        if not os.path.exists(self.config.PATH_TO_YOLO_MODEL):
            self.status_bar.showMessage("Error: YOLO model file path not found!")
            return
        if not os.path.exists(self.config.PATH_TO_REFERENCE_MAP):
            self.status_bar.showMessage("Error: Reference map file not found!")
            return

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_pause.setEnabled(True)
        self.btn_realign.setEnabled(True)
        self.btn_pause.setText("Pause")
        self.lbl_status_run.setText("Status: CONNECTING...")
        self.status_bar.showMessage("Connecting to stream thread...")

        self.processing_thread.start()

    def toggle_pause_inspection(self):
        if self.processing_thread.isRunning():
            is_paused = self.processing_thread.toggle_pause()
            if is_paused:
                self.btn_pause.setText("Resume")
                self.status_bar.showMessage("Paused")
                self.lbl_status_run.setText("Status: PAUSED")
            else:
                self.btn_pause.setText("Pause")
                self.status_bar.showMessage("Running...")
                self.lbl_status_run.setText("Status: RUNNING")

    def trigger_realign(self):
        if self.processing_thread.isRunning():
            self.processing_thread.trigger_realign()
            self.status_bar.showMessage("Realignment triggered.")

    def stop_inspection(self):
        if self.processing_thread.isRunning():
            self.processing_thread.stop_processing()
            self.status_bar.showMessage("Stopping inspection thread...")
            self.btn_stop.setEnabled(False)
            self.btn_pause.setEnabled(False)
            self.btn_realign.setEnabled(False)

    def on_processing_finished(self):
        self.status_bar.showMessage("Stopped")
        self.video_display_label.setText("Camera stream inactive. Click 'Start Inspection' to begin.")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_realign.setEnabled(False)
        self.btn_pause.setText("Pause")
        self.lbl_status_run.setText("Status: IDLE")

    @pyqtSlot(np.ndarray)
    def update_video_display(self, cv_img):
        try:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.video_display_label.setPixmap(pixmap.scaled(
                self.video_display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            self.lbl_status_run.setText("Status: RUNNING")
        except Exception as e:
            print(f"[ERROR] GUI: Update frame display failed: {e}")

    @pyqtSlot(dict)
    def update_stats_display(self, stats):
        self.lbl_fps.setText(f"FPS: {stats.get('fps', 0.0):.1f}")
        
        align = stats.get('alignment_status', False)
        self.lbl_alignment.setText(f"Alignment: {'OK' if align else 'FAILED'}")
        self.lbl_alignment.setStyleSheet(f"color: {'green' if align else 'red'};")

        self.lbl_detected.setText(f"Detected: {stats.get('detected', 0)}")
        self.lbl_matched.setText(f"Matched: {stats.get('matched', 0)}")
        
        missing = stats.get('missing', 0)
        self.lbl_missing.setText(f"Missing: {missing}")
        self.lbl_missing.setStyleSheet(f"color: {'red' if missing > 0 else 'green'};")

        unexpected = stats.get('unexpected', 0)
        self.lbl_unexpected.setText(f"Unexpected: {unexpected}")
        self.lbl_unexpected.setStyleSheet(f"color: {'orange' if unexpected > 0 else 'black'};")

        self.lbl_pass_rate.setText(f"Pass Rate: {stats.get('pass_rate', 0.0):.1f}%")
        self.lbl_pass_rate.setStyleSheet(f"color: {'green' if stats.get('pass_rate', 0.0) == 100.0 else 'red'};")
        
        self.lbl_produced_qty.setText(f"Produced Qty: {stats.get('produced_qty', 0)}")
        self.lbl_good_qty.setText(f"Good Qty: {stats.get('good_qty', 0)}")
        self.lbl_ng_qty.setText(f"NG Qty: {stats.get('ng_qty', 0)}")

    def closeEvent(self, event):
        print("[INFO] Close event triggered. Shutting down worker thread...")
        self.stop_inspection()
        if self.processing_thread.isRunning():
            self.processing_thread.wait(3000)
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    config = Config()
    main_window = MainWindow(config)
    main_window.show()
    sys.exit(app.exec_())
