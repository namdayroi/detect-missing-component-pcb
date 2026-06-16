# -*- coding: utf-8 -*-
import cv2
import numpy as np
import json
import math
import time
import threading
from queue import Queue, Empty
import torch
import os

# --- Configurations ---
class Config:
    # Paths (adjust as needed)
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

    # Visual Styles
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

# --- Helper Alignment & Geometry Math ---
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
    """Matches anchor labels between current frame detections and reference map."""
    # Anchors are chosen dynamically as matching high-confidence components
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
        
        # Add a dummy row to create 3x3 to compute inverse
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

# --- Main Inspector Engine ---
class PCBInspectorEngine:
    def __init__(self, config=Config()):
        self.config = config
        self.model = None
        self.model_labels = {}
        self.reference_map_dict = None
        self.expected_components_list = None
        self.webcam_stream = None

        # States
        self.is_paused = False
        self.alignment_status = False
        self.current_M_det_to_ref = None
        self.current_M_ref_to_det = None
        
        self.last_processed_components = []
        self.last_matched_indices = set()
        self.last_missing_components_info = []
        self.last_unmatched_indices = set()

        self.total_frame_count = 0
        self.processed_frame_count = 0
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.last_fps = 0.0

        self._load_resources()

    def _load_resources(self):
        print("[INFO] Loading YOLO model...")
        try:
            self.model = YOLO(self.config.PATH_TO_YOLO_MODEL)
            self.model.to(self.config.DEVICE)
            self.model_labels = load_yolo_labels(self.model, self.config.PATH_TO_CLASSES_TXT)
            print(f"[INFO] AI Model loaded successfully with {len(self.model_labels)} classes.")
        except Exception as e:
            print(f"[FATAL] YOLO load error: {e}")
            raise SystemExit("YOLO load error")

        print("[INFO] Loading Reference Golden Map...")
        self.reference_map_dict, self.expected_components_list = load_reference_map(self.config.PATH_TO_REFERENCE_MAP)
        if not self.expected_components_list:
            raise SystemExit("Failed loading reference map")

        # Handle fixed alignment loading
        if self.config.USE_FIXED_ALIGNMENT:
            loaded_M = load_fixed_alignment_matrix(self.config.PATH_TO_FIXED_ALIGNMENT_MATRIX)
            if loaded_M is not None:
                self.current_M_ref_to_det = loaded_M
                # M3x3 check
                M3x3 = np.vstack([self.current_M_ref_to_det, [0, 0, 1]])
                if abs(np.linalg.det(M3x3)) > 1e-6:
                    self.current_M_det_to_ref = cv2.invertAffineTransform(self.current_M_ref_to_det)
                    self.alignment_status = True
                    print("[INFO] Loaded fixed alignment matrix successfully.")
                else:
                    print("[WARNING] Loaded matrix is singular. Re-calculating alignment dynamically.")

    def _recalculate_and_save_fixed_alignment(self, det_comps):
        print("[INFO] Re-calculating and saving fixed alignment matrix...")
        ref_coords, det_coords = find_anchor_points(det_comps, self.reference_map_dict)
        if len(ref_coords) >= 3:
            M_det_to_ref, M_ref_to_det = calculate_affine_transform(ref_coords, det_coords)
            if M_ref_to_det is not None and M_det_to_ref is not None:
                self.current_M_det_to_ref = M_det_to_ref
                self.current_M_ref_to_det = M_ref_to_det
                self.alignment_status = True
                save_affine_matrix(self.current_M_ref_to_det, self.config.PATH_TO_FIXED_ALIGNMENT_MATRIX)
                print("[INFO] Saved fixed alignment matrix successfully.")
            else:
                print("[WARNING] Affine computation failed.")
                self.alignment_status = False
        else:
            print(f"[WARNING] Not enough anchor components matched ({len(ref_coords)}/3 required).")
            self.alignment_status = False

    def start_camera(self):
        try:
            self.webcam_stream = WebcamStream(src=self.config.IP_WEBCAM_URL,
                                              read_timeout=self.config.WEBCAM_READ_TIMEOUT).start()
            time.sleep(2.0)  # Wait for stream thread to fill buffer
            test_frame = self.webcam_stream.read()
            if test_frame is None:
                raise ValueError("Webcam stream is empty.")
            print("[INFO] Camera stream started successfully.")
        except Exception as e:
            print(f"[FATAL] Camera initialization error: {e}")
            if self.webcam_stream:
                self.webcam_stream.stop()
            raise SystemExit("Camera error")

    def run(self):
        self.start_camera()
        print("\n=== RUNNING PCB INSPECTION ENGINE ===")
        print("  - Press 'p' to PAUSE/RESUME.")
        print("  - Press 'r' to RE-ALIGN and save fixed matrix (in Fixed mode).")
        print("  - Press 'q' to QUIT.")
        print("======================================\n")

        try:
            while True:
                frame = self.webcam_stream.read()
                if frame is None:
                    if self.webcam_stream.stopped:
                        break
                    time.sleep(0.01)
                    continue

                self.total_frame_count += 1
                display_image = frame.copy()

                if not self.is_paused:
                    if self.total_frame_count % self.config.FRAME_SKIP_INTERVAL == 0:
                        self._process_frame(frame)

                self._update_visualization(display_image)

                # Resize to fit screen
                h, w = display_image.shape[:2]
                scale = min(1280 / w, 720 / h)
                nw, nh = int(w * scale), int(h * scale)
                display_image_resized = cv2.resize(display_image, (nw, nh), interpolation=cv2.INTER_AREA)

                cv2.imshow("PCB Component Inspection (AOI Engine)", display_image_resized)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[INFO] Stopping inspector...")
                    break
                elif key == ord('p'):
                    self.is_paused = not self.is_paused
                    print(f"[INFO] Inspection: {'PAUSED' if self.is_paused else 'RESUMED'}")
                elif key == ord('r'):
                    if self.config.USE_FIXED_ALIGNMENT:
                        recalc_frame = self.webcam_stream.read()
                        if recalc_frame is not None:
                            self._trigger_fixed_realignment(recalc_frame)
                    else:
                        print("[INFO] Realignment command ignored (USE_FIXED_ALIGNMENT is False).")
        finally:
            self._cleanup()

    def _trigger_fixed_realignment(self, frame):
        h_frame, w_frame = frame.shape[:2]
        resized = cv2.resize(frame, self.config.INFERENCE_SIZE, interpolation=cv2.INTER_LINEAR)
        results = self.model.predict(resized, verbose=False,
                                     conf=self.config.STRICT_YOLO_CONFIDENCE_THRESHOLD,
                                     device=self.config.DEVICE, half=self.config.USE_HALF)
        dets = []
        if results and results[0].boxes is not None:
            scale_x = w_frame / self.config.INFERENCE_SIZE[0]
            scale_y = h_frame / self.config.INFERENCE_SIZE[1]
            for box in results[0].boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, label_id = box[:6]
                dets.append({
                    'label': self.model_labels.get(int(label_id), f"ID_{int(label_id)}"),
                    'center_det': (int((x1 + (x2 - x1) / 2) * scale_x),
                                   int((y1 + (y2 - y1) / 2) * scale_y)),
                    'confidence': conf
                })
        if dets:
            self._recalculate_and_save_fixed_alignment(dets)
        else:
            print("[WARNING] Could not find any high-confidence anchors in current frame.")

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
            print(f"[ERROR] YOLO Predict Exception: {e}")

        # --- Alignment processing ---
        if self.config.USE_FIXED_ALIGNMENT and not self.alignment_status:
            # Recalculate fixed alignment initially
            if current_dets:
                high_conf = [d for d in current_dets if d['is_high_confidence_detection']]
                self._recalculate_and_save_fixed_alignment(high_conf)
        elif not self.config.USE_FIXED_ALIGNMENT:
            # Dynamic alignment calculated periodically
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

        # --- Component Matching ---
        current_matched_indices = set()
        current_missing_info = []
        current_unmatched_indices = set()

        if self.alignment_status and self.current_M_ref_to_det is not None:
            # Collect expected component coordinates mapped to current frame
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
                        
                        # Geometric validations
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

            # Contextual filters for unmatched boxes
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
            # Alignment failed: all detections are unmatched, all expected are missing
            current_unmatched_indices = set(range(len(current_dets)))
            for i in range(len(current_dets)):
                current_dets[i]['is_contextually_verified'] = True
            current_missing_info = list(self.expected_components_list)

        self.last_matched_indices = current_matched_indices
        self.last_missing_components_info = current_missing_info
        self.last_unmatched_indices = current_unmatched_indices

    def _update_visualization(self, image):
        # 1. Status Bar Panel
        panel_h = self.config.STATUS_PANEL_HEIGHT
        cv2.rectangle(image, (0, 0), (image.shape[1], panel_h), (0, 0, 0), -1)

        # 2. Draw Matched Boxes
        for idx in self.last_matched_indices:
            if idx < len(self.last_processed_components):
                comp = self.last_processed_components[idx]
                bbox = comp['bbox_det']
                color = self.config.COLOR_MATCHED if comp['confidence'] >= self.config.STRICT_YOLO_CONFIDENCE_THRESHOLD else self.config.COLOR_MATCHED_LOW_CONF
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(image, f"{comp['label']} ({comp['confidence']:.2f})", (bbox[0], bbox[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # 3. Draw Unexpected / Extra Boxes
        for idx in self.last_unmatched_indices:
            if idx < len(self.last_processed_components):
                comp = self.last_processed_components[idx]
                if comp.get('is_contextually_verified', False):
                    bbox = comp['bbox_det']
                    color = self.config.COLOR_UNEXPECTED if comp['confidence'] >= self.config.STRICT_YOLO_CONFIDENCE_THRESHOLD else self.config.COLOR_UNEXPECTED_LOW_CONF
                    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    cv2.putText(image, f"Extra: {comp['label']} ({comp['confidence']:.2f})", (bbox[0], bbox[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # 4. Draw Missing Boxes
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

        # 5. Draw Top Status Labels
        self.fps_frame_count += 1
        curr_time = time.time()
        time_diff = curr_time - self.fps_start_time
        if time_diff >= 1.0:
            self.last_fps = self.fps_frame_count / time_diff
            self.fps_start_time = curr_time
            self.fps_frame_count = 0

        align_text = f"Alignment: {'OK' if self.alignment_status else 'FAILED'}"
        align_color = self.config.COLOR_ALIGN_OK if self.alignment_status else self.config.COLOR_ALIGN_FAIL
        cv2.putText(image, align_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, align_color, 2, cv2.LINE_AA)
        
        cv2.putText(image, f"FPS: {self.last_fps:.1f}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.config.COLOR_FPS, 1, cv2.LINE_AA)
        
        stat_text = f"Det: {len(self.last_processed_components)} | Match: {len(self.last_matched_indices)} | Miss: {len(self.last_missing_components_info)} | Extra: {len(self.last_unmatched_indices)}"
        cv2.putText(image, stat_text, (350, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.config.COLOR_INFO_TEXT, 2, cv2.LINE_AA)

    def _cleanup(self):
        print("[INFO] Cleaning up camera and window resources...")
        if self.webcam_stream:
            self.webcam_stream.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    inspector = PCBInspectorEngine()
    inspector.run()
