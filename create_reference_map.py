# -*- coding: utf-8 -*-
import cv2
import json
import numpy as np
import time
import torch
from ultralytics import YOLO
import os
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog

# --- Default Configurations ---
DEFAULT_CAMERA_URL = "rtsp://192.168.0.101:8080/h264.sdp"
DEFAULT_MODEL_PATH = r"C:\Users\Namdr\Downloads\best (1).pt"
DEFAULT_CLASSES_PATH = r"C:\Users\Namdr\Downloads\dataset\classes.txt"
DEFAULT_OUTPUT_JSON = r"C:\Users\Namdr\Downloads\reference_map_generated.json"

WINDOW_NAME = "Golden Reference Map Creator - Interactive"
DRAG_MARGIN = 10  # Pixel threshold for resizing handles

# --- Global Variables ---
components = []
frame_original = None
annotated_frame = None
model_labels = {}
selected_box_index = -1
resizing_mode = None  # "move", "tl", "tr", "bl", "br", "top", "bottom", "left", "right"
mouse_down_pos = None
original_selected_bbox = None
drawing_manual = False
manual_start_point = (-1, -1)
manual_temp_end = (-1, -1)

def load_yolo_model(model_path, classes_path=""):
    global model_labels
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Using device: {device}")

        model = YOLO(model_path)
        model.to(device)

        if hasattr(model, 'names'):
            model_labels = model.names
            if isinstance(model_labels, list):
                model_labels = {i: name for i, name in enumerate(model_labels)}
            elif not isinstance(model_labels, dict):
                print("[WARNING] Unsupported model.names format. Reading from classes.txt...")
                model_labels = None

        if model_labels is None and classes_path and os.path.exists(classes_path):
            print(f"[INFO] Loading labels from file: {classes_path}")
            try:
                with open(classes_path, 'r', encoding='utf-8') as f:
                    labels_list = [line.strip() for line in f if line.strip()]
                    model_labels = {i: name for i, name in enumerate(labels_list)}
            except Exception as e:
                print(f"[ERROR] Loading labels file failed: {e}")
                model_labels = None

        if not model_labels:
            print("[WARNING] No class labels loaded. Generating default labels (ID_0, ID_1, ...).")
            model_labels = {i: f"ID_{i}" for i in range(100)}

        print(f"[INFO] Model loaded successfully. Classes: {model_labels}")
        return model, device, model_labels

    except Exception as e:
        print(f"[FATAL] Failed to load YOLO model: {e}")
        return None, None, None

def draw_all_annotations(image, comp_list, current_sel_idx=-1):
    output_image = image.copy()
    
    # 1. Draw confirmed/suggested boxes
    for i, comp in enumerate(comp_list):
        bbox = comp.get("bbox_ref")
        label = comp.get("label", "N/A")
        if not bbox or len(bbox) != 4:
            continue

        x, y, w, h = map(int, bbox)
        color = (0, 255, 0)  # Green for normal boxes
        thickness = 2

        if i == current_sel_idx:
            color = (0, 0, 255)  # Red for currently selected box
            thickness = 3

        cv2.rectangle(output_image, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(output_image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness - 1 if thickness > 1 else 1)

        # Draw handles for the selected box
        if i == current_sel_idx:
            handles = [
                (x, y), (x + w // 2, y), (x + w, y),
                (x, y + h // 2), (x + w, y + h // 2),
                (x, y + h), (x + w // 2, y + h), (x + w, y + h)
            ]
            for hx, hy in handles:
                cv2.circle(output_image, (hx, hy), DRAG_MARGIN // 2 + 1, (255, 255, 255), -1)
                cv2.circle(output_image, (hx, hy), DRAG_MARGIN // 2, color, -1)

    # 2. Draw active manual bounding box (magenta while drawing)
    if drawing_manual and manual_start_point != (-1, -1) and manual_temp_end != (-1, -1):
        x1, y1 = manual_start_point
        x2, y2 = manual_temp_end
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 255), 1)
        cv2.putText(output_image, "Drawing...", (min(x1, x2), min(y1, y2) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

    return output_image

def get_resize_mode(bbox_x, bbox_y, bbox_w, bbox_h, mouse_x, mouse_y):
    if abs(mouse_x - bbox_x) < DRAG_MARGIN and abs(mouse_y - bbox_y) < DRAG_MARGIN: return "tl"
    if abs(mouse_x - (bbox_x + bbox_w)) < DRAG_MARGIN and abs(mouse_y - bbox_y) < DRAG_MARGIN: return "tr"
    if abs(mouse_x - bbox_x) < DRAG_MARGIN and abs(mouse_y - (bbox_y + bbox_h)) < DRAG_MARGIN: return "bl"
    if abs(mouse_x - (bbox_x + bbox_w)) < DRAG_MARGIN and abs(mouse_y - (bbox_y + bbox_h)) < DRAG_MARGIN: return "br"

    if abs(mouse_y - bbox_y) < DRAG_MARGIN and bbox_x <= mouse_x <= bbox_x + bbox_w: return "top"
    if abs(mouse_y - (bbox_y + bbox_h)) < DRAG_MARGIN and bbox_x <= mouse_x <= bbox_x + bbox_w: return "bottom"
    if abs(mouse_x - bbox_x) < DRAG_MARGIN and bbox_y <= mouse_y <= bbox_y + bbox_h: return "left"
    if abs(mouse_x - (bbox_x + bbox_w)) < DRAG_MARGIN and bbox_y <= mouse_y <= bbox_y + bbox_h: return "right"

    if bbox_x < mouse_x < bbox_x + bbox_w and bbox_y < mouse_y < bbox_y + bbox_h: return "move"
    return None

def mouse_callback(event, x, y, flags, param):
    global components, frame_original, annotated_frame, model_labels
    global selected_box_index, resizing_mode, mouse_down_pos, original_selected_bbox
    global drawing_manual, manual_start_point, manual_temp_end

    if frame_original is None: return

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_down_pos = (x, y)
        new_selection_index = -1
        temp_resizing_mode = None

        # Check if click hits a box or handles of existing selected box
        for i, comp in reversed(list(enumerate(components))):
            if "bbox_ref" not in comp: continue
            bx, by, bw, bh = comp["bbox_ref"]
            mode = get_resize_mode(bx, by, bw, bh, x, y)
            if mode:
                new_selection_index = i
                temp_resizing_mode = mode
                break

        selected_box_index = new_selection_index
        if selected_box_index != -1:
            resizing_mode = temp_resizing_mode
            original_selected_bbox = list(components[selected_box_index]["bbox_ref"])
            drawing_manual = False
        else:
            resizing_mode = None
            original_selected_bbox = None
            # Start manual box drawing
            drawing_manual = True
            manual_start_point = (x, y)
            manual_temp_end = (x, y)

        annotated_frame = draw_all_annotations(frame_original, components, selected_box_index)
        cv2.imshow(WINDOW_NAME, annotated_frame)

    elif event == cv2.EVENT_MOUSEMOVE:
        if selected_box_index != -1 and mouse_down_pos is not None and resizing_mode is not None and (flags & cv2.EVENT_FLAG_LBUTTON):
            if original_selected_bbox is None: return
            dx = x - mouse_down_pos[0]
            dy = y - mouse_down_pos[1]
            new_bbox = list(original_selected_bbox)

            if resizing_mode == "move":
                new_bbox[0] = original_selected_bbox[0] + dx
                new_bbox[1] = original_selected_bbox[1] + dy
            elif resizing_mode == "tl":
                new_bbox[0] = original_selected_bbox[0] + dx
                new_bbox[1] = original_selected_bbox[1] + dy
                new_bbox[2] = original_selected_bbox[2] - dx
                new_bbox[3] = original_selected_bbox[3] - dy
            elif resizing_mode == "br":
                new_bbox[2] = original_selected_bbox[2] + dx
                new_bbox[3] = original_selected_bbox[3] + dy
            elif resizing_mode == "tr":
                new_bbox[1] = original_selected_bbox[1] + dy
                new_bbox[2] = original_selected_bbox[2] + dx
                new_bbox[3] = original_selected_bbox[3] - dy
            elif resizing_mode == "bl":
                new_bbox[0] = original_selected_bbox[0] + dx
                new_bbox[2] = original_selected_bbox[2] - dx
                new_bbox[3] = original_selected_bbox[3] + dy
            elif resizing_mode == "top":
                new_bbox[1] = original_selected_bbox[1] + dy
                new_bbox[3] = original_selected_bbox[3] - dy
            elif resizing_mode == "bottom":
                new_bbox[3] = original_selected_bbox[3] + dy
            elif resizing_mode == "left":
                new_bbox[0] = original_selected_bbox[0] + dx
                new_bbox[2] = original_selected_bbox[2] - dx
            elif resizing_mode == "right":
                new_bbox[2] = original_selected_bbox[2] + dx

            if new_bbox[2] < DRAG_MARGIN: new_bbox[2] = DRAG_MARGIN
            if new_bbox[3] < DRAG_MARGIN: new_bbox[3] = DRAG_MARGIN

            components[selected_box_index]["bbox_ref"] = [int(val) for val in new_bbox]
            annotated_frame = draw_all_annotations(frame_original, components, selected_box_index)
            cv2.imshow(WINDOW_NAME, annotated_frame)
            
        elif drawing_manual and manual_start_point != (-1, -1):
            manual_temp_end = (x, y)
            annotated_frame = draw_all_annotations(frame_original, components, selected_box_index)
            cv2.imshow(WINDOW_NAME, annotated_frame)

    elif event == cv2.EVENT_LBUTTONUP:
        if selected_box_index != -1:
            print(f"[INFO] Modified box {selected_box_index}: {components[selected_box_index]['bbox_ref']}")
        elif drawing_manual and manual_start_point != (-1, -1):
            drawing_manual = False
            x1, y1 = manual_start_point
            x2, y2 = manual_temp_end
            
            x_min, y_min = min(x1, x2), min(y1, y2)
            width, height = abs(x2 - x1), abs(y2 - y1)
            
            if width > 10 and height > 10:
                # Open Tkinter popup dialog to get the label
                root = tk.Tk()
                root.withdraw()  # Hide main window
                label = simpledialog.askstring("New Bounding Box", "Enter component label name:", parent=root)
                root.destroy()
                
                if label and label.strip():
                    components.append({
                        "label": label.strip(),
                        "bbox_ref": [x_min, y_min, width, height],
                        "confidence": 1.0
                    })
                    print(f"[INFO] Manually added box: '{label.strip()}' at {[x_min, y_min, width, height]}")
            
            manual_start_point = (-1, -1)
            manual_temp_end = (-1, -1)
            
            annotated_frame = draw_all_annotations(frame_original, components, selected_box_index)
            cv2.imshow(WINDOW_NAME, annotated_frame)
            
        mouse_down_pos = None

def detect_components(frame, model, device, current_labels, conf_threshold):
    if model is None:
        return []
    
    detected = []
    try:
        results = model.predict(frame, verbose=False, conf=conf_threshold, device=device)
        if results and results[0].boxes is not None:
            box_data = results[0].boxes.data.cpu().numpy()
            for box in box_data:
                if len(box) >= 6:
                    x1, y1, x2, y2, conf, label_id = box[:6]
                    lbl_name = current_labels.get(int(label_id), f"ID_{int(label_id)}")
                    w, h = int(x2 - x1), int(y2 - y1)
                    if w > 0 and h > 0:
                        detected.append({
                            "label": lbl_name,
                            "bbox_ref": [int(x1), int(y1), w, h],
                            "confidence": float(conf)
                        })
    except Exception as e:
        print(f"[ERROR] YOLO prediction failed: {e}")
    print(f"[INFO] YOLO detected {len(detected)} reference components.")
    return detected

def save_to_json(data, path):
    try:
        dir_name = os.path.dirname(path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Saved {len(data)} components to Golden Reference Map: {path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save JSON reference map: {e}")
        return False

def main():
    global components, frame_original, annotated_frame, model_labels
    global selected_box_index, resizing_mode, mouse_down_pos, original_selected_bbox

    # Show simple Tkinter setup dialog for loading paths
    root = tk.Tk()
    root.title("Setup Paths")
    root.geometry("550x300")
    
    # Path variables
    model_path_var = tk.StringVar(value=DEFAULT_MODEL_PATH)
    classes_path_var = tk.StringVar(value=DEFAULT_CLASSES_PATH)
    output_path_var = tk.StringVar(value=DEFAULT_OUTPUT_JSON)
    camera_url_var = tk.StringVar(value=DEFAULT_CAMERA_URL)
    image_file_var = tk.StringVar(value="")
    use_cam_var = tk.BooleanVar(value=False)

    # UI Widgets
    tk.Label(root, text="YOLO Model (.pt):").grid(row=0, column=0, sticky="w", padx=10, pady=5)
    tk.Entry(root, textvariable=model_path_var, width=50).grid(row=0, column=1, padx=5, pady=5)
    tk.Button(root, text="Browse", command=lambda: model_path_var.set(filedialog.askopenfilename(filetypes=[("YOLO Model", "*.pt")]))).grid(row=0, column=2, padx=5, pady=5)

    tk.Label(root, text="classes.txt:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
    tk.Entry(root, textvariable=classes_path_var, width=50).grid(row=1, column=1, padx=5, pady=5)
    tk.Button(root, text="Browse", command=lambda: classes_path_var.set(filedialog.askopenfilename(filetypes=[("Text File", "*.txt")]))).grid(row=1, column=2, padx=5, pady=5)

    tk.Label(root, text="Output JSON:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
    tk.Entry(root, textvariable=output_path_var, width=50).grid(row=2, column=1, padx=5, pady=5)
    tk.Button(root, text="Browse", command=lambda: output_path_var.set(filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON File", "*.json")]))).grid(row=2, column=2, padx=5, pady=5)

    tk.Label(root, text="Local Image File:").grid(row=3, column=0, sticky="w", padx=10, pady=5)
    tk.Entry(root, textvariable=image_file_var, width=50).grid(row=3, column=1, padx=5, pady=5)
    tk.Button(root, text="Browse", command=lambda: image_file_var.set(filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]))).grid(row=3, column=2, padx=5, pady=5)

    tk.Label(root, text="IP Camera Stream:").grid(row=4, column=0, sticky="w", padx=10, pady=5)
    tk.Entry(root, textvariable=camera_url_var, width=50).grid(row=4, column=1, padx=5, pady=5)
    tk.Checkbutton(root, text="Use Camera", variable=use_cam_var).grid(row=4, column=2, padx=5, pady=5)

    def on_submit():
        if use_cam_var.get() == False and not image_file_var.get():
            messagebox.showerror("Error", "Please select either a local image file or check 'Use Camera'")
            return
        root.quit()

    tk.Button(root, text="Start Creator", command=on_submit, bg="green", fg="white", font=("Arial", 11, "bold")).grid(row=5, column=1, pady=15)
    
    root.mainloop()
    
    # Extract values
    model_p = model_path_var.get()
    classes_p = classes_path_var.get()
    output_p = output_path_var.get()
    camera_url = camera_url_var.get()
    image_p = image_file_var.get()
    use_cam = use_cam_var.get()
    root.destroy()

    if not model_p or not output_p:
        print("[ERROR] Missing required paths.")
        return

    model, device, loaded_labels = load_yolo_model(model_p, classes_p)
    if model is None: return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    while True:
        if use_cam:
            print(f"[INFO] Connecting to camera: {camera_url}...")
            cap = cv2.VideoCapture(camera_url)
            if not cap.isOpened():
                print(f"[WARNING] Cannot open camera: {camera_url}. Falling back to default WebCam(0)...")
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("[ERROR] Cannot open default Webcam. Exiting.")
                    cv2.destroyAllWindows()
                    return
            
            print("[INFO] Camera connected! Capturing frame in 1 second...")
            time.sleep(1.0)
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                print("[ERROR] Failed to capture frame. Retrying...")
                cv2.waitKey(2000)
                continue
        else:
            print(f"[INFO] Reading local image file: {image_p}")
            try:
                img_np = np.fromfile(image_p, dtype=np.uint8)
                frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            except Exception:
                frame = cv2.imread(image_p)
                
            if frame is None:
                print(f"[ERROR] Cannot open image file: {image_p}. Exiting.")
                cv2.destroyAllWindows()
                return

        frame_original = frame.copy()
        
        # Reset mouse state
        selected_box_index = -1
        resizing_mode = None
        mouse_down_pos = None
        original_selected_bbox = None

        # Predict components using YOLO
        components = detect_components(frame_original, model, device, model_labels, 0.35)

        # Draw and show window
        annotated_frame = draw_all_annotations(frame_original, components, selected_box_index)
        cv2.imshow(WINDOW_NAME, annotated_frame)

        print("\n=== GOLDEN REFERENCE MAP CREATOR INSTRUCTIONS ===")
        print("  - CLICK & DRAG a box to MOVE it.")
        print("  - CLICK & DRAG handles (corners/edges) of selected red box to RESIZE it.")
        print("  - CLICK & DRAG empty space to DRAW a new box (Input label in popup).")
        print("  - Press 'd' to DELETE the selected box.")
        print("  - Press 's' to SAVE reference map and EXIT.")
        print("  - Press 'r' to RE-CAPTURE snapshot (if using Camera).")
        print("  - Press 'q' or Escape to QUIT without saving.")
        print("===================================================\n")

        inner_loop = True
        while inner_loop:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print("[INFO] Window closed by user. Exiting.")
                cv2.destroyAllWindows()
                return

            key = cv2.waitKey(100) & 0xFF
            
            if key == 27 or key == ord('q'):
                print("[INFO] Exited without saving.")
                cv2.destroyAllWindows()
                return

            elif key == ord('s'):
                save_data = []
                for comp in components:
                    save_data.append({
                        "label": comp["label"],
                        "bbox_ref": comp["bbox_ref"]
                    })
                
                if not save_data:
                    root = tk.Tk()
                    root.withdraw()
                    confirm = messagebox.askyesno("Confirm Exit", "Golden reference map has no components. Still save and exit?")
                    root.destroy()
                    if not confirm:
                        continue
                
                if save_to_json(save_data, output_p):
                    cv2.destroyAllWindows()
                    return

            elif key == ord('d'):
                if selected_box_index != -1 and selected_box_index < len(components):
                    removed = components.pop(selected_box_index)
                    print(f"[INFO] Deleted component: '{removed['label']}'")
                    selected_box_index = -1
                    resizing_mode = None
                    annotated_frame = draw_all_annotations(frame_original, components, selected_box_index)
                    cv2.imshow(WINDOW_NAME, annotated_frame)
                else:
                    print("[WARNING] Click on a box first to select it for deletion.")

            elif key == ord('r'):
                if use_cam:
                    print("[INFO] Re-capturing frame from stream...")
                    inner_loop = False
                else:
                    print("[WARNING] Re-capture is only available when 'Use Camera' is active.")

if __name__ == "__main__":
    main()
