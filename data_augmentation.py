# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import random
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import concurrent.futures
import threading

# ==============================================================================
# 1. CORE IMAGE PROCESSING & AUGMENTATION FUNCTIONS
# ==============================================================================

def rotate_image_and_bboxes(img, bboxes_yolo, angle_degrees):
    h, w = img.shape[:2]
    center_x, center_y = w / 2, h / 2

    # Compute rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle_degrees, 1.0)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    rotation_matrix[0, 2] += (new_w / 2) - center_x
    rotation_matrix[1, 2] += (new_h / 2) - center_y

    border_color = (114, 114, 114)  # Padding color (gray)
    rotated_image = cv2.warpAffine(img, rotation_matrix, (new_w, new_h),
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=border_color)

    new_annotations = []
    for bbox in bboxes_yolo:
        class_id = bbox[0]
        cx_norm, cy_norm, w_norm, h_norm = map(float, bbox[1:])

        cx = cx_norm * w
        cy = cy_norm * h
        rw = w_norm * w
        rh = h_norm * h
        x_min = cx - rw / 2
        y_min = cy - rh / 2
        x_max = cx + rw / 2
        y_max = cy + rh / 2

        # Transform bbox corners
        corners = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
        ones = np.ones(shape=(len(corners), 1))
        points_ones = np.hstack([corners, ones])
        transformed_corners = rotation_matrix.dot(points_ones.T).T

        new_x_min = np.min(transformed_corners[:, 0])
        new_y_min = np.min(transformed_corners[:, 1])
        new_x_max = np.max(transformed_corners[:, 0])
        new_y_max = np.max(transformed_corners[:, 1])

        new_x_min_clip = max(0.0, new_x_min)
        new_y_min_clip = max(0.0, new_y_min)
        new_x_max_clip = min(float(new_w), new_x_max)
        new_y_max_clip = min(float(new_h), new_y_max)

        new_box_w = new_x_max_clip - new_x_min_clip
        new_box_h = new_y_max_clip - new_y_min_clip

        if new_box_w > 1 and new_box_h > 1:
            new_cx = (new_x_min_clip + new_x_max_clip) / 2
            new_cy = (new_y_min_clip + new_y_max_clip) / 2

            new_cx_norm = np.clip(new_cx / new_w, 0.0, 1.0)
            new_cy_norm = np.clip(new_cy / new_h, 0.0, 1.0)
            new_w_norm = np.clip(new_box_w / new_w, 0.0, 1.0)
            new_h_norm = np.clip(new_box_h / new_h, 0.0, 1.0)

            if new_w_norm > 1e-6 and new_h_norm > 1e-6:
                new_annotation_str = f"{class_id} {new_cx_norm:.6f} {new_cy_norm:.6f} {new_w_norm:.6f} {new_h_norm:.6f}"
                new_annotations.append(new_annotation_str)

    return rotated_image, new_annotations


def apply_hsv_enhancements(img, brightness_range=(-30, 30), contrast_range=(0.7, 1.3), saturation_range=(0.6, 1.4)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 1. Saturation
    sat_factor = random.uniform(*saturation_range)
    s = np.clip(s * sat_factor, 0, 255).astype(np.uint8)

    # Recombine to apply brightness & contrast in BGR/HSV
    hsv = cv2.merge([h, s, v])
    augmented = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 2. Contrast & Brightness
    alpha = random.uniform(*contrast_range)
    beta = random.randint(*brightness_range)
    augmented = np.clip(augmented * alpha + beta, 0, 255).astype(np.uint8)

    return augmented


def apply_random_erasing(img, num_regions_max=3, size_ratio_min=0.02, size_ratio_max=0.08, fill_color=(114, 114, 114)):
    augmented = img.copy()
    h, w = img.shape[:2]
    num_regions = random.randint(1, num_regions_max)

    for _ in range(num_regions):
        # Determine size of erasing area
        area = h * w
        erasing_area = random.uniform(size_ratio_min, size_ratio_max) * area
        aspect_ratio = random.uniform(0.3, 3.3)

        eh = int(np.sqrt(erasing_area / aspect_ratio))
        ew = int(np.sqrt(erasing_area * aspect_ratio))

        eh = max(2, min(eh, h - 2))
        ew = max(2, min(ew, w - 2))

        # Random position
        ey = random.randint(0, h - eh)
        ex = random.randint(0, w - ew)

        augmented[ey:ey + eh, ex:ex + ew] = fill_color

    return augmented


# ==============================================================================
# 2. MULTI-THREADED FILE PROCESSOR
# ==============================================================================

def process_single_file(filename, input_image_dir, input_label_dir, output_image_dir, output_label_dir,
                        settings, progress_callback):
    """Processes one image and its labels, generating augmented variations."""
    generated_count = 0
    image_path = os.path.join(input_image_dir, filename)
    base_name, _ = os.path.splitext(filename)
    annotation_path = os.path.join(input_label_dir, base_name + '.txt')

    # Load image safely (handles unicode paths on Windows)
    img = None
    try:
        img_np = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imread(image_path)
    except Exception:
        pass

    if img is None:
        progress_callback(0)
        return

    # Load annotations (YOLO format)
    bboxes = []
    if os.path.exists(annotation_path):
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        bboxes.append(parts)
        except Exception:
            pass

    # Generate Rotation variations
    angle_list = settings.get("angles", [])
    if settings.get("use_rotation") and angle_list:
        for angle in angle_list:
            rotated_img, rotated_ann = rotate_image_and_bboxes(img, bboxes, angle)
            if rotated_img is None: continue
            
            # Sub-augmentation: optionally apply HSV & Erasing on rotated images
            final_img = rotated_img.copy()
            if settings.get("use_hsv"):
                final_img = apply_hsv_enhancements(final_img)
            if settings.get("use_erasure"):
                final_img = apply_random_erasing(final_img)

            angle_suffix = f"rot_{int(round(angle)):03d}"
            out_img_name = f"{base_name}_{angle_suffix}.jpg"
            out_ann_name = f"{base_name}_{angle_suffix}.txt"

            out_img_path = os.path.join(output_image_dir, out_img_name)
            out_ann_path = os.path.join(output_label_dir, out_ann_name)

            # Write image (safely)
            is_success, im_buf = cv2.imencode(".jpg", final_img)
            if is_success:
                im_buf.tofile(out_img_path)
                generated_count += 1
                if rotated_ann:
                    try:
                        with open(out_ann_path, 'w', encoding='utf-8') as f:
                            for ann in rotated_ann: f.write(ann + "\n")
                    except Exception: pass
    
    # Generate HSV & Erasing variations without rotation (if rotation is not applied or as extra)
    # Generate N straight augmented copies
    num_flat_copies = settings.get("flat_copies", 0)
    if num_flat_copies > 0 and (settings.get("use_hsv") or settings.get("use_erasure")):
        for copy_idx in range(num_flat_copies):
            final_img = img.copy()
            if settings.get("use_hsv"):
                final_img = apply_hsv_enhancements(final_img)
            if settings.get("use_erasure"):
                final_img = apply_random_erasing(final_img)

            suffix = f"aug_{copy_idx:02d}"
            out_img_name = f"{base_name}_{suffix}.jpg"
            out_ann_name = f"{base_name}_{suffix}.txt"

            out_img_path = os.path.join(output_image_dir, out_img_name)
            out_ann_path = os.path.join(output_label_dir, out_ann_name)

            is_success, im_buf = cv2.imencode(".jpg", final_img)
            if is_success:
                im_buf.tofile(out_img_path)
                generated_count += 1
                if bboxes:
                    try:
                        with open(out_ann_path, 'w', encoding='utf-8') as f:
                            for bbox in bboxes:
                                f.write(" ".join(bbox) + "\n")
                    except Exception: pass

    progress_callback(generated_count)


# ==============================================================================
# 3. TKINTER GRAPHICAL INTERFACE
# ==============================================================================

class AugmentationApp:
    def __init__(self, master):
        self.master = master
        master.title("PCB YOLO Dataset Augmenter")
        master.geometry("680x560")

        # Variables
        self.input_img_dir = tk.StringVar(value="")
        self.input_lbl_dir = tk.StringVar(value="")
        self.output_img_dir = tk.StringVar(value="")
        self.output_lbl_dir = tk.StringVar(value="")

        self.use_rotation = tk.BooleanVar(value=True)
        self.rotation_angles_str = tk.StringVar(value="90, 180, 270")
        
        self.use_hsv = tk.BooleanVar(value=True)
        self.use_erasure = tk.BooleanVar(value=True)
        self.flat_copies = tk.IntVar(value=2)  # Non-rotated augmented copies

        self.is_augmenting = False
        self.processed_files = 0
        self.total_generated = 0
        self.total_files = 0

        self._create_widgets()

    def _create_widgets(self):
        style = ttk.Style(self.master)
        style.theme_use('clam')

        # Folder selection frame
        folder_frame = ttk.LabelFrame(self.master, text="Paths Setup", padding="10")
        folder_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(folder_frame, text="Input Images Folder:").grid(row=0, column=0, sticky="w", pady=2)
        tk.Entry(folder_frame, textvariable=self.input_img_dir, width=50).grid(row=0, column=1, padx=5, pady=2)
        tk.Button(folder_frame, text="Browse", command=lambda: self.input_img_dir.set(filedialog.askdirectory())).grid(row=0, column=2, pady=2)

        tk.Label(folder_frame, text="Input Labels Folder:").grid(row=1, column=0, sticky="w", pady=2)
        tk.Entry(folder_frame, textvariable=self.input_lbl_dir, width=50).grid(row=1, column=1, padx=5, pady=2)
        tk.Button(folder_frame, text="Browse", command=lambda: self.input_lbl_dir.set(filedialog.askdirectory())).grid(row=1, column=2, pady=2)

        tk.Label(folder_frame, text="Output Images Folder:").grid(row=2, column=0, sticky="w", pady=2)
        tk.Entry(folder_frame, textvariable=self.output_img_dir, width=50).grid(row=2, column=1, padx=5, pady=2)
        tk.Button(folder_frame, text="Browse", command=lambda: self.output_img_dir.set(filedialog.askdirectory())).grid(row=2, column=2, pady=2)

        tk.Label(folder_frame, text="Output Labels Folder:").grid(row=3, column=0, sticky="w", pady=2)
        tk.Entry(folder_frame, textvariable=self.output_lbl_dir, width=50).grid(row=3, column=1, padx=5, pady=2)
        tk.Button(folder_frame, text="Browse", command=lambda: self.output_lbl_dir.set(filedialog.askdirectory())).grid(row=3, column=2, pady=2)

        # Augmentation Options Frame
        opt_frame = ttk.LabelFrame(self.master, text="Augmentation Configurations", padding="10")
        opt_frame.pack(fill="x", padx=10, pady=5)

        # 1. Rotation settings
        tk.Checkbutton(opt_frame, text="Apply Rotation Augmentation", variable=self.use_rotation).grid(row=0, column=0, sticky="w", columnspan=2)
        tk.Label(opt_frame, text="Angles (comma-separated):").grid(row=1, column=0, sticky="w", padx=20)
        tk.Entry(opt_frame, textvariable=self.rotation_angles_str, width=30).grid(row=1, column=1, sticky="w")

        # 2. Color HSV and Random Erasing
        tk.Checkbutton(opt_frame, text="Apply HSV Enhancement (Brightness, Saturation, Contrast)", variable=self.use_hsv).grid(row=2, column=0, sticky="w", columnspan=2, pady=(10, 0))
        tk.Checkbutton(opt_frame, text="Apply Random Erasing (Cutout)", variable=self.use_erasure).grid(row=3, column=0, sticky="w", columnspan=2)

        # 3. Straight copies configuration
        tk.Label(opt_frame, text="Flat Augmented Copies per Image (No rotation):").grid(row=4, column=0, sticky="w", pady=(10, 0))
        tk.Entry(opt_frame, textvariable=self.flat_copies, width=10).grid(row=4, column=1, sticky="w", pady=(10, 0))

        # Progress / Console Frame
        progress_frame = ttk.LabelFrame(self.master, text="Execution Progress", padding="10")
        progress_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", mode="determinate")
        self.progress_bar.pack(fill="x", pady=5)

        self.status_label = tk.Label(progress_frame, text="Status: Ready", font=("Arial", 10, "bold"))
        self.status_label.pack(pady=2)

        # Text Console for Logging
        self.console = tk.Text(progress_frame, height=8, bg="black", fg="lime", font=("Consolas", 9))
        self.console.pack(fill="both", expand=True, pady=5)

        # Bottom Button Frame
        btn_frame = ttk.Frame(self.master, padding="5")
        btn_frame.pack(fill="x")

        self.start_btn = tk.Button(btn_frame, text="Start Augmentation", command=self.start_augmentation, bg="green", fg="white", font=("Arial", 11, "bold"), height=2)
        self.start_btn.pack(side="right", padx=10)

    def log(self, message):
        self.console.insert("end", message + "\n")
        self.console.see("end")

    def start_augmentation(self):
        if self.is_augmenting:
            return

        # Path validations
        img_in = self.input_img_dir.get().strip()
        lbl_in = self.input_lbl_dir.get().strip()
        img_out = self.output_img_dir.get().strip()
        lbl_out = self.output_lbl_dir.get().strip()

        if not (img_in and lbl_in and img_out and lbl_out):
            messagebox.showerror("Error", "All folder paths are required.")
            return

        if not (os.path.exists(img_in) and os.path.exists(lbl_in)):
            messagebox.showerror("Error", "Input directories do not exist.")
            return

        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        # Config extraction
        settings = {
            "use_rotation": self.use_rotation.get(),
            "use_hsv": self.use_hsv.get(),
            "use_erasure": self.use_erasure.get(),
            "flat_copies": self.flat_copies.get(),
            "angles": []
        }

        if settings["use_rotation"]:
            try:
                angles_raw = self.rotation_angles_str.get().split(",")
                settings["angles"] = [float(a.strip()) for a in angles_raw if a.strip()]
            except ValueError:
                messagebox.showerror("Error", "Angles list must be numbers separated by commas.")
                return

        # Scan files
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
        self.image_files = [f for f in os.listdir(img_in) if f.lower().endswith(valid_extensions)]
        self.total_files = len(self.image_files)

        if self.total_files == 0:
            messagebox.showwarning("Warning", "No valid image files found in Input Images folder.")
            return

        # Prepare UI
        self.is_augmenting = True
        self.start_btn.config(state="disabled")
        self.processed_files = 0
        self.total_generated = 0
        self.progress_bar["maximum"] = self.total_files
        self.progress_bar["value"] = 0
        self.console.delete(1.0, "end")

        self.log(f"[INFO] Found {self.total_files} images to process.")
        self.log(f"[INFO] Augmentation Settings: {settings}")
        self.status_label.config(text="Status: Processing...", fg="blue")

        # Run process in background thread
        threading.Thread(target=self.run_process, args=(img_in, lbl_in, img_out, lbl_out, settings), daemon=True).start()

    def run_process(self, img_in, lbl_in, img_out, lbl_out, settings):
        lock = threading.Lock()

        def update_progress(generated):
            with lock:
                self.processed_files += 1
                self.total_generated += generated
                
                # Update UI in main thread safely
                self.master.after(0, self.update_ui_progress)

        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            futures = [
                executor.submit(process_single_file, filename, img_in, lbl_in, img_out, lbl_out, settings, update_progress)
                for filename in self.image_files
            ]
            concurrent.futures.wait(futures)

        # Complete signal
        self.master.after(0, self.complete_augmentation)

    def update_ui_progress(self):
        self.progress_bar["value"] = self.processed_files
        self.status_label.config(text=f"Processed: {self.processed_files}/{self.total_files} | Total Generated: {self.total_generated}")
        if self.processed_files % 10 == 0 or self.processed_files == self.total_files:
            self.log(f"Processed file {self.processed_files}/{self.total_files}...")

    def complete_augmentation(self):
        self.is_augmenting = False
        self.start_btn.config(state="normal")
        self.status_label.config(text="Status: Augmentation Completed!", fg="green")
        self.log(f"\n[SUCCESS] Augmented successfully!")
        self.log(f"[SUCCESS] Total files processed: {self.processed_files}")
        self.log(f"[SUCCESS] Total augmented images & labels created: {self.total_generated}")
        messagebox.showinfo("Success", f"Augmentation Completed!\nTotal generated files: {self.total_generated}")


if __name__ == "__main__":
    root = tk.Tk()
    app = AugmentationApp(root)
    root.mainloop()
