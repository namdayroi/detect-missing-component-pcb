# PCB Missing Component Detection System (Hệ thống Phát hiện Linh kiện Thiếu trên PCB)

An advanced, AI-powered Automatic Optical Inspection (AOI) system using **YOLOv11** and **OpenCV** to identify missing, misplaced, or unexpected components on PCB assemblies in real-time.

---

## 📌 Project Overview (Tổng quan Dự án)

During PCB manufacturing and assembly, ensuring that all components (resistors, capacitors, ICs, connectors) are present and correctly placed is crucial. This project provides a complete end-to-end pipeline:
1. **Data Augmentation**: Enhancing the training dataset using color, rotation, and cutout augmentations.
2. **Model Training**: Training a YOLOv11 model on Colab using custom PCB datasets.
3. **Reference Mapping**: Interactive creation of a "Golden Board" reference map (JSON).
4. **AOI Inspection**: Real-time comparison of live camera feeds against the golden board reference using partial affine transformation (RANSAC-aligned) and geometric verification.

---

## 🛠 Project Structure (Cấu trúc Dự án)

The codebase has been refactored and consolidated into the following standard files:

*   **`main_gui.py`**: The production-ready desktop application written in **PyQt5**. Handles multithreaded video capture (Webcam or RTSP IP Camera), live statistics calculations (FPS, Pass Rate, Good/NG quantities), and overlays detection results.
*   **`inspector_engine.py`**: The core inspection engine in a standalone CLI format with OpenCV visual output.
*   **`create_reference_map.py`**: An interactive OpenCV/Tkinter tool to generate the golden reference JSON. Users can run YOLO to get suggested boxes, then click to select, drag to resize/move, or draw new boxes manually.
*   **`data_augmentation.py`**: A multithreaded Tkinter GUI utility to apply **Rotation** (updating YOLO label coordinates), **HSV Color shifts**, and **Random Erasing (Cutout)** to input datasets.
*   **`Training_model.ipynb`**: Google Colab notebook containing the pipeline for data splitting, auto-generating `data.yaml`, and training YOLOv11s.

---

## 🔬 Core Algorithms (Thuật toán Cốt lõi)

### 1. Board Alignment (Căn chỉnh Bo mạch)
Since the physical PCB board under the camera can shift, rotate, or scale, the system uses **Anchor Points** (unique landmarks such as Micro USB ports or ICs) to dynamically align the live frame with the golden reference map.
*   Matches anchors between YOLO detections and the reference JSON.
*   Computes a $2 \times 3$ **Partial Affine Transformation Matrix** using RANSAC:

$$\begin{bmatrix} x_{det} \\\\ y_{det} \end{bmatrix} = M \cdot \begin{bmatrix} x_{ref} \\\\ y_{ref} \\\\ 1 \end{bmatrix}$$

*   Aligns expected golden bboxes to current camera coordinate space.

### 2. Geometric Validation (Xác minh Hình học)
To prevent wrong component matching, the system validates detections mathematically using:
*   **Area Ratio**: The area of the detected component must match the expected area ($0.8 < \text{Ratio} < 1.25$).
*   **Aspect Ratio**: The width-to-height ratio must be consistent ($0.7 < \text{Ratio} < 1.4$).

### 3. Contextual Filtering (Bộ lọc Ngữ cảnh)
*   **Context Rescue**: If a component's YOLO confidence is low ($0.35 \le \text{Conf} < 0.6$) but it is located exactly where a reference component is missing, it is "rescued" (matched) rather than flagged as missing.
*   **Context Unexpected Filter**: Detections with confidence below $0.6$ that are far from any expected components are filtered out as background noise to prevent false positives.

---

## 🚀 Installation & Setup (Cài đặt & Thiết lập)

### Prerequisites (Yêu cầu hệ thống)
*   Python 3.8+
*   NVIDIA GPU with CUDA support (Recommended for real-time inspection)

### Installation (Cài đặt thư viện)
```bash
pip install ultralytics opencv-python numpy PyQt5 torch Pillow pyyaml
```

---

## 📖 How to Run (Hướng dẫn Sử dụng)

> [!IMPORTANT]
> **QUAN TRỌNG**: Bạn **BẮT BUỘC** phải chạy bước Tăng cường dữ liệu (Step 1) để chuẩn bị bộ dữ liệu mở rộng trước khi tiến hành Huấn luyện mô hình (Step 2). Việc này giúp tăng độ chính xác của mô hình YOLOv11 đối với bo mạch PCB thực tế.
> 
> **IMPORTANT**: You **MUST** run the Data Augmentation step (Step 1) to generate augmented training data before proceeding to Model Training (Step 2). This is critical for improving the accuracy of the YOLOv11 model on actual PCBs.

### Step 1: Augment your Dataset
Run `data_augmentation.py` to expand your training images:
```bash
python data_augmentation.py
```
*   Select your images and labels folders.
*   Configure rotation angles (e.g. `90, 180, 270`) and check HSV/Erasure.
*   Click **Start Augmentation** to run the multithreaded generation.

### Step 2: Train the YOLO model
1. Upload your dataset (zip format) to Google Drive.
2. Open `Training_model.ipynb` in Google Colab.
3. Run the cells sequentially to unzip, partition dataset, build `data.yaml`, and train the model using:
   ```bash
   yolo detect train data=data.yaml model=yolo11s.pt epochs=100 imgsz=640
   ```
4. Download the trained `best.pt` model weights.

### Step 3: Create the Golden Reference Map
Prepare your reference board image (or live camera stream) and run:
```bash
python create_reference_map.py
```
*   Input the paths to your trained `best.pt` model, `classes.txt`, and output JSON.
*   Use the interactive OpenCV interface to adjust bounding boxes:
    *   **Drag** a box to move.
    *   **Drag handles** (white circles) to resize.
    *   **Drag on empty space** to draw a new manual component box (type name in pop-up dialog).
    *   Press `d` to delete selected.
    *   Press `s` to save to JSON and exit.

### Step 4: Run Live Inspection
Launch the PyQt5 main GUI:
```bash
python main_gui.py
```
*   Input your YOLO Model path, Reference Map JSON, `classes.txt`, and RTSP IP Camera stream URL.
*   Click **Start Inspection**.
*   The system will automatically detect and overlay components:
    *   **Green**: Matched components (Present and correct).
    *   **Red**: Missing components.
    *   **Orange**: Unexpected / Extra components.
*   Use the **Recalculate Alignment (r)** button to calibrate the fixed alignment matrix if the camera shifts.

---

## 📊 Evaluation (Thống kê AOI)

The dashboard tracks production statistics dynamically:
*   **Pass Rate**: Calculated as:
    $$\text{Pass Rate} = \frac{\text{Matched} - \text{Missing} - \text{Unexpected}}{\text{Expected}} \times 100\%$$
*   **Good / NG Count**: Categorizes boards as Good (100% matched, 0 missing/unexpected) or NG (No Good) if defects are found.
