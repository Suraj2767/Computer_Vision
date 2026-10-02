# Computer Vision Pipeline

## Objective

This project implements a real-time computer vision pipeline using
YOLO object detection, object tracking, centroid calculation,
bounding-box size estimation, FPS measurement, and inference latency
measurement.

The target object used in this project is a person.

## Technologies Used

- Python
- OpenCV
- Ultralytics YOLO
- NumPy

## Pipeline

Input Video
    ↓
YOLO Object Detection
    ↓
Object Tracking
    ↓
Bounding Box
    ↓
Centroid Calculation
    ↓
Bounding Box Area
    ↓
FPS + Latency Measurement
    ↓
Annotated Output Video

## Features

### 1. Object Detection

YOLO11n is used to detect persons in every video frame.

### 2. Object Tracking

YOLO tracking assigns a unique ID to detected persons and maintains
their identity across consecutive frames.

### 3. Bounding Box

Each detected person is represented using:

(x1, y1, x2, y2)

where the coordinates represent the bounding-box corners.

### 4. Centroid

The center of each bounding box is calculated using:

cx = (x1 + x2) / 2

cy = (y1 + y2) / 2

### 5. Size / Distance Heuristic

Bounding-box area is calculated as:

Area = Width × Height

A larger bounding box generally indicates that the object appears
closer to the camera, while a smaller bounding box generally
indicates that it appears farther away.

This is a relative heuristic and not an actual metric distance
measurement.

### 6. Latency

Inference latency is measured for every processed frame in
milliseconds.

### 7. FPS

FPS is estimated from the measured inference latency:

FPS = 1000 / latency_ms

## Project Structure

```text
Computer_Vision/
│
├── cv_pipeline.py
├── input.mp4
├── output_annotated.mp4
├── latency.csv
├── yolo11n.pt
├── README.md
└── venv/

<img width="306" height="394" alt="image" src="https://github.com/user-attachments/assets/2fbb688a-fde2-4966-b10f-4adb9641b809" />




