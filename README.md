# Computer-Vision-project
# YOLOv8-TensorRT

# 🚀 Real-time Object Detection with TensorRT

High-performance Computer Vision system using **YOLOv8 + ONNX + TensorRT** for low-latency inference.

---

## 📌 Overview

This project builds an end-to-end inference pipeline:

```
PyTorch (.pt) → ONNX → TensorRT Engine → Real-time Inference
```

* Optimized for **low latency**
* Supports **GPU acceleration (CUDA)**
* Designed for **real-time applications**

---

## ⚙️ Setup

### 1. Install dependencies

```bash


install CUDA 


pip install TensorRT 

pip install ultralytics
```

### 2. Requirements

* CUDA >= 11.x
* TensorRT >= 8.x

---

## 🔧 Usage

### 1. Export ONNX

```bash
python onnx.py --weights yolov8s.pt
```

---

### 2. build TensorRT Engine

```bash
python engine.py --weights yolov8s.onnx
```

---

### 3. Run Inference

```bash
python infer_trt.py --engine yolov8s.engine --source data/
```

---

## ⚡ Performance

* Low-latency inference (~ms level)
* GPU-accelerated using TensorRT
* Optimized with FP16
## ⚡ Performance

| Method        | Latency (ms) | FPS  |
|--------------|-------------|------|
| PyTorch      | 28 ms       | 40   |
| onnx         | 15 ms        | 80|
| TensorRT FP16| 8 ms        | 120  |

Tested on RTX 3050, batch size = 1

---

## 📁 Project Structure

```
.
├── export_onnx.py
├── build_engine.py
├── infer_trt.py
├── models/
└── utils/
```

---

## 🧠 Tech Stack

* Python
* YOLOv8
* ONNX
* TensorRT
* CUDA

