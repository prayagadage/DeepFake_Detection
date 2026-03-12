<h1 align="center">🔍 DeepFake Detection</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/Accuracy-~85%25-success?style=for-the-badge"/>
</p>

<p align="center">
  A deep learning system to detect DeepFake videos using a hybrid <strong>CNN + RNN</strong> architecture — extracting spatial face features frame-by-frame and classifying videos as <strong>REAL</strong> or <strong>FAKE</strong> with ~85% accuracy.
</p>

---

## 📚 Table of Contents

- [What is DeepFake?](#-what-is-deepfake)
- [Impact of DeepFakes](#-impact-of-deepfakes)
- [Project Objectives](#-project-objectives)
- [Project Pipeline](#-project-pipeline)
- [Workflow](#-workflow)
- [Models & Architecture](#-models--architecture)
- [Getting Started](#-getting-started)
- [Technologies Used](#-technologies-used)
- [Conclusion](#-conclusion)

---

## 🤔 What is DeepFake?

DeepFakes are synthetic media — images or videos manipulated using AI to replace a person's likeness with someone else's. They are typically generated using **Generative Adversarial Networks (GANs)**, which are becoming more sophisticated and harder to detect over time.

> *"As DeepFake technology improves, so must our ability to detect it."*

---

## ⚠️ Impact of DeepFakes

- 📰 **Misinformation** — Fake news, misleading political content, fabricated celebrity videos
- 💸 **Financial Fraud** — Impersonation of executives to authorize fraudulent transactions
- 😰 **Mental Harm** — False rumours causing unrest and psychological distress
- 🎬 **Industry Threat** — Film, media, and social platforms are actively combating DeepFakes

---

## 🎯 Project Objectives

1. **Detect** whether a given video is **REAL or FAKE** using deep learning
2. **Analyze** video frames to identify subtle facial imperfections introduced by DeepFake algorithms
3. **Deploy** a usable web application that accepts video uploads and returns predictions in real-time

> **Goal:** Build a robust model that learns the distinguishing features between authentic and face-swapped DeepFake frames.

---

## 🔄 Project Pipeline

| Step | Description |
|------|-------------|
| **1** | 📂 Load the dataset (DFDC / FaceForensics++) |
| **2** | 🎬 Extract individual video clips |
| **3** | 🖼️ Extract all frames from each video (real & fake) |
| **4** | 👤 Detect and crop the face sub-frame from each image |
| **5** | 📍 Locate facial landmarks using Dlib |
| **6** | 📊 Perform frame-by-frame analysis for changes in facial landmarks |
| **7** | ✅ Classify the video as **REAL** or **FAKE** |

---

## 🔧 Workflow

### Pre-Processing

```
Raw Video
   ↓
Frame Extraction
   ↓
Face Detection (Dlib / OpenCV)
   ↓
Face Cropping & Landmark Detection
   ↓
Normalized Image Frames → Model Input
```

### Prediction

```
Video Upload
   ↓
Frame-by-Frame Feature Extraction (CNN)
   ↓
Temporal Sequence Modelling (GRU / RNN)
   ↓
Aggregate Predictions Across Frames
   ↓
Output: REAL ✅ or FAKE ❌
```

---

## 🧠 Models & Architecture

### CNN-Only Models

| Model | Notes |
|-------|-------|
| **MesoNet** | Pre-trained for DeepFake images; less effective on video frames |
| **ResNet50V2** | Trained on cropped DeepFake frames with ImageNet weights |
| **EfficientNetB0** | Trained on cropped video frames with ImageNet pre-training |

> Standalone CNN models showed limited accuracy — combining with sequential models improved performance significantly.

---

### CNN + RNN Hybrid Models ⭐

#### InceptionV3 + GRU
| Parameter | Value |
|-----------|-------|
| **Test Accuracy** | ~82% |
| **Optimizer** | Adam |
| **Loss Function** | `sparse_categorical_crossentropy` |
| **Feature Extractor** | InceptionV3 (per-frame vectors) |
| **Classifier** | GRU (temporal sequence) |
| **Limitation** | Struggles with multiple faces per frame |

#### EfficientNetB2 + GRU ✅ *(Best Model)*
| Parameter | Value |
|-----------|-------|
| **Test Accuracy** | ~85% |
| **Optimizer** | Adam |
| **Loss Function** | `sparse_categorical_crossentropy` |
| **Feature Extractor** | EfficientNetB2 (per-frame vectors) |
| **Classifier** | GRU (temporal sequence) |
| **Limitation** | Reduced accuracy on low-light / dark background videos |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- GPU recommended (CUDA-compatible)
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/prayagadage/DeepFake_Detection.git
cd DeepFake_Detection
```

```bash
# Install dependencies
pip install -r Deploy/requirments.txt
```

### Running the App

```bash
# Navigate to the Deploy folder
cd Deploy

# Start the Flask app
python app.py
```

> 💡 **Note:** Results for a 10-second, 30fps video are typically returned within ~1 minute. GPU acceleration is strongly recommended.

Then open your browser and go to `http://localhost:5000` to upload a video and get a prediction.

---

## 🛠️ Technologies Used

<p align="left">
  <a href="https://www.python.org" target="_blank"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/></a>&nbsp;&nbsp;
  <a href="https://www.tensorflow.org" target="_blank"><img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/></a>&nbsp;&nbsp;
  <a href="https://opencv.org/" target="_blank"><img src="https://www.vectorlogo.zone/logos/opencv/opencv-icon.svg" alt="opencv" width="40" height="40"/></a>&nbsp;&nbsp;
  <a href="https://pandas.pydata.org/" target="_blank"><img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/></a>&nbsp;&nbsp;
  <a href="https://scikit-learn.org/" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/></a>&nbsp;&nbsp;
  <a href="https://seaborn.pydata.org/" target="_blank"><img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="seaborn" width="40" height="40"/></a>&nbsp;&nbsp;
  <a href="https://www.w3.org/html/" target="_blank"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/html5/html5-original-wordmark.svg" alt="html5" width="40" height="40"/></a>&nbsp;&nbsp;
  <a href="https://www.w3schools.com/css/" target="_blank"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/css3/css3-original-wordmark.svg" alt="css3" width="40" height="40"/></a>
</p>

| Technology | Purpose |
|------------|---------|
| **Python** | Core programming language |
| **TensorFlow / Keras** | Model training and inference |
| **OpenCV** | Video & frame processing |
| **Dlib** | Face detection & landmark extraction |
| **Flask** | Web application backend |
| **HTML/CSS** | Frontend interface |
| **Pandas / NumPy** | Data preprocessing |

---

## 📝 Conclusion

This project successfully demonstrates DeepFake video detection using a **hybrid CNN + RNN** approach:

- ✅ **Standalone CNN models** (ResNet, EfficientNet) were tested but showed limited performance on video-level classification
- ✅ **CNN + GRU models** leverage both spatial (per-frame) and temporal (across frames) features, dramatically improving accuracy
- ✅ **Best model** — `EfficientNetB2 + GRU` — achieved **~85% test accuracy** on the DFDC sample dataset
- ✅ The model is optimised for **high precision on FAKE videos**, reducing the risk of DeepFakes slipping through undetected

### Future Improvements
- 🔄 Multi-face detection support
- 🌙 Better performance in low-light conditions
- 📱 Mobile-optimised lightweight model (ONNX / TFLite)
- 🌐 Real-time browser-based inference

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/prayagadage"><strong>Prayag Adage</strong></a>
</p>
