# 🌀 Cyclone Detection & Impact Analysis System

> **Predict the storm. Map the damage. Before it happens.**

A machine-learning-driven prototype for automated cyclone intensity classification and geographical impact estimation. Powered by a fine-tuned **ResNet-18** deep learning model, this system reads satellite imagery, predicts cyclone intensity, and estimates the human and infrastructural cost of the storm — all before the winds pick up.

---

## ✨ Features

| Feature | What it does |
|---|---|
| 🛰️ **Satellite Image Intensity Classification** | A fine-tuned PyTorch ResNet-18 model predicts cyclone intensity categories directly from satellite images |
| 📍 **Geographical & Coordinate-Based Analysis** | Computes impact zones, radii of maximum winds, and regional exposure around any input coordinates |
| 🏚️ **Impact & Affected-Area Estimation** | Estimates affected land area, infrastructure risk, and population exposure based on predicted intensity |
| 📊 **Interactive Frontend Dashboard** | A web dashboard (`index.html`) for real-time input, prediction visualization, and impact mapping |
| ⚡ **Flask REST API** | Backend endpoints for image uploads, coordinate telemetry, batch evaluation, and JSON payloads |

---

## 🗂️ Project Structure

```
.
├── model.py                 # PyTorch ResNet-18 architecture, preprocessing, and training routines
├── risk_assessment.py       # Geographical impact modeling, population exposure, and risk logic
├── test_app.py              # Flask API backend serving predictions and spatial analysis
├── index.html               # Frontend dashboard UI for image upload, mapping, and metric display
├── requirements.txt         # Python dependencies
├── cyclone_resnet18.pth     # Trained PyTorch model weights (local artifact)
└── README.md                # Project documentation
```

---

## 🧩 Key Components

- **`model.py`** — The brains 🧠. Defines the `CycloneResNet18` architecture, data transforms (resizing, normalization, augmentation), dataset loaders, and training/validation loops.
- **`risk_assessment.py`** — The math behind the map 🗺️. Deterministic algorithms that turn a predicted intensity + coordinates into storm radius, population exposure, and risk categories.
- **`test_app.py`** — The messenger 📡. A Flask server that handles requests, runs inference, and talks to the frontend.
- **`index.html`** — The face of the operation 🎛️. Upload imagery, set coordinates, and watch the predictions and impact charts come to life.

---

## 🚀 Installation & Requirements

### System Requirements

- 🐍 **Python**: 3.9 to 3.11
- 🎮 **GPU / CUDA**: Recommended for training (CUDA 11.8+ or 12.x) — CPU inference works fine too

### Setup Instructions

**1️⃣ Clone the Repository**

```bash
git clone <repository-url>
cd <repository-directory>
```

**2️⃣ Create and Activate a Virtual Environment**

```bash
python -m venv venv
# On Linux / macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

**3️⃣ Install Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🏋️ Generating the Cyclone ResNet-18 Model Weights

The core prediction engine needs `cyclone_resnet18.pth` sitting in the project root. No checkpoint, no predictions — here's how to make one.

1. **Prepare Training Data** — Arrange satellite image sets into class directories per the loader schema in `model.py`.
2. **Execute Model Training**:

```bash
python model.py
```

> ⏱️ **Heads up**: Training deep learning models is computationally intensive. A CUDA-compatible GPU is strongly recommended — your laptop fan will thank you.

3. **Verify Output Checkpoint**:

```bash
ls -lh cyclone_resnet18.pth
```

---

## 🔍 Model Usage

The model classifies cyclone intensity from normalized satellite imagery, step by step:

```
Satellite Image (224x224 RGB)
              ↓
  Torchvision Transformations (Resize, Normalize)
              ↓
   ResNet-18 Backbone (Feature Extractor)
              ↓
 Fully Connected Classification Layer
              ↓
 Predicted Intensity Class & Confidence Score
```

### Loading and Inference Flow

- **Input**: Single or batch 3-channel RGB satellite images, resized to 224×224 and normalized with standard ImageNet mean/std.
- **Weights**: Loaded dynamically via `torch.load('cyclone_resnet18.pth', map_location=device)` into the `CycloneResNet18` instance.
- **Inference Pipeline**: `test_app.py` parses the uploaded image, converts it to a tensor, runs the forward pass, and extracts the top prediction + class probabilities.
- **Downstream Integration**: The predicted intensity flows straight into `risk_assessment.py`, alongside coordinates, to compute impact radius and risk stats.

---

## ▶️ Running the Application

### 1. Fire Up the Flask Backend

```bash
python test_app.py
```

🌐 Default Port: [http://127.0.0.1:5000](http://127.0.0.1:5000) (or `http://localhost:5000`)

The server loads model weights onto available hardware (GPU/CPU) and stands by for incoming HTTP POST requests.

### 2. Launch the Frontend Interface

```bash
# Using Python's built-in HTTP server:
python -m http.server 8000
```

Then head to `http://localhost:8000` in your browser and watch it come alive.

---

## 📡 Communication

The frontend sends async `multipart/form-data` requests (image + coordinates) to the Flask backend. In return, it gets a JSON payload packed with:

- 🌪️ Cyclone classification category
- 📈 Prediction confidence percentage
- 🎯 Estimated impact radius and zone parameters
- 📋 Risk assessment summary

---

## 🔄 Prototype Workflow

An end-to-end pipeline, from raw pixels to actionable risk map:

```
+--------------------------+
| User Image & Coordinates |
+--------------------------+
             |
             v
+--------------------------+
|  Flask Backend Service   |
+--------------------------+
             |
             v
+--------------------------+
|   ResNet-18 Classifier   |  --> Cyclone Intensity Prediction
+--------------------------+
             |
             v
+--------------------------+
|  Risk & Spatial Engine   |  --> Impact Radius & Exposure Calculations
+--------------------------+
             |
             v
+--------------------------+
| Dashboard Visualizations |  --> Mapped Risk Zones & Statistical Output
+--------------------------+
```

---

## ⚠️ Limitations

- **Prototype Status** — This is a proof-of-concept research demo, not certified for operational meteorological deployment.
- **Model Generalization** — Predictions live and die by the training dataset. Different satellite sensor bands or image quality will shift accuracy.
- **Impact Estimations** — Affected-area and population exposure figures are algorithmic approximations, not official disaster assessments.
- **No Real-Time Telemetry Feed** — Runs on manually supplied images/coordinates, not live calibrated Doppler radar or satellite feeds.
- 🚨 **Official Warning Notice** — Do **not** use this software for emergency management, evacuation orders, or safety-critical decisions. Always defer to national meteorological agencies (IMD, NOAA, JTWC).

---

## 🔮 Future Improvements

- 🛰️ **Live Satellite Pipeline** — Direct integration with real-time telemetry APIs (INSAT, GOES, Himawari)
- 🌡️ **Multi-Modal Architecture** — Fold in atmospheric variables (sea surface temperature, central pressure, wind shear) alongside imagery
- 🧠 **Advanced Neural Architectures** — Benchmark Vision Transformers (ViT) and larger backbones (ConvNeXt, EfficientNet)
- 🗺️ **High-Resolution GIS Integration** — Dynamic census data, topographical elevation maps, and real-time storm-surge modeling
- 🔔 **Automated Early Warning Alerts** — Push-notification infrastructure for geo-fenced stakeholder warnings

---

*Built for research and prototyping — not for saving lives (yet). Stay safe, stay informed, trust the pros at IMD/NOAA/JTWC.* 🌊
