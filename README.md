# PGNN Pile Capacity Predictor — GUI

A lightweight desktop GUI (built with Python `tkinter`) for predicting the **ultimate bearing capacity (Q<sub>max</sub>)** of piles using an **Physics-Guided Neural Network (PGNN)** surrogate model trained on geotechnical site-investigation data.

This tool accompanies the research draft *"[A Physics-Guided Machine Learning Approach for Estimating the Uplift Capacity of 
Driven Piles: Bridging the Gap Between Accuracy and Physical Consistency ]"* and is provided for reproducibility, demonstration, and inference on new pile design cases.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research--prototype-orange)

---

## Overview

Given seven geotechnical and geometric design parameters, the trained PGNN model estimates the ultimate pile capacity in kilonewtons (kN). The GUI wraps the trained PyTorch model and its input/output scalers in a simple form-based interface, so predictions can be generated without writing any code.

| Input | Symbol | Description | Unit |
|---|---|---|---|
| Unit weight of soil | γt | Total/bulk unit weight | kN/m³ |
| Effective vertical stress | σv' | In-situ effective overburden stress | kPa |
| Friction angle | ϕ | Soil internal friction angle | ° |
| SPT blow count | N_spt | Standard Penetration Test resistance | – |
| Pile embedment length | L | Length of pile in the ground | m |
| Pile diameter | D | Pile shaft diameter | m |
| Shaft surface area | A_s | Total lateral shaft surface area | m² |

**Output:** Predicted ultimate pile capacity, Q<sub>max</sub> (kN).

---

## Repository Structure

```
.
├── pile_capacity_gui.py     # Main GUI application
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── model/                    # (not tracked) trained model + metadata files
    ├── Adaptive_PINN_Model_<timestamp>.pth
    └── Adaptive_PINN_Metadata_<timestamp>.joblib
```

> **Note:** The trained model (`.pth`) and metadata (`.joblib`) files are not included in this repository due to size/licensing. See [Model Files](#model-files) below.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

`tkinter` ships with most standard Python installations. On some Linux distributions it must be installed separately:
```bash
sudo apt-get install python3-tk
```

---

## Model Files

The GUI expects a trained PyTorch model and a corresponding metadata file (feature order + fitted scalers), produced by the Bayesian-Optimization training pipeline described in the paper. These files are available on resonable request.

1. Place the two files in a folder of your choice.
2. Open `pile_capacity_gui.py` and update the path configuration near the top:
   ```python
   TIMESTAMP = "20260629_213135"
   BASE_DIR = r"D:\AHmad\BO PINN"
   ```
   to match your local file locations and timestamp.

If the model files are not found, the app automatically falls back to a **Demo/Mock Mode**, using an approximate shaft-friction + end-bearing formula so the interface remains fully functional for demonstration purposes.

---

## Usage

Run the application:
```bash
python pile_capacity_gui.py
```

1. Enter the seven input parameters (default values are pre-filled with representative averages).
2. Click **Predict ultimate capacity (Q_max)** to generate a prediction.
3. Click **Clear all** to reset the form.
4. Click **Save results** to capture a screenshot of the current prediction (saved as `gui_screenshot_<timestamp>.png` in the working directory).

The status bar at the bottom of the window indicates whether a trained model was successfully loaded or whether the app is running in demo mode.

---

## License

This project is released under the [MIT License](LICENSE).

---

## Contact

For questions regarding the model, dataset, or GUI, please open an issue in this repository or contact [Davidahmadazeem@gmail.com]
