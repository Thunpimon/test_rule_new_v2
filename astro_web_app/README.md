# Astro Image Quality Inspection Lab (Web Application)

A standalone, self-contained astronomy image defect inspection dashboard.
Integrates **Physics-Based Multi-Track Astronomy Rules (V2)** with **Deep Learning (EfficientNet-B0 ONNX)** for real-time telescope diagnostic consensus.

---

## 🚀 How to Run

1. Open your terminal in this directory (`d:\Internship\Test_Rule\test_rule_new_v2\astro_web_app`):
   ```bash
   python server.py
   ```
   *(Optionally specify port: `python server.py --port 8080`)*

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

---

## 🌟 Key Features

1. **Dual-Consensus Inspection**:
   - Compares **Physics Rule-Based V2 Score** against **CNN ONNX Deep Learning Confidence** side-by-side.
   - Highlights **Consensus (MATCH)** in green and flags **Disagreements** in amber.
2. **Batch Drag & Drop**:
   - Drop 1 to 100+ astronomy images (PNG, JPG, WebP) at once.
   - Asynchronous real-time batch queue with individual progress cards.
3. **Interactive Star & Physics Metrics Modal**:
   - Click on any thumbnail to open the high-resolution inspection modal.
   - Smooth **Zoom & Pan** (mouse wheel + drag) to inspect individual star shapes.
   - Live **MaxIm DL Metrics Panel**:
     - Star Count, FWHM (median), Eccentricity ($e$), Aspect Ratio, Circularity, Hollowness Ratio ($H$), Satellite Streak Length, CCD Blooming Spike diff.
   - 6-Class comparative probability bars.
4. **Session Reporting**:
   - Download the full batch session inspection results as a `.csv` spreadsheet in one click.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Web dashboard UI |
| `GET` | `/api/health` | Health check & model status |
| `POST` | `/api/analyze` | Multipart image upload for quality diagnosis |

### Example API Request (Python):
```python
import requests

url = "http://127.0.0.1:5000/api/analyze"
with open("star_frame.png", "rb") as f:
    response = requests.post(url, files={"file": f})
print(response.json())
```
