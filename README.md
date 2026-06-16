# elbow-angle-tracker
# 🦾 Elbow Angle Tracker

Real-time elbow joint angle detection using **pure OpenCV** — no deep learning required.

Detects the elbow from a webcam feed using skin color segmentation, extracts the arm skeleton, and outputs a continuous **score value**:

| Elbow Position | Angle | Score |
|----------------|-------|-------|
| 90° (reference) | 90° | `0` |
| Flexed (bent more) | > 90° | `positive` |
| Extended (straighter) | < 90° | `negative` |

---

## 📸 How It Works

```
Webcam Frame
    ↓
Skin color detection (HSV mask)
    ↓
Largest contour → arm region
    ↓
Skeleton fitting (5 axis points)
    ↓
Elbow = point with max deviation from straight line
    ↓
Angle between 3 points → Score = angle − 90
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Webcam

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/elbow-angle-tracker.git
cd elbow-angle-tracker
pip install -r requirements.txt
```

### Run

```bash
python elbow_tracker.py
```

### Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `R` | Reset smoothing |

---

## 📦 Dependencies

```
opencv-python
numpy
```

---

## 💡 Tips for Best Results

- Wear a **short-sleeve** shirt
- Use a **plain background** (avoid skin-colored backgrounds)
- Ensure **good lighting**
- Keep your **full arm visible** in the frame

---

## 🔧 Configuration

Inside `elbow_tracker.py` you can tune:

```python
REFERENCE_ANGLE = 90.0   # angle that maps to score 0
SMOOTHING_ALPHA = 0.25   # temporal smoothing (0=none, 1=full)
MIN_CONTOUR_AREA = 3000  # minimum arm region size (pixels)
```

---

## 📌 Limitations & Future Improvements

- Works best with consistent skin tone and lighting
- For higher accuracy, consider switching to **MediaPipe Pose** (provides shoulder, elbow, and wrist landmarks directly)
- Could be extended to track **both arms** simultaneously

---

## 📄 License

MIT License — free to use and modify.
