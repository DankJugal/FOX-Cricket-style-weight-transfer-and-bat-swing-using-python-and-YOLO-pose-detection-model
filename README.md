# FOX-Cricket inspired Weight transfer and batswing predictor

## Overview

This python project provides detailed insights into two fundamental aspects of cricket batting performance: **weight transfer analysis** and **bat swing speed measurement**. The application processes video inputs and generates annotated output videos with intuitive visual overlays, making complex biomechanical data accessible and actionable.

---
## Video Comparison

### Weight Transfer Analysis: Virat Kohli

| Input Video | Output Video |
|-------------|--------------|
| <video src="videos/virat_kohli.mp4" controls width="320"></video> | <video src="videos/virat_kohli_weight_transfer.webm" controls width="320"></video> |

---

### Bat Swing Speed Analysis: Rohit Sharma

| Input Video | Output Video |
|-------------|--------------|
| <video src="videos/rohit_sharma.mp4" controls width="320"></video> | <video src="videos/bat_swing_speed_rohit_sharma.webm" controls width="320"></video> |


## Key Features

- **Real-time Pose Detection**: Uses YOLOv11n-pose model to detect 17 keypoints on the human body
- **Weight Transfer Analysis**: Calculates and visualizes front-to-back weight distribution during batting
- **Bat Swing Speed Measurement**: Measures linear and angular velocity of bat movement in km/h
- **Smooth Data Processing**: Implements moving average and exponential moving average (EMA) filtering for noise reduction
- **Professional UI Overlays**: Dark-themed, real-time visualization panels with smooth transitions
- **Slow Motion Support**: Optional frame interpolation for detailed slow-motion analysis
- **Skeleton Visualization**: Full-body pose skeleton rendering with enhanced visual clarity

---

## Technology Stack

| Category | Technology |
|----------|-----------|
| **Computer Vision** | OpenCV, YOLOv11 (Ultralytics) |
| **Pose Detection** | YOLOv11n-pose (17-keypoint model) |
| **Numerical Computing** | NumPy |
| **Data Structures** | Python Collections (deque) |
| **Math & Physics** | Standard Python math library |
| **Video Processing** | OpenCV VideoCapture & VideoWriter |
| **Language** | Python 3.7+ |
| **Framework** | Ultralytics YOLOv11 |

### Dependencies

```bash
pip install ultralytics opencv-python numpy
```

---

## Project Structure

```
cricket-analysis/
│
├── weight_transfer.py          # Weight transfer analysis module
├── bat_swing.py                # Bat swing speed analysis module
├── videos/                     # Input video directory
│   ├── virat_kohli.mp4
│   ├── ishan_kishan.mp4
│   ├── virat_kohli_weight_transfer.mp4 (output)
│   └── bat_swing_speed_ishan_kishan.mp4 (output)
│
├── models/                     # Pre-trained models
│   └── yolo11n-pose.pt        # YOLOv11 Nano pose detection model
│
└── README.md                   # Project documentation
```

---

## Core Algorithms & Formulas

### 1. Weight Transfer Analysis (`weight_transfer.py`)

#### Center of Mass (COM) Calculation

The center of mass is computed as a weighted sum of body part positions:

\[
\text{COM}_x = \sum (\text{keypoint}_x \times \text{weight}_i)
\]

\[
\text{COM}_y = \sum (\text{keypoint}_y \times \text{weight}_i)
\]

**Body Part Weights:**
- Shoulders (keypoints 5, 6): 0.075 each
- Elbows (keypoints 7, 8): 0.05 each
- Wrists (keypoints 9, 10): 0.025 each
- Hips (keypoints 11, 12): 0.20 each
- Knees (keypoints 13, 14): 0.10 each
- Ankles (keypoints 15, 16): 0.05 each

#### Front-Back Weight Distribution

**Foot Detection:** Determines which ankle is the front foot based on relative positioning:
- Horizontal orientation: Compare x-coordinates of ankles
- Vertical orientation: Compare y-coordinates of ankles

**Distance Calculation:**
\[
\text{front\_distance} = |\text{COM} - \text{front\_foot}|
\]

\[
\text{back\_distance} = |\text{COM} - \text{back\_foot}|
\]

**Weight Distribution:**
\[
\text{back\_weight}\% = \frac{\text{front\_distance}}{\text{front\_distance} + \text{back\_distance}} \times 100
\]

\[
\text{front\_weight}\% = 100 - \text{back\_weight}\%
\]

#### Smoothing with Moving Average & EMA

**Moving Average (15-frame window):**
\[
\text{MA}_{\text{front}} = \frac{1}{n} \sum_{i=0}^{n-1} \text{front\_weight}_i
\]

**Exponential Moving Average (EMA with α = 0.15):**
\[
\text{EMA}_{\text{new}} = \alpha \times \text{MA}_{\text{current}} + (1 - \alpha) \times \text{EMA}_{\text{previous}}
\]

Where \(\alpha = 0.15\) provides smooth transitions with rapid responsiveness.

---

### 2. Bat Swing Speed Analysis (`bat_swing.py`)

#### Linear Velocity Calculation

**Wrist Displacement:**
\[
\text{distance}_{\text{left}} = \sqrt{(\Delta x_{\text{left}})^2 + (\Delta y_{\text{left}})^2}
\]

\[
\text{distance}_{\text{right}} = \sqrt{(\Delta x_{\text{right}})^2 + (\Delta y_{\text{right}})^2}
\]

**Average Linear Speed (pixels/frame):**
\[
\text{linear\_speed} = \frac{\text{distance}_{\text{left}} + \text{distance}_{\text{right}}}{2 \times \text{frame\_time}}
\]

#### Angular Velocity Calculation

**Bat Angle:**
\[
\text{bat\_angle} = \arctan2(\Delta y, \Delta x) \times \frac{180}{\pi}
\]

Where \(\Delta x\) and \(\Delta y\) are differences between left and right wrist positions.

**Angular Velocity (degrees/second):**
\[
\text{angular\_velocity} = \frac{|\Delta \text{angle}|}{{\text{frame\_time}}}
\]

Angle wrapping is applied to handle discontinuities at ±180°:
\[
\text{if } \Delta\text{angle} > 180: \Delta\text{angle} -= 360
\]

#### Pixel-to-Real-World Conversion

**Kilometers per Hour:**
\[
\text{km/h} = \left(\frac{\text{pixel\_speed}}{\text{pixels\_per\_meter}}\right) \times 3.6
\]

Where:
- `pixel_speed` is in pixels per frame
- `pixels_per_meter` is the calibration constant (typically 50-100)
- 3.6 converts m/s to km/h

#### Speed Smoothing

Same EMA technique as weight transfer:
\[
\text{EMA}_{\text{speed}} = 0.15 \times \text{MA}_{\text{speed}} + 0.85 \times \text{EMA}_{\text{previous}}
\]

---

## Usage Guide

### Weight Transfer Analysis

```python
from weight_transfer import CricketWeightTransferAnalyzer

# Initialize analyzer
analyzer = CricketWeightTransferAnalyzer()

# Process video
analyzer.process_video(
    video_path="videos/virat_kohli.mp4",
    output_path="videos/virat_kohli_weight_transfer.mp4",
    slow_motion=False
)

# Optional: Enable slow motion (2x slowdown)
# analyzer.process_video(
#     video_path="videos/sample.mp4",
#     output_path="videos/sample_weight_slow.mp4",
#     slow_motion=True
# )
```

### Bat Swing Speed Analysis

```python
from bat_swing import CricketBatSpeedAnalyzer

# Initialize analyzer
analyzer = CricketBatSpeedAnalyzer()

# Process video with calibration
analyzer.process_video(
    video_path="videos/ishan_kishan.mp4",
    output_path="videos/bat_swing_speed_ishan_kishan.mp4",
    slow_motion=False,
    pixels_per_meter=50  # Calibration parameter
)

# Optional: Adjust calibration based on camera setup
# analyzer.process_video(
#     video_path="videos/sample.mp4",
#     output_path="videos/sample_bat_speed.mp4",
#     slow_motion=True,
#     pixels_per_meter=100
# )
```

### Calibration Guide

The `pixels_per_meter` parameter adjusts the conversion from pixel measurements to real-world speed:
- **Lower values** (50-70): Closer camera, more sensitive to motion
- **Higher values** (100-150): Distant camera, less sensitive to motion

To calibrate accurately, measure a known distance (e.g., 1 meter) in your video frame and count pixels.

---

## Output Visualization

### Weight Transfer Panel

The output video displays a sophisticated UI panel at the bottom showing:

- **Center Position Indicator**: Fixed vertical line at 50% representing neutral weight distribution
- **Back Weight Bar** (Blue): Percentage of weight on back foot (0-100%)
- **Forward Weight Bar** (Orange): Percentage of weight on front foot (0-100%)
- **Real-time Percentage Values**: Updated EMA-smoothed percentages
- **Dynamic Color Coding**: Visual feedback for weight distribution state

**Interpretation:**
- **50/50 Split**: Balanced stance
- **70% Back**: Player transitioning from back foot drive
- **70% Forward**: Player completing forward shot execution

### Bat Swing Speed Panel

The output video displays comprehensive swing metrics:

- **Current Speed** (km/h): Real-time EMA-smoothed bat swing velocity
- **Speed Bar Graph**: Visual representation with color-coded zones:
  - Green Zone (0-72 km/h): Gentle movements
  - Yellow Zone (72-108 km/h): Moderate swing
  - Orange Zone (108-144 km/h): Strong swing
  - Red Zone (144+ km/h): Peak power swing
- **Peak Speed**: Maximum swing speed recorded during video
- **Rotation Rate** (°/s): Angular velocity of bat rotation

**Interpretation:**
- **Peak Speed >140 km/h**: Professional-grade power batting
- **Consistent 100+ km/h**: Strong technical execution
- **High Rotation Rate**: Aggressive swing mechanics

---

## Visualization Features

### Skeleton Rendering

- **Full Body Pose**: 17-keypoint skeleton covering shoulders, elbows, wrists, hips, knees, and ankles
- **Transparency Layering**: Semi-transparent overlays for depth perception
- **Connection Lines**: Anatomically accurate skeletal connections with 3px white lines
- **Joint Circles**: 5px white circles with 6px gray outline at each keypoint

### UI Design

- **Dark Theme**: Professional dark background (RGB: 8,8,10) for video content
- **High Contrast**: Light text (RGB: 240,240,245) for readability
- **Semi-transparent Panels**: 92% opacity base panel with gradient darkening
- **Smooth Animations**: EMA-based smooth transitions between values
- **Professional Fonts**: HERSHEY_DUPLEX with appropriate scaling

---

## Performance Metrics

- **Model**: YOLOv11n-pose (Nano variant - optimized for speed)
- **Keypoints Detected**: 17 anatomical points per frame
- **Processing Speed**: Real-time processing at original video FPS
- **Output Format**: MP4 (H.264 codec via 'mp4v')
- **Supported Resolutions**: All standard video resolutions (720p, 1080p, 4K, etc.)

---

## Advanced Configuration

### Smoothing Parameters

```python
# In weight_transfer.py or bat_swing.py
self.ema_alpha = 0.15  # Control EMA responsiveness
                        # Lower = smoother, Higher = more responsive
self.weight_history = deque(maxlen=5)  # Moving average window
                                        # Adjust for different smoothing
```

### Performance Optimization

- Use `yolo11n-pose.pt` for real-time processing on CPU/GPU
- Enable `slow_motion=False` for faster initial processing
- Reduce input video resolution for faster analysis
- Process shorter clips (< 5 minutes) for quick validation

---

## Input & Output Examples

### Input Requirements

- **Format**: MP4, AVI, MOV, or any OpenCV-supported video format
- **Resolution**: Any standard resolution (720p minimum recommended)
- **Frame Rate**: 24-60 FPS (typical)
- **Content**: Full-body visible cricket batting footage

### Output Features

- **Weight Transfer Output**: Video with:
  - Skeleton overlay on batter
  - Real-time weight distribution UI
  - 15-frame moving average smoothing
  - EMA filtering (α=0.15)

- **Bat Swing Speed Output**: Video with:
  - Bat line visualization (between wrists)
  - Speed measurement UI
  - Peak speed tracking
  - Angular velocity display
  - Calibrated measurements (km/h)

---

## Installation & Setup

```bash
# Clone or download the project
cd cricket-analysis

# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install ultralytics opencv-python numpy

# Download YOLOv11 model (automatic on first run)
# Or pre-download:
# yolo detect predict model=yolo11n-pose.pt

# Place your cricket videos in the videos/ directory
# Run analysis
python weight_transfer.py
python bat_swing.py
```

### requirements.txt

```
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
```

---