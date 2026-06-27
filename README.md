# Face Anti‑Spoofing System

Real‑time face liveness detection using MobileNetV2 + texture + glare.

## Features

- Offline – no internet required
- Runs on standard CPU – no GPU needed
- ~50 ms per frame (~20 FPS) on Intel Core i5
- Web interface with live video and real‑time feedback

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/rathishakarthikeyan-cmd/Face_spoof_detection.git
cd Face_spoof_detection
```

### 2. Create and activate virtual environment

```bash
python -m venv venv_new
.\venv_new\Scripts\Activate.ps1   # Windows PowerShell
# OR
source venv_new/bin/activate      # Linux/Mac
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

### 5. Open your browser

Navigate to: http://127.0.0.1:5000
Grant camera permission. The system will start detecting faces and classifying them as REAL or SPOOF.

