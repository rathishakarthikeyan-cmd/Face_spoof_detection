from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import threading

app = Flask(__name__)
model = load_model("model/model.h5")

# DNN Face Detector
prototxt = "face_detection_files/deploy.prototxt"
caffemodel = "face_detection_files/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

status_lock = threading.Lock()
latest_status = {
    "label": "Detecting...",
    "confidence": 0.0,
    "glare_percent": 0.0,
    "texture": 0.0,
    "motion": 0.0,
    "suggestion": "Align your face"
}

def update_status(new_status):
    global latest_status
    with status_lock:
        latest_status = new_status

def glare_detection(face_roi):
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    overexposed = np.sum(gray > 200)
    total = gray.size
    return (overexposed / total) * 100

def texture_variance(face_roi):
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (200, 200))
    return cv2.Laplacian(gray_resized, cv2.CV_64F).var()

def preprocess_face(face):
    face = cv2.resize(face, (224, 224))
    face = face / 255.0
    return face

def average_brightness(frame):
    """Return mean grayscale value of the whole frame (0-255)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def generate_suggestion(model_label, confidence, is_textured, has_glare,
                        motion_score, frame_brightness,
                        TEXTURE_THRESH, GLARE_THRESH):
    """Return a user-friendly suggestion string."""
    if frame_brightness < 50:
        return "Improve lighting (room is too dark)"
    elif frame_brightness > 220:
        return "Reduce overexposure (move away from direct light)"
    elif not is_textured and model_label == "SPOOF":
        return "Move closer to camera or ensure face is clearly visible"
    elif has_glare and model_label == "REAL":
        return "Glare detected – try to avoid bright reflections (e.g., from glasses or windows)"
    elif confidence < 0.6 and confidence > 0.4:
        return "Hold steady and look directly at the camera"
    elif model_label == "REAL" and is_textured and not has_glare:
        return "Authentic face detected"
    elif model_label == "SPOOF":
        return "Possible spoof detected (photo or screen)"
    else:
        return "Align your face and ensure good lighting"

def generate_frames():
    camera = cv2.VideoCapture(0)
    pred_buffer = deque(maxlen=30)
    decision_buffer = deque(maxlen=5)
    prev_gray_frame = None

    TEXTURE_THRESH = 20    # adjust as needed
    GLARE_THRESH = 5.0     # increased to avoid false reject (user can tune)

    frame_count = 0
    while True:
        success, frame = camera.read()
        if not success:
            break
        frame_count += 1

        # Motion detection (for display and suggestion)
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_resized = cv2.resize(current_gray, (320, 240))
        motion_score = 0.0
        if prev_gray_frame is not None:
            diff = cv2.absdiff(current_resized, prev_gray_frame)
            motion_score = np.mean(diff)
        prev_gray_frame = current_resized

        # DNN face detection
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        current_status = {
            "label": "No face",
            "confidence": 0.0,
            "glare_percent": 0.0,
            "texture": 0.0,
            "motion": motion_score,
            "suggestion": "Show your face"
        }

        # No face detected -> use default suggestion
        face_detected = False
        for i in range(0, detections.shape[2]):
            detection_conf = detections[0, 0, i, 2]
            if detection_conf > 0.5:
                face_detected = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                x = max(0, x - 10)
                y = max(0, y - 10)
                x2 = min(w, x2 + 10)
                y2 = min(h, y2 + 10)

                face_roi = frame[y:y2, x:x2]
                if face_roi.size == 0:
                    continue

                # Model prediction
                proc_face = preprocess_face(face_roi)
                input_face = np.reshape(proc_face, (1, 224, 224, 3))
                pred = model.predict(input_face, verbose=0)[0][0]
                pred_buffer.append(pred)
                smoothed_pred = sum(pred_buffer) / len(pred_buffer)
                confidence = smoothed_pred if smoothed_pred > 0.5 else 1 - smoothed_pred
                model_label = "REAL" if smoothed_pred > 0.5 else "SPOOF"

                texture_score = texture_variance(face_roi)
                is_textured = texture_score > TEXTURE_THRESH

                glare_pct = glare_detection(face_roi)
                has_glare = glare_pct > GLARE_THRESH

                # ---- Fusion decision (same as your working version) ----
                if model_label == "REAL" and is_textured and not has_glare:
                    final_label = "REAL"
                else:
                    final_label = "SPOOF"

                # Majority voting
                decision_buffer.append(final_label)
                if len(decision_buffer) == 5:
                    real_votes = decision_buffer.count("REAL")
                    spoof_votes = decision_buffer.count("SPOOF")
                    if real_votes > spoof_votes:
                        final_label = "REAL"
                    elif spoof_votes > real_votes:
                        final_label = "SPOOF"

                # Generate user‑friendly suggestion
                avg_bright = average_brightness(frame)
                user_suggestion = generate_suggestion(
                    model_label, confidence, is_textured, has_glare,
                    motion_score, avg_bright,
                    TEXTURE_THRESH, GLARE_THRESH
                )

                color = (0, 255, 0) if final_label == "REAL" else (0, 0, 255)

                current_status = {
                    "label": final_label,
                    "confidence": float(round(confidence, 3)),
                    "glare_percent": float(round(glare_pct, 1)),
                    "texture": float(round(texture_score, 1)),
                    "motion": float(round(motion_score, 1)),
                    "suggestion": user_suggestion
                }

                cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
                cv2.putText(frame, f"{final_label} (conf:{confidence:.2f})",
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # If no face, set suggestion accordingly
        if not face_detected:
            avg_bright = average_brightness(frame)
            if avg_bright < 50:
                current_status["suggestion"] = "Improve lighting (room too dark)"
            elif avg_bright > 220:
                current_status["suggestion"] = "Reduce overexposure (light too bright)"
            else:
                current_status["suggestion"] = "Show your face"

        update_status(current_status)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_status')
def get_status():
    with status_lock:
        return jsonify(latest_status)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)