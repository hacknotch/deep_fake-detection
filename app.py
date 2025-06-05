from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import shutil
import gdown

app = Flask(__name__)

# Constants
UPLOAD_FOLDER = 'uploads'
WATERMARKED_FOLDER = 'watermarked'
MODEL_FOLDER = 'models'
MODEL_PATH = os.path.join(MODEL_FOLDER, 'mesonet_model.h5')
MODEL_DRIVE_ID = '1X7RgfjG38M_Uqonwmr3cFq-Wl1SkLaIr'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(WATERMARKED_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Download model from Google Drive if missing
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?export=download&id={MODEL_DRIVE_ID}", MODEL_PATH, quiet=False)

# Load the model
model = load_model(MODEL_PATH)

def add_watermark(frame):
    watermarked = frame.copy()
    height, width = watermarked.shape[:2]
    overlay_height = height // 6
    overlay_width = width // 3
    overlay = np.zeros((overlay_height, overlay_width, 3), dtype=np.uint8)
    cv2.rectangle(overlay, (0, 0), (overlay_width, overlay_height), (0, 0, 0), -1)
    y_offset = height - overlay_height
    x_offset = 0
    roi = watermarked[y_offset:y_offset + overlay_height, x_offset:x_offset + overlay_width]
    cv2.addWeighted(overlay, 0.7, roi, 0.3, 0, roi)
    text = "DEEPFAKE DETECTED"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = x_offset + 10
    text_y = y_offset + (overlay_height + text_size[1]) // 2
    cv2.putText(watermarked, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 1)
    cv2.putText(watermarked, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)
    return watermarked

def process_video_with_watermark(input_path, output_path):
    try:
        shutil.copy2(input_path, output_path)
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        raise

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_skip = 30
    frame_count = 0
    fake_count = 0
    real_count = 0
    results = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                resized = cv2.resize(frame, (256, 256))
                norm = resized / 255.0
                input_tensor = np.expand_dims(norm, axis=0)
                pred = model.predict(input_tensor, verbose=0)
                fake_score = pred[0][0]
                label = "Fake" if fake_score > 0.5 else "Real"
                fake_count += label == "Fake"
                real_count += label == "Real"
                results.append({
                    "frame": frame_count,
                    "label": label,
                    "score": float(fake_score)
                })
            frame_count += 1
    finally:
        cap.release()

    final_result = "FAKE" if fake_count > real_count else "REAL"
    return {
        "frame_results": results,
        "total_fake_frames": fake_count,
        "total_real_frames": real_count,
        "final_decision": final_result
    }

@app.route('/')
def home():
    return render_template('frontend.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(upload_path)

    try:
        results = analyze_video(upload_path)
        if results['final_decision'] == 'FAKE':
            output_name = f"watermarked_{filename}"
            output_path = os.path.join(WATERMARKED_FOLDER, output_name)
            process_video_with_watermark(upload_path, output_path)
            results['watermarked_path'] = output_name
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(upload_path):
            os.remove(upload_path)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(
        os.path.join(WATERMARKED_FOLDER, filename),
        as_attachment=True,
        mimetype='video/mp4'
    )

if __name__ == '__main__':
    app.run(debug=True)
