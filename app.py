# app.py - Enhanced Flask Backend for Helmet Detection
import os
import time
import base64
import uuid
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO
import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model configuration
model = None
class_names = {0: 'Without Helmet', 1: 'With Helmet'}
colors = {0: (255, 0, 0), 1: (0, 255, 0)}  # Red for without helmet, green for with helmet

# Define your model path directly
MODEL_PATH = "/home/sohan/Desktop/ML/helmet-detection-app/weights/best.pt"

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(image_path, confidence=0.4):
    """Process image for helmet detection"""
    # Run inference
    results = model.predict(image_path, conf=confidence)
    result = results[0]
    
    # Load and process the image for display
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    # Draw bounding boxes
    annotated_img = img.copy()
    
    # Detection data
    detection_data = []
    
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        
        for box, cls, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = box
            label = f"{class_names[cls]} {conf:.2f}"
            color = colors[cls]
            
            # Draw bounding box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Add detection to data
            detection_data.append({
                'class_id': int(cls),
                'class_name': class_names[cls],
                'confidence': float(conf),
                'box': [int(x) for x in [x1, y1, x2, y2]]
            })
    
    # Convert the image to BGR for saving
    annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
    
    return annotated_img, annotated_img_bgr, detection_data

def process_image_from_array(img_array, confidence=0.4):
    """Process image from numpy array for helmet detection"""
    # Create a temporary file
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{uuid.uuid4()}.jpg")
    cv2.imwrite(temp_path, img_array)
    
    # Process the image
    annotated_img, annotated_img_bgr, detection_data = process_image(temp_path, confidence)
    
    # Remove the temporary file
    try:
        os.remove(temp_path)
    except:
        pass
    
    return annotated_img, annotated_img_bgr, detection_data

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Detect helmets in uploaded image"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate a unique filename
        filename = f"{uuid.uuid4()}.{file.filename.rsplit('.', 1)[1].lower()}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        try:
            confidence = float(request.form.get('confidence', 0.4))
            _, output_img_bgr, detection_data = process_image(filepath, confidence)
            
            # Save the output image
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"result_{filename}")
            cv2.imwrite(output_path, output_img_bgr)
            
            # Count detections by class
            class_counts = {}
            for cls in class_names:
                class_counts[cls] = sum(1 for d in detection_data if d['class_id'] == cls)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'result_filename': f"result_{filename}",
                'detections': detection_data,
                'class_counts': class_counts
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/detect_frame', methods=['POST'])
def detect_frame():
    """Detect helmets in a base64 encoded frame"""
    try:
        # Get base64 data
        data = request.get_json()
        
        if not data or 'frame' not in data:
            return jsonify({'error': 'No frame data provided'}), 400
            
        try:
            base64_image = data.get('frame', '').split(',')[1]
        except IndexError:
            # If no comma in the string, just use the whole string
            base64_image = data.get('frame', '')
            
        confidence = float(data.get('confidence', 0.4))
        
        # Decode base64 to image
        img_bytes = base64.b64decode(base64_image)
        img = Image.open(BytesIO(img_bytes))
        img_array = np.array(img)
        
        # Convert RGB to BGR for OpenCV (if needed)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Process the image
        _, output_img_bgr, detection_data = process_image_from_array(img_array, confidence)
        
        # Convert back to base64
        _, buffer = cv2.imencode('.jpg', output_img_bgr)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        # Count detections by class
        class_counts = {}
        for cls in class_names:
            class_counts[cls] = sum(1 for d in detection_data if d['class_id'] == cls)
        
        return jsonify({
            'success': True,
            'image': f"data:image/jpeg;base64,{img_str}",
            'detections': detection_data,
            'class_counts': class_counts
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/get_stats', methods=['GET'])
def get_stats():
    """Get model statistics"""
    try:
        # Return model information
        model_path = model.ckpt_path
        model_info = {
            'model_path': model_path,
            'classes': list(class_names.values()),
            'device': 'GPU' if torch.cuda.is_available() else 'CPU'
        }
        return jsonify(model_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
            
        print(f"Loading model from: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        print(f"Model loaded successfully on {'GPU' if torch.cuda.is_available() else 'CPU'}")
        
        # Start the Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Error: {str(e)}")