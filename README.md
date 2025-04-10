# Helmet Detection System

A real-time safety monitoring application that detects whether people are wearing helmets in images and live video streams. Built with Python, Flask, YOLO, OpenCV, and Bootstrap.

## Features

- **Image Upload Detection**: Upload images to detect people with and without helmets
- **Real-time Webcam Detection**: Use your camera for live helmet detection monitoring
- **Advanced Visualization**: Clear visual indicators for detection results with color-coded bounding boxes
- **Detailed Statistics**: Real-time counts of people with and without helmets
- **Adjustable Parameters**: Customize detection confidence thresholds
- **Responsive Design**: Modern Bootstrap UI that works on desktop and mobile devices

## Technologies Used

- **Backend**:
  - Python 3.8+
  - Flask (Web Framework)
  - OpenCV (Computer Vision)
  - YOLOv8 (Object Detection)
  - NumPy, Pillow (Image Processing)
  
- **Frontend**:
  - HTML5, CSS3, JavaScript
  - Bootstrap 5.3
  - Font Awesome Icons

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Web camera (for live detection)
- Git (optional)

### Setup Instructions

1. **Clone the repository or download the source code**:
   ```bash
   git clone https://github.com/Scodeme17/Helmet-detection/.git
   cd Helmet-detection
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the YOLO model weights**:
   - Place your trained YOLOv8 helmet detection model in the `weights` folder
   - Ensure the model path in `app.py` points to the correct location:
     ```python
     MODEL_PATH = "weights/best.pt"  # Update this path if needed
     ```

5. **Run the application**:
   ```bash
   python app.py
   ```

6. **Open your web browser** and navigate to:
   ```
   http://localhost:5000
   ```

## Usage Guide

### Image Upload Detection

1. Open the application in your web browser
2. Go to the "Upload Image" tab (default)
3. Either drag and drop an image or click to browse files
4. Adjust the confidence threshold if needed (default is 0.4)
5. Click "Detect Helmets"
6. Review the results in the right panel with detection counts


### Settings

1. Go to the "Settings" tab to adjust global parameters
2. Change the default confidence threshold
3. Adjust webcam processing rate for performance optimization
4. View model information and device being used for inference

## Project Structure

```
helmet-detection-system/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── weights/                  # YOLO model weights
│   └── best.pt               # Trained helmet detection model
├── static/                   # Static assets
│   └── uploads/              # Uploaded and processed images
└── templates/                # HTML templates
    └── index.html            # Main application interface
```

