# Smart Vision Assistant

## Overview

Smart Vision Assistant is an AI-powered assistive application designed to help visually impaired individuals understand and navigate their surroundings. The system combines object detection, optical character recognition (OCR), navigation guidance, and live camera analysis to provide real-time environmental awareness.

The project uses YOLOv8 for object detection, Tesseract OCR for text recognition, FastAPI for backend services, and React for the frontend interface.

---

## Features

### Object Detection

* Detects common objects using YOLOv8.
* Displays object name, confidence score, direction, and estimated distance.

### OCR (Optical Character Recognition)

* Extracts text from images and camera frames.
* Helps users read signs, labels, and printed content.

### Navigation Guidance

* Analyzes detected objects.
* Generates movement suggestions such as:

  * Move Left
  * Move Right
  * Stop
  * Path Clear

### Image Analysis

* Upload an image for AI analysis.
* Receive object detection, OCR results, and navigation guidance.

### Live Camera Mode

* Access webcam directly from the browser.
* Analyze current camera frame.
* Detect objects in real time.

---

## Technology Stack

### Frontend

* React
* Vite
* Axios
* CSS

### Backend

* FastAPI
* Python

### AI & Computer Vision

* YOLOv8 (Ultralytics)
* OpenCV
* Tesseract OCR

---

## Project Structure

```text
smart-vision-assistant/
│
├── backend/
│   ├── src/
│   │   ├── api/
│   │   ├── detection/
│   │   ├── navigation/
│   │   ├── ocr/
│   │   └── voice/
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── App.jsx
│   │   └── App.css
│
└── README.md
```

---

## Backend API Endpoints

### Health Check

```http
GET /
```

Returns application status.

### Object Detection

```http
POST /detect
```

Returns detected objects from an uploaded image.

### OCR

```http
POST /ocr
```

Extracts text from an uploaded image.

### Navigation

```http
POST /navigation
```

Returns navigation guidance based on detected obstacles.

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd smart-vision-assistant
```

### Backend Setup

```bash
cd backend

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
```

### Start Backend

```bash
python -m uvicorn src.api.app:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

Swagger Documentation:

```text
http://127.0.0.1:8000/docs
```

---

### Frontend Setup

```bash
cd frontend

npm install

npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

---

## How It Works

### Image Upload Workflow

```text
Upload Image
      ↓
FastAPI
      ↓
YOLOv8 Detection
      ↓
OCR Extraction
      ↓
Navigation Analysis
      ↓
Results Displayed
```

### Live Camera Workflow

```text
Webcam Feed
      ↓
Capture Frame
      ↓
Object Detection
      ↓
OCR Extraction
      ↓
Navigation Guidance
      ↓
Results Displayed
```

---

## Future Enhancements

* Automatic frame analysis
* Real-time voice announcements
* Mobile application support
* GPS integration
* Obstacle distance estimation using depth models
* Multilingual voice guidance
* Real-time navigation assistance

---

## Project Goal

The goal of Smart Vision Assistant is to create an affordable and accessible AI-powered vision aid that helps visually impaired individuals perceive their environment, recognize objects, read text, and navigate safely.

---

## Author

Developed as an AI-powered assistive technology project using Computer Vision, OCR, and Real-Time Web Technologies.
