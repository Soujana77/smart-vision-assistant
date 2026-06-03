from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np

from src.detection.yolo_detector import detect_objects
from src.ocr.ocr_reader import read_text
from src.navigation.navigator import analyze_path

app = FastAPI(
    title="Smart Vision Assistant API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",
    "http://localhost:5174"
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():

    return {
        "message": "Smart Vision Assistant API is running"
    }

@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    contents = await file.read()

    image_array = np.frombuffer(
        contents,
        np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    detections = detect_objects(frame)

    return {
        "detections": detections
    }

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):

    contents = await file.read()

    image_array = np.frombuffer(
        contents,
        np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    extracted_text = read_text(frame)

    return {
        "text": extracted_text
    }

@app.post("/navigation")
async def navigation(file: UploadFile = File(...)):

    contents = await file.read()

    image_array = np.frombuffer(
        contents,
        np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    detections = detect_objects(frame)

    guidance = analyze_path(detections)

    return {
        "guidance": guidance,
        "detections": detections
    }