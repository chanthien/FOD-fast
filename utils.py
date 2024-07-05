# utils.py
import psutil
import time
import cv2
import numpy as np
from ultralytics import YOLO
import logging
import sqlite3
from datetime import datetime
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
from typing import List
from fastapi import WebSocket
# from telegram import Bot, Update
# from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# BOT_TOKEN = "AnUme123bot"
# bot = Bot(token=BOT_TOKEN)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




sahi_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path='models/yolov8n.pt',
    confidence_threshold=0.3,
    device='cuda:0'  # or "cpu"
)


def load_model(model_path):
    model = YOLO(model_path)
    return model

def draw_bounding_boxes(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det['name']} {det['confidence']:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

def detect_objects(sahi_model, frame):
    result = get_sliced_prediction(
        frame,
        sahi_model,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    detections = []
    for object_prediction in result.object_prediction_list:
        bbox = object_prediction.bbox.to_xyxy()
        category = object_prediction.category
        score = object_prediction.score.value

        detections.append({
            'id': None,
            'name': category.name,
            'bbox': bbox,
            'confidence': score
        })

    # logger.info(f"Detections: {detections}")
    return detections

def compute_perspective_transform(gps_points, image_points):
    matrix, _ = cv2.findHomography(np.array(image_points), np.array(gps_points))
    return matrix


def transform_to_gps(detections, matrix):
    transformed_detections = []
    for det in detections:
        # get central of bounding box
        center_x = (det['bbox'][0] + det['bbox'][2]) / 2
        center_y = (det['bbox'][1] + det['bbox'][3]) / 2

        points = np.array([[[center_x, center_y]]], dtype='float32')
        gps_points = cv2.perspectiveTransform(points, matrix)[0][0]

        transformed_detections.append({
            'id': det['id'],
            'name': det['name'],
            'lat': float(gps_points[1]),
            'lon': float(gps_points[0]),
            'confidence': det['confidence']
        })
    logger.info(f"Detections: {transformed_detections}")
    return transformed_detections


def initialize_capture(input_source: str, input_path: str):
    if input_source == "webcam":
        cap = cv2.VideoCapture(0)
    elif input_source == "ip_camera":
        if not input_path:
            raise ValueError("IP Camera URL is required")
        cap = cv2.VideoCapture(input_path)
    elif input_source == "video_url":
        if not input_path:
            raise ValueError("Video URL is required")

        cap = cv2.VideoCapture(input_path)
    else:
        raise ValueError(f"Unknown input source: {input_source}")

    if not cap or not cap.isOpened():
        raise ValueError(f"Unable to open video source: {input_path}")

    return cap


def save_detections_to_db(detections):
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    try:
        for det in detections:
            c.execute('''INSERT INTO detections 
                         (timestamp, object_name, latitude, longitude, confidence)
                         VALUES (?, ?, ?, ?, ?)''',
                      (timestamp, det['name'], det['lat'], det['lon'], det['confidence']))

        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Database error: {str(e)}")
        conn.rollback()
    finally:
        conn.close()

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

    async def close_all(self):
        for connection in self.active_connections:
            await connection.close()
        self.active_connections.clear()

# System resource monitor
def get_cpu_usage():
    return psutil.cpu_percent(interval=1)


def get_gpu_usage():
    return "GPU usage information"


class SystemResourceMonitor:
    def __init__(self):
        self.start_time = time.time()

    def get_memory_usage(self):
        memory = psutil.virtual_memory()
        return memory.used/(1024*1024)

    def calculate_fps(self, num_frames):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        fps = num_frames / elapsed_time
        return fps

    def get_processing_time(self):
        end_time = time.time()
        processing_time = end_time - self.start_time
        return processing_time