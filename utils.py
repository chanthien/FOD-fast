# utils.py
import psutil
import time
import cv2
import numpy as np
import subprocess
from ultralytics import YOLO
import logging
import sqlite3
import base64
import ffmpeg
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


def base64_to_cv2(base64_string):
    # Loại bỏ header của chuỗi base64 nếu có
    if 'data:image' in base64_string:
        base64_string = base64_string.split(',')[1]

    # Giải mã base64
    img_data = base64.b64decode(base64_string)

    # Chuyển đổi thành mảng numpy
    nparr = np.frombuffer(img_data, np.uint8)

    # Đọc hình ảnh bằng cv2
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return img

# def initialize_capture(input_source: str, input_path: str):
#     if input_source == "webcam":
#         cap = cv2.VideoCapture(0)
#     elif input_source == "ip_camera":
#         if not input_path:
#             raise ValueError("IP Camera URL is required")
#         cap = cv2.VideoCapture(input_path)
#     elif input_source == "video_url":
#         if not input_path:
#             raise ValueError("Video URL is required")
#
#         cap = cv2.VideoCapture(input_path)
#     else:
#         raise ValueError(f"Unknown input source: {input_source}")
#
#     if not cap or not cap.isOpened():
#         raise ValueError(f"Unable to open video source: {input_path}")
#
#     return cap
# def initialize_capture(input_source: str, input_path: str):
#     if input_source == "webcam":
#         cap = cv2.VideoCapture(0)
#     elif input_source in ["ip_camera", "video_url"]:
#         if not input_path:
#             raise ValueError("IP Camera URL or Video URL is required")
#
#         # Sử dụng FFmpeg để lấy luồng video
#         ffmpeg_command = [
#             'ffmpeg',
#             '-i', input_path,
#             '-f', 'rawvideo',
#             '-pix_fmt', 'bgr24',
#             '-'
#         ]
#         process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         return process
#     else:
#         raise ValueError(f"Unknown input source: {input_source}")
#
#
# def read_frame_from_ffmpeg(process):
#     width = 640  # Chiều rộng của khung hình
#     height = 480  # Chiều cao của khung hình
#     frame_size = width * height * 3  # Kích thước của khung hình (3 kênh màu)
#
#     raw_frame = process.stdout.read(frame_size)
#     if len(raw_frame) != frame_size:
#         return None
#
#     frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
#     return frame

def initialize_capture(input_source: str, input_path: str):
    if input_source == "webcam":
        cap = cv2.VideoCapture(0)
    elif input_source == "ip_camera":
        if not input_path:
            raise ValueError("IP Camera URL is required")
        cap = _process_ip_camera(input_path)
    elif input_source == "video_url":
        if not input_path:
            raise ValueError("Video URL is required")
        cap = _process_video_url(input_path)
    else:
        raise ValueError(f"Unknown input source: {input_source}")

    return cap

def _process_ip_camera(ip_address: str):
    # Use ffmpeg-python to process the IP camera stream
    ffmpeg_cmd = f"ffmpeg -i {ip_address} -c:v libx264 -crf 18 -c:a aac -b:a 128k -f flv pipe:1"
    proc = subprocess.Popen(ffmpeg_cmd, shell=True, stdout=subprocess.PIPE)
    cap = cv2.VideoCapture(proc.stdout)
    return cap

def _process_video_url(video_url: str):
    # Use ffmpeg-python to process the video URL stream
    ffmpeg_cmd = f"ffmpeg -i {video_url} -c:v libx264 -crf 18 -c:a aac -b:a 128k -f flv pipe:1"
    proc = subprocess.Popen(ffmpeg_cmd, shell=True, stdout=subprocess.PIPE)
    cap = cv2.VideoCapture(proc.stdout)
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

class SystemResourceMonitor:
    def __init__(self):
        self.start_time = time.time()

    def get_gpu_usage(self):
        return "GPU usage information"
    def get_cpu_usage(self):
        return psutil.cpu_percent(interval=1)

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