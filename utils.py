# utils.py

import cv2
import numpy as np
from ultralytics import YOLO
import logging
import sqlite3
from datetime import datetime
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




sahi_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path='models/yolov8-fod01.pt',
    confidence_threshold=0.5,
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


def initialize_capture(input_source, form_data):
    try:
        if input_source == "webcam":
            logger.info("Attempting to open webcam")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logger.error("Failed to open webcam")
                raise IOError("Unable to open webcam")
            logger.info("Webcam opened successfully")
        elif input_source == "ip_camera":
            cap = cv2.VideoCapture(form_data.get("ip_address"))
        elif input_source == "video_url":
            cap = cv2.VideoCapture(form_data.get("video_url"))
        elif input_source == "mp4_file":
            cap = cv2.VideoCapture(form_data.get("file_path"))
        elif input_source == "image":
            cap = cv2.imread(form_data.get("file_path"))
        else:
            raise ValueError("Invalid input source")

        if not cap.isOpened() and input_source != "image":
            raise IOError(f"Unable to open video source: {input_source}")

        return cap
    except Exception as e:
        logger.error(f"Error initializing capture: {str(e)}", exc_info=True)
        raise


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
                      # ( det['name'], det['lat'], det['lon'], det['confidence']))
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Database error: {str(e)}")
        conn.rollback()
    finally:
        conn.close()