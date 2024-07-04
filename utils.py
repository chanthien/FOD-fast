# utils.py

import cv2
import numpy as np
from ultralytics import YOLO
import logging
from sahi.predict import get_sliced_prediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path):
    model = YOLO(model_path)
    return model


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