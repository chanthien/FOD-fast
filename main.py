# main.py
import logging
from fastapi import FastAPI, Request, WebSocket
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
import cv2
from utils import load_model, detect_objects, compute_perspective_transform, transform_to_gps
import asyncio
import json
import websockets
from datetime import datetime
from sahi import AutoDetectionModel
import sqlite3
from typing import List, Dict
from ResourceMonitor import ResourceMonitor
import time
# to run api: uvicorn main:app --reload


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

TELEGRAM_TOKEN = "your_telegram_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"
# Mount static files and templates
app.mount("/static", StaticFiles(directory="E:/fast-claude\static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load the YOLOv8 model

sahi_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path= 'models/yolov8-fod01.pt',
    confidence_threshold=0.5,
    device='cuda:0' #or "cpu"
)

# Define GPS points and image points for perspective transform
# gps_points = [
#     (longitude1, latitude1),
#     (longitude2, latitude2),
#     (longitude3, latitude3),
#     (longitude4, latitude4)
# ]
# image_points = [
#     (x1, y1),
#     (x2, y2),
#     (x3, y3),
#     (x4, y4)
# ]
# Compute perspective transform matrix
perspective_matrix = compute_perspective_transform(
    [(105.843583, 21.004964), (105.844614, 21.005134), ( 105.844700, 21.004923 ), (105.843556, 21.005169)],
    [(0, 0), (1200, 0), (1200, 720), (0, 720)]
)
# Compute perspective transform matrix
# transform_matrix = compute_perspective_transform(gps_points, image_points)

# Database setup
conn = sqlite3.connect('detections.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS detections
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              timestamp TEXT,
              object_name TEXT,
              latitude REAL,
              longitude REAL,
              confidence REAL)''')
conn.commit()

# Global variables for storing the latest frame and detections
latest_frame = None
latest_detections = []


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/process")

async def process_video(request: Request):
    start_time = time.time()
    global latest_frame, latest_detections
    form_data = await request.form()
    input_source = form_data.get("input_source")

    # Initialize video capture based on input source
    if input_source == "webcam":
        cap = cv2.VideoCapture(0)
    elif input_source == "ip_camera":
        cap = cv2.VideoCapture(form_data.get("ip_address"))
    elif input_source == "video_url":
        cap = cv2.VideoCapture(form_data.get("video_url"))
    elif input_source == "mp4_file":
        cap = cv2.VideoCapture(form_data.get("file_path"))
    else:
        return {"error": "Invalid input source"}

    async def process_frames():
        global latest_frame, latest_detections

        while True:

            ret, frame = cap.read()
            if not ret:
                break

            latest_frame = frame

            # Perform object detection using SAHI

            detections = detect_objects(sahi_model, frame)
            # detections = result
            system_usage = ResourceMonitor( interval=1)
            system_usage.print_usage()
            # Transform detections to GPS coordinates
            gps_detections = transform_to_gps(detections, perspective_matrix)

            # Save detections to database
            save_detections_to_db(gps_detections)
            latest_detections = gps_detections

            try:
                # Your asynchronous code here
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"An exception occurred: {e}")    # Add a small delay to prevent excessive CPU usage


    # Start processing frames in the background
    asyncio.create_task(process_frames())
    logger.info("Processing frames in the background")
    end_time = time.time()
    logger.info(f"Processing time: {end_time - start_time}")
    return templates.TemplateResponse("results.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Send latest detections to the client
            await websocket.send_json({"detections": latest_detections})
            logger.info("Sent latest detections to client")
            await asyncio.sleep(3)  # Update every 3 seconds
    except websockets.exceptions.ConnectionClosedError:
        logger.info("WebSocket disconnected")

    save_detections_to_db
def save_detections_to_db(detections: List[Dict]):
    timestamp = datetime.now().isoformat()
    for det in detections:
        c.execute('''INSERT INTO detections 
                     (timestamp, object_name, latitude, longitude, confidence)
                     VALUES (?, ?, ?, ?, ?)''',
                  (timestamp, det['name'], det['lat'], det['lon'], det['confidence']))
    conn.commit()


@app.get("/video_feed")
async def video_feed():
    async def generate():
        while True:
            if latest_frame is not None:
                _, buffer = cv2.imencode('.jpg', latest_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            await asyncio.sleep(0.1)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)