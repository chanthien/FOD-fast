import asyncio
import signal
import logging
from typing import List

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse

from utils import initialize_capture, save_detections_to_db
from video_processor import VideoProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

video_processor = VideoProcessor()


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


manager = ConnectionManager()


@app.on_event("startup")
async def startup_event():
    # Any startup tasks can be added here
    logger.info("Application is starting up...")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application is shutting down...")
    await manager.close_all()
    # Add any other cleanup tasks here


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/process")
async def process_video(request: Request):
    form_data = await request.form()
    input_source = form_data.get("input_source")
    cap = initialize_capture(input_source, form_data)
    logger.info(f"Capture initialized for {input_source}")

    if input_source == "image":
        await video_processor.process_single_image(cap)
    else:
        asyncio.create_task(video_processor.process_frames(cap))
        logger.info("process_frames task created")

    return templates.TemplateResponse("results.html", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            if video_processor.latest_detections:
                await manager.broadcast(video_processor.latest_detections)
                logger.info(
                    f"Broadcast detections: {len(video_processor.latest_detections['video'])} video, {len(video_processor.latest_detections['gps'])} GPS")
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
        manager.disconnect(websocket)


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(video_processor.generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


def signal_handler(sig, frame):
    logger.info(f"Received signal {sig}. Shutting down gracefully...")
    asyncio.get_event_loop().stop()


if __name__ == "__main__":
    import uvicorn

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)