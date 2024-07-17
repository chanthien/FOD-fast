import asyncio
import signal
import logging

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse

from utils import initialize_capture, ConnectionManager, SystemResourceMonitor, base64_to_cv2
from video_processor import VideoProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

video_processor = VideoProcessor()
manager = ConnectionManager()
resource_monitor = SystemResourceMonitor()

@app.on_event("startup")
async def startup_event():
    logger.info("Application is starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application is shutting down...")
    await manager.close_all()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/process")

async def process_video(request: Request):
    form_data = await request.form()
    input_source = form_data.get("input_source")
    input_path = form_data.get("input_path")

    logger.info(f"Form data received: input_source={input_source}, input_path={input_path}")

    try:
        cap = initialize_capture(input_source, input_path)
        logger.info(f"Capture initialized for {input_source}")
    except Exception as e:
        logger.error(f"Error initializing capture: {str(e)}", exc_info=True)
        return HTMLResponse(content=f"Error initializing capture: {str(e)}", status_code=500)

    await asyncio.create_task(video_processor.process_frames(cap))
    logger.info("process_frames task created")

    return templates.TemplateResponse("results.html", {"request": request})

@app.post("/sprocess")
async def sprocess_video(request: Request):
    form_data = await request.form()
    processed_data = form_data.get("processed_data")
    cap = base64_to_cv2(processed_data)
    await asyncio.create_task(video_processor.process_frames(cap))


    return templates.TemplateResponse("results.html", {"request": request})




@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            if video_processor.latest_detections:
                await manager.broadcast(video_processor.latest_detections)
                # await websocket.send_json(video_processor.latest_detections)
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

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    uvicorn.run(app, host="0.0.0.0", port=8000)