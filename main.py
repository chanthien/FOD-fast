import logging
from fastapi import FastAPI, Request, WebSocket
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
import asyncio
from utils import initialize_capture, save_detections_to_db
from video_processor import VideoProcessor

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

video_processor = VideoProcessor()

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
        logger.info("Starting process_frames")
        asyncio.create_task(video_processor.process_frames(cap))
        logger.info("process_frames task created")

    return templates.TemplateResponse("results.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            if video_processor.latest_detections:
                await websocket.send_json(video_processor.latest_detections)
                logger.info(f"Sent detections: {len(video_processor.latest_detections['video'])} video, {len(video_processor.latest_detections['gps'])} GPS")
            else:
                logger.warning("No detections to send")
            await asyncio.sleep(3)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(video_processor.generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)