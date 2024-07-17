from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from camera_utils import get_camera_source, initialize_capture, read_frame, release_capture

app = FastAPI()

class StreamSource(BaseModel):
    source: str = None

@app.post("/process_stream")
async def process_video_stream(source: StreamSource):
    camera_source = source.source or get_camera_source()
    try:
        capture = initialize_capture(camera_source)
        results = []
        for _ in range(10):  # Xử lý 10 frame
            ret, frame = read_frame(capture)
            if not ret:
                break
            # Xử lý frame ở đây (ví dụ: phát hiện FOD)
            results.append("Processed frame")
        release_capture(capture)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)