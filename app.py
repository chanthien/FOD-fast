from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from ultralytics import YOLOv10
import base64
import json

app = FastAPI()

# Tải mô hình YOLOv8 đã được huấn luyện trước
model = YOLOv10('models/best.pt')

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def get():
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FOD Detection Results</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <style>
        body {
            padding-top: 20px;
            background-color: #f8f9fa;
            font-family: 'Arial', sans-serif;
            overflow: hidden;
        }

        .title-bar {
            background-color: #343a40;
            color: white;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }

        #map {
            height: calc(100vh - 100px);
            width: 100%;
            position: absolute;
            bottom: 0;
            left: 0;
            z-index: 1;
        }

        #video-container {
            position: fixed;
            bottom: 20px;
            left: 0px;
            width: 300px;
            height: 225px;
            z-index: 3;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        #video-container.fullscreen {
            width: 80%;
            height: 80%;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }

        #video-feed {
            width: 100%;
            height: 100%;
            object-fit: cover;
            border: 2px solid #343a40;
            border-radius: 5px;
        }

        #detections-container {
            position: fixed;
            top: 100px;
            left: 0px;
            z-index: 2;
            background-color: rgba(0, 0, 0, 0.1);
            padding: 10px;
            border-radius: 5px;
            max-height: calc(150vh - 400px);
            overflow-y: auto;
        }

        #detections {
            color: red;
            list-style-type: none;
            padding: 0;
        }

        .list-group-item {
            background-color: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: red;
            margin-bottom: 5px;
            padding: 5px;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="title-bar">
            <h1>FOD Detection</h1>
        </div>

        <div id="map"></div>

        <div id="video-container">
            <img id="video-feed" alt="Video Feed">
        </div>

        <div id="detections-container">
            <h2 style="color: red;">Detected Objects:</h2>
            <ul id="detections" class="list-group"></ul>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>

    <script>
        // Initialize WebSocket connection
        const ws = new WebSocket("ws://" + window.location.host + "/ws");
        const detectionsList = document.getElementById("detections");
        const videoContainer = document.getElementById("video-container");
        const videoFeed = document.getElementById("video-feed");

        // Initialize map
        const map = L.map('map').setView([21.0050, 105.8441], 13);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);

        // Toggle video fullscreen
        videoContainer.addEventListener('click', function() {
            this.classList.toggle('fullscreen');
        });

        // Handle WebSocket messages
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            if (data.image) {
                videoFeed.src = "data:image/jpeg;base64," + data.image;
            }
            if (data.detections) {
                updateDetections(data.detections);
            }
        };

        // Update detections list
        function updateDetections(detections) {
            detectionsList.innerHTML = "";
            detections.forEach(det => {
                const li = document.createElement("li");
                li.className = "list-group-item";
                li.innerHTML = `
                    <strong>${det.name}</strong><br>
                    Confidence: ${det.confidence.toFixed(2)}
                `;
                detectionsList.appendChild(li);
            });
        }

        // Start video stream
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function(stream) {
                const video = document.createElement('video');
                video.srcObject = stream;
                video.play();

                function sendFrame() {
                    const canvas = document.createElement('canvas');
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    canvas.getContext('2d').drawImage(video, 0, 0);
                    const data = canvas.toDataURL('image/jpeg', 0.8);
                    ws.send(data);
                }

                setInterval(sendFrame, 100);  // Send frame every 100ms
            })
            .catch(function(err) {
                console.error("Error accessing webcam:", err);
            });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        img = base64.b64decode(data.split(',')[1])
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)

        # Thực hiện phát hiện đối tượng
        results = model(img)

        # Vẽ bounding box và chuẩn bị dữ liệu phát hiện
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = model.names[cls]
                detections.append({"name": name, "confidence": conf})

        # Chuyển đổi hình ảnh trở lại base64
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Gửi dữ liệu phát hiện và hình ảnh về client
        await websocket.send_text(json.dumps({"image": img_base64, "detections": detections}))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
