from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
from ultralytics import YOLOv10
import base64
import json

app = FastAPI()

# Tải mô hình YOLOv8 đã được huấn luyện trước
model = YOLOv10('best.pt')

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
        /* Các style giữ nguyên như cũ */
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="title-bar">
            <h1>FOD Detection</h1>
        </div>

        <div id="map"></div>

        <div id="video-container">
            <video id="video-feed" autoplay playsinline></video>
            <canvas id="canvas" style="display:none;"></canvas>
        </div>

        <div id="detections-container">
            <h2 style="color: red;">Detected Objects:</h2>
            <ul id="detections" class="list-group"></ul>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>

    <script>
        const ws = new WebSocket("ws://" + window.location.host + "/ws");
        const video = document.getElementById('video-feed');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const detectionsList = document.getElementById("detections");
        const videoContainer = document.getElementById("video-container");

        // Initialize map
        const map = L.map('map').setView([21.0050, 105.8441], 13);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);

        // Toggle video fullscreen
        videoContainer.addEventListener('click', function() {
            this.classList.toggle('fullscreen');
        });

        // Access user's camera
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
                video.play();
            })
            .catch(err => console.error("Error accessing camera:", err));

        // Send video frames to server
        function sendFrame() {
            if (video.readyState === video.HAVE_ENOUGH_DATA) {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const imageData = canvas.toDataURL('image/jpeg', 0.8);
                ws.send(JSON.stringify({image: imageData}));
            }
            requestAnimationFrame(sendFrame);
        }

        video.onloadedmetadata = sendFrame;

        // Handle WebSocket messages
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
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
        data = json.loads(data)
        img_data = data['image'].split(',')[1]
        img = base64.b64decode(img_data)
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        
        # Thực hiện phát hiện đối tượng
        results = model(img)
        
        # Chuẩn bị dữ liệu phát hiện
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = model.names[cls]
                detections.append({"name": name, "confidence": conf})
        
        # Gửi dữ liệu phát hiện về client
        await websocket.send_text(json.dumps({"detections": detections}))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
