async function startVideoProcessing() {
    if (!checkCompatibility()) return;

    try {
        const stream = await navigator.mediaDevices.getUserMedia({video: { facingMode: "user" }});
        const video = document.createElement('video');
        video.srcObject = stream;
        await video.play();

        const processingInterval = 200; // 0.1 seconds

        const processFrame = () => {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);

            const scaleFactor = 0.5;
            const scaledCanvas = document.createElement('canvas');
            scaledCanvas.width = canvas.width * scaleFactor;
            scaledCanvas.height = canvas.height * scaleFactor;
            scaledCanvas.getContext('2d').drawImage(canvas, 0, 0, scaledCanvas.width, scaledCanvas.height);

            const imageData = scaledCanvas.toDataURL('image/jpeg', 0.7);

            fetch('/process', {
                method: 'POST',
                body: new FormData().append('processed_data', imageData)
            })
            .then(response => response.text())
            .then(result => {
                document.getElementById('results').innerHTML = result;
            })
            .catch(error => console.error('Error:', error));
        };

        setInterval(processFrame, processingInterval);
    } catch (error) {
        handleError(error);
    }
}

function checkCompatibility() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Your browser doesn't support accessing the camera.");
        return false;
    }
    return true;
}

function handleError(error) {
    console.error("An error occurred:", error);
    let message = "An error occurred while processing the video.";
    if (error.name === "NotAllowedError") {
        message = "Camera access was denied. Please allow camera access and try again.";
    } else if (error.name === "NotFoundError") {
        message = "No camera found on your device.";
    }
    alert(message);
}