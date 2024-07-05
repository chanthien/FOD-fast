import cv2
import logging
import asyncio
from utils import save_detections_to_db, detect_objects, transform_to_gps,sahi_model, compute_perspective_transform

logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(self):
        self.latest_frame = None
        self.latest_detections = None
        self.frame_buffer = asyncio.Queue(maxsize=10)

    async def process_frames(self, cap):
        asyncio.create_task(self._capture_frames(cap))
        asyncio.create_task(self._process_frame())

    async def _capture_frames(self, cap):
        logger.info("Starting capture_frames")
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error(f"Failed to capture frame {frame_count}")
                break
            await self.frame_buffer.put(frame)
            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(f"Captured {frame_count} frames")
            logger.info("Captured frame")
            await asyncio.sleep(0.1)

    async def _process_frame(self):
        logger.info("Starting process_frame")
        frame_count = 0
        while True:
            try:
                frame = await self.frame_buffer.get()
                self.latest_frame = frame
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")

                detections = detect_objects(sahi_model, frame)
                logger.debug(f"Detected {len(detections)} objects in frame {frame_count}")

                video_detections = [{
                    'name': det['name'],
                    'confidence': det['confidence'],
                    'bbox': det['bbox']
                } for det in detections]
                matrix = compute_perspective_transform(
                    [(105.843583, 21.004964), (105.844614, 21.005134), (105.844700, 21.004923),
                     (105.843556, 21.005169)],
                    [(0, 0), (1200, 0), (1200, 720), (0, 720)]
                )

                gps_detections = transform_to_gps(detections,matrix)
                logger.debug(f"Transformed {len(gps_detections)} detections to GPS")

                save_detections_to_db(gps_detections)

                self.latest_detections = {
                    'video': video_detections,
                    'gps': gps_detections
                }

            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {str(e)}", exc_info=True)
            await asyncio.sleep(0.01)

    async def process_single_image(self, image):
        self.latest_frame = image
        detections = detect_objects(sahi_model, image)
        gps_detections = transform_to_gps(detections)
        save_detections_to_db(gps_detections)
        self.latest_detections = {
            'video': [{
                'name': det['name'],
                'confidence': det['confidence'],
                'bbox': det['bbox']
            } for det in detections],
            'gps': gps_detections
        }

    async def generate_frames(self):
        frame_count = 0
        while True:
            if self.latest_frame is not None:
                try:
                    _, buffer = cv2.imencode('.jpg', self.latest_frame)
                    frame_count += 1
                    if frame_count % 100 == 0:
                        logger.info(f"Generated {frame_count} video frames")
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                except Exception as e:
                    logger.error(f"Error generating video frame: {str(e)}", exc_info=True)
            else:
                logger.warning("No frame available for video feed")
            await asyncio.sleep(0.1)