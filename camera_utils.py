import os
import cv2
import ffmpeg
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def get_camera_source():
    return os.getenv('CAMERA_SOURCE', '0')

def initialize_capture(source):
    if os.getenv('ENVIRONMENT') == 'local':
        if source.isdigit():
            return cv2.VideoCapture(int(source))
        return cv2.VideoCapture(source)
    else:
        # Sử dụng ffmpeg và av cho môi trường server
        stream = ffmpeg.input(source)
        stream = ffmpeg.output(stream, 'pipe:', format='rawvideo', pix_fmt='rgb24')
        process = ffmpeg.run_async(stream, pipe_stdout=True)
        return 3

def read_frame(capture):
    if isinstance(capture, cv2.VideoCapture):
        ret, frame = capture.read()
        return ret, frame
    else:
        for frame in capture.decode(video=0):
            return True, np.frombuffer(frame.to_rgb().planes[0], np.uint8).reshape(frame.height, frame.width, 3)

def release_capture(capture):
    if isinstance(capture, cv2.VideoCapture):
        capture.release()
    else:
        capture.close()