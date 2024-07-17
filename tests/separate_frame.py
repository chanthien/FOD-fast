import ffmpeg
import numpy as np


class VideoCapture:
    def __init__(self):
        self.process = None
        self.width = 640  # Giá trị mặc định, bạn có thể thay đổi
        self.height = 480  # Giá trị mặc định, bạn có thể thay đổi

    def initialize_capture(self, input_source: str, input_path: str) -> None:
        """
        Khởi tạo video capture từ input source và path.

        :param input_source: Loại input ('webcam', 'ip', 'url', etc.)
        :param input_path: Đường dẫn cụ thể của video source
        """
        if input_source == 'webcam':
            input_url = f'video={input_path}'
        elif input_source in ['ip', 'url']:
            input_url = input_path
        else:
            input_url = input_path  # Cho các loại input khác

        try:
            self.process = (
                ffmpeg
                .input(input_url)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run_async(pipe_stdout=True)
            )
        except ffmpeg.Error as e:
            print(f"Error initializing ffmpeg: {e.stderr.decode()}")
            raise

    def read(self):
        """
        Đọc và trả về frame tiếp theo từ video stream.

        :return: numpy array của frame hoặc None nếu không còn frame
        """
        try:
            in_bytes = self.process.stdout.read(self.width * self.height * 3)
            if not in_bytes:
                return None
            frame = (
                np
                .frombuffer(in_bytes, np.uint8)
                .reshape([self.height, self.width, 3])
            )
            return frame
        except Exception as e:
            print(f"Error reading frame: {e}")
            return None

    def release(self):
        """
        Giải phóng tài nguyên.
        """
        if self.process:
            self.process.stdout.close()
            self.process.wait()


# Ví dụ sử dụng
if __name__ == "__main__":
    cap = VideoCapture()
    try:
        cap.initialize_capture('ip', 'http://192.168.1.2:4747/video')

        frame_count = 0
        while True:
            frame = cap.read()
            if frame is None:
                break

            frame_count += 1
            print(f"Read frame {frame_count}: shape {frame.shape}")

            if frame_count >= 100:  # Dừng sau 100 frame
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        cap.release()