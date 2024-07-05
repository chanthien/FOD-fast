import torch
import onnxruntime
import numpy as np
from PIL import Image


# Bước 1: Chuyển đổi từ PyTorch sang ONNX
model = torch.load('yolov10m.pt', map_location=torch.device('cpu'))
model = model['model']  # Nếu mô hình được lưu trong một từ điển
model.float()  # Chuyển về float để đảm bảo tương thích
dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(model, dummy_input, 'yolov10m.onnx', opset_version=13)

# Bước 2: Suy luận với mô hình ONNX
def preprocess_image(image_path, input_size=(640, 640)):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(input_size)
    image_array = np.array(image) / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0).astype(np.float32)
    return image_array

# Tải mô hình ONNX
ort_session = onnxruntime.InferenceSession('yolov10m.onnx')

# Chuẩn bị đầu vào
image_path = 'tests/d2455831-206c-4361-b482-797e1fde4855.jpg'
input_data = preprocess_image(image_path)

# Chạy suy luận
outputs = ort_session.run(None, {'images': input_data})

# Xử lý kết quả đầu ra (tùy thuộc vào cấu trúc đầu ra cụ thể của YOLOv10)
# Ví dụ:
boxes, scores, class_ids = outputs

# Áp dụng ngưỡng và NMS nếu cần
# ...

# In kết quả
print("Detected objects:", len(boxes))
for box, score, class_id in zip(boxes, scores, class_ids):
    print(f"Class: {class_id}, Score: {score}, Box: {box}")