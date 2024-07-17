import cv2
import onnxruntime as ort
import numpy as np
import yaml

# Đường dẫn đến mô hình ONNX
model_path = "models/best.onnx"
data_yaml_path = "tests/data.yaml"

# Tải mô hình
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

# Đọc tên các class từ file data.yaml
with open(data_yaml_path, 'r') as f:
    data = yaml.safe_load(f)
class_names = data['names']

# Hàm tiền xử lý ảnh
def preprocess_image(image, target_size=(640, 640)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

# Hàm suy luận
def infer_image(image):
    input_data = preprocess_image(image)
    result = session.run(None, {input_name: input_data})
    return result

# Hàm xử lý kết quả
# Hàm xử lý kết quả (sửa đổi)
def process_result(result):
    detections = []  # Tạo danh sách detections ở ngoài vòng lặp
    for obj in result[0]:
        bbox = obj[0][0:4]
        score = obj[0][4]
        category_id = int(obj[0][-1])

        detections.append({
            'id': category_id,
            'name': class_names[category_id],
            'bbox': bbox,
            'confidence': score
        })
    return detections

# Suy luận trên ảnh
image_path = "tests/d2455831-206c-4361-b482-797e1fde4855.jpg"
image = cv2.imread(image_path)
result = infer_image(image)
detections = process_result(result)
print(detections)
# print(result)
