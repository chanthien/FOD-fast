import onnxruntime
import numpy as np
import cv2

# Load the ONNX model
onnx_session = onnxruntime.InferenceSession("models/best.onnx")


def preprocess_image(image_path, input_size=(640, 640)):
    image = cv2.imread(image_path)
    original_image = image.copy()
    image = cv2.resize(image, input_size)
    image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image.astype(np.float32), original_image


def run_inference(onnx_session, preprocessed_image):
    input_name = onnx_session.get_inputs()[0].name
    output = onnx_session.run(None, {input_name: preprocessed_image})
    return output[0]  # Assuming the first output is the detection results


def draw_bounding_boxes(image, detections, conf_threshold=0.5):
    for detection in detections[0]:  # Note the [0] here
        x1, y1, x2, y2, conf, class_id = detection
        if conf >= conf_threshold:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class: {int(class_id)}, Conf: {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


def process_image(image_path, onnx_session, output_path):
    preprocessed_image, original_image = preprocess_image(image_path)
    detections = run_inference(onnx_session, preprocessed_image)

    # No need to normalize coordinates as they seem to be in pixel values already

    result_image = draw_bounding_boxes(original_image, detections)
    cv2.imwrite(output_path, result_image)
    print(f"Processed image saved to: {output_path}")

    # Uncomment these lines if you want to display the image
    # cv2.imshow("Output", result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# Example usage
input_image_path = "tests/002570_jpg.rf.e0a3d8bd1f93b2de26ba88c2f181c51c.jpg"
output_image_path = "output.jpg"
process_image(input_image_path, onnx_session, output_image_path)