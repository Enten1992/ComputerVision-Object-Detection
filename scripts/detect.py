
import cv2
import numpy as np
import os

# Configuration
MODEL_CONFIG = os.getenv("MODEL_CONFIG", "yolov3.cfg")
MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", "yolov3.weights")
CLASSES_FILE = os.getenv("CLASSES_FILE", "coco.names")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", 0.5))
NMS_THRESHOLD = float(os.getenv("NMS_THRESHOLD", 0.4))

def load_model():
    """
    Loads the YOLOv3 model and class names.

    Returns:
        tuple: The loaded model, output layers, and class names.
    """
    print("Loading YOLOv3 model...")
    if not os.path.exists(MODEL_CONFIG) or not os.path.exists(MODEL_WEIGHTS):
        print("Error: Model config or weights file not found. Please download them.")
        return None, None, None

    net = cv2.dnn.readNet(MODEL_WEIGHTS, MODEL_CONFIG)
    with open(CLASSES_FILE, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    print("YOLOv3 model loaded successfully.")
    return net, output_layers, classes

def detect_objects(image_path, net, output_layers, classes):
    """
    Detects objects in an image using the YOLOv3 model.

    Args:
        image_path (str): Path to the input image.
        net: The loaded YOLOv3 model.
        output_layers: The output layers of the model.
        classes (list): The list of class names.

    Returns:
        np.array: The image with detected objects and bounding boxes.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None

    img = cv2.imread(image_path)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONF_THRESHOLD:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

    return img

if __name__ == "__main__":
    # Example usage
    # Before running, download yolov3.weights, yolov3.cfg, and coco.names
    # wget https://pjreddie.com/media/files/yolov3.weights
    # wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true -O yolov3.cfg
    # wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
    
    # Create a dummy image for testing if one doesn\\'t exist
    if not os.path.exists("test_image.jpg"):
        dummy_image = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.imwrite("test_image.jpg", dummy_image)

    net, output_layers, classes = load_model()
    if net is not None:
        result_image = detect_objects("test_image.jpg", net, output_layers, classes)
        if result_image is not None:
            cv2.imwrite("result_image.jpg", result_image)
            print("Object detection complete. Result saved to result_image.jpg")
