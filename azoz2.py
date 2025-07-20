import cv2
import time
import random
import argparse
import numpy as np
import pyttsx3  # speach to text 

# --------- Helper Functions ---------

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 190)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

def load_labels(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

def get_position(image_width, object_x):
    if object_x < image_width * 0.33:
        return "left"
    elif object_x > image_width * 0.66:
        return "right"
    else:
        return "center"

# --------- Main ---------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Video source")
    parser.add_argument("--names", type=str, default="class.names", help="Path to class names")
    parser.add_argument("--model", type=str, default="yolov8n.onnx", help="Path to ONNX model")
    parser.add_argument("--tresh", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--thickness", type=int, default=2, help="Bounding box thickness")
    args = parser.parse_args()

    # Load labels and model
    NAMES = load_labels(args.names)
    COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in NAMES]
    net = cv2.dnn.readNet(args.model)

    # Setup camera
    cap = cv2.VideoCapture(int(args.source))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    IMAGE_SIZE = 640

    while True:
        cap.grab()
        ret, frame = cap.retrieve()
        if not ret:
            break

        image = cv2.resize(frame.copy(), (IMAGE_SIZE, IMAGE_SIZE))
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (IMAGE_SIZE, IMAGE_SIZE), swapRB=True, crop=False)
        net.setInput(blob)
        preds = net.forward()
        preds = preds.transpose((0, 2, 1))

        image_height, image_width, _ = frame.shape
        x_factor = image_width / IMAGE_SIZE
        y_factor = image_height / IMAGE_SIZE

        class_ids, confs, boxes = [], [], []
        rows = preds[0].shape[0]

        for i in range(rows):
            row = preds[0][i]
            conf = row[4]
            class_scores = row[4:]
            _, _, _, max_idx = cv2.minMaxLoc(class_scores)
            class_id = max_idx[1]

            if class_scores[class_id] > args.tresh:
                class_ids.append(class_id)
                confs.append(class_scores[class_id])
                x, y, w, h = row[0], row[1], row[2], row[3]
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                boxes.append([left, top, width, height])

        indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.2, 0.5)
        for i in indexes:
            i = i[0] if isinstance(i, (tuple, list, np.ndarray)) else i
            box = boxes[i]
            class_id = class_ids[i]
            left, top, width, height = box
            position = get_position(image_width, left + width // 2)
            label = f"{NAMES[class_id]} {position}"

            # ينطق في كل مرة يتم فيها كشف الكائن
            speak(label)

            cv2.rectangle(frame, (left, top), (left + width, top + height), COLORS[class_id], args.thickness)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[class_id], args.thickness)

        cv2.imshow("YOLOv8 Detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
