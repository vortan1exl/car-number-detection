import cv2
import numpy as np
from ultralytics import YOLO
import re

VIDEO_INPUT_PATH = 'C:/car_plate_detection-master/cars/0416.mp4'
VIDEO_OUTPUT_PATH = 'C:/car_plate_detection-master/output_video.mp4'
YOLO_PATH = 'C:/car_plate_detection-master/runs/detect/train_numbers/weights/best.pt'
HAAR_CASCADE_PATH = 'haar_cascades/haarcascade_russian_plate_number.xml'
FRAME = 5
RED_COLOR = (0, 0, 255)
pattern = r'^[А-ЯA-Z]{1}\d{3}[А-ЯA-Z]{2}\d{2,3}$'

def carplate_extract(image, carplate_haar_cascade):
    carplate_rects = carplate_haar_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10)

    if len(carplate_rects) == 0:
        return None

    for x, y, w, h in carplate_rects:
        carplate_img = image[y:y+h, x:x+w]
    return carplate_img

def enlarge_img(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image

def process_video(video_path, output_path):

    model = YOLO(YOLO_PATH)

    carplate_haar_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    plate_number = "Не распознано"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % FRAME == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            carplate_extract_img = carplate_extract(frame_rgb, carplate_haar_cascade)

            if carplate_extract_img is not None:
                carplate_extract_img = enlarge_img(carplate_extract_img, 150)

                results = model.predict(carplate_extract_img, conf=0.7)

                characters = []
                for result in results:
                    boxes = result.boxes.xyxy
                    classes = result.boxes.cls
                    if len(boxes) > 0:
                        for box, cls in zip(boxes, classes):
                            x1, y1, x2, y2 = map(int, box[:4])
                            char = model.names[int(cls)]
                            characters.append((x1, char))

                if characters:
                    characters.sort(key=lambda x: x[0])
                    plate_number = ""
                    for x1, char in characters:
                        plate_number += char

        if re.match(pattern, plate_number.upper()):
            frame = cv2.putText(frame, f"Number: {plate_number}", (300, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, RED_COLOR, 2, cv2.LINE_AA)
            out.write(frame)


    cap.release()
    out.release()
    print(f"Обработка завершена. Выходное видео сохранено как: {output_path}")

if __name__ == '__main__':
    process_video(VIDEO_INPUT_PATH, VIDEO_OUTPUT_PATH)