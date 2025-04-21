import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import re
from kafka import KafkaProducer
import json
from datetime import datetime

VIDEO_INPUT_PATH = './cars/test.mp4'
VIDEO_OUTPUT_PATH = './output_video.mp4'

class NoСarPlateDetected(Exception):
    pass

class СarPlateDetect():
    YOLO_PATH = 'runs/detect/train_numbers/weights/best.pt'
    HAAR_CASCADE_PATH = 'haarcascade/haarcascade_russian_plate_number.xml'
    IMAGE_PATH = './cars/10.jpg'
    FRAME = 5
    RED_COLOR = (0, 0, 255)
    pattern = r'^[А-ЯA-Z]{1}\d{3}[А-ЯA-Z]{2}\d{2,3}$'
    
    def open_img(self ,img_path):
        try:
            carplate_img = cv2.imread(img_path)
        except:
            raise FileNotFoundError(f"Не удалось загрузить/загрузить изображение с путём {img_path}")
        
        carplate_img = cv2.cvtColor(carplate_img, cv2.COLOR_BGR2RGB)
        return carplate_img
    
    def carplate_extract(self,image, carplate_haar_cascade):
        carplate_rects = carplate_haar_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10)

        if len(carplate_rects) == 0:
            print("Номерной знак на изображении не обнаружен.")
            return None

        for x, y, w, h in carplate_rects:
            carplate_img = image[y:y+h, x:x+w]
        return carplate_img

    def enlarge_img(self, image, scale_percent):
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return resized_image

    def detect_number_from_img(self):
        model = YOLO(self.YOLO_PATH)
        carplate_haar_cascade = cv2.CascadeClassifier(self.HAAR_CASCADE_PATH)
        kafka_producer = CarPlateKafkaProducer()
        carplate_img_rgb = self.open_img(self.IMAGE_PATH)

        carplate_extract_img = self.carplate_extract(carplate_img_rgb, carplate_haar_cascade)

        if carplate_extract_img is None:
            raise NoСarPlateDetected("Невозможно продолжить распознавание номерного знака из-за сбоя обнаружения.")
        
        carplate_extract_img = self.enlarge_img(carplate_extract_img, 150)
        results = model.predict(carplate_extract_img, conf=0.3)
        
        characters = []

        for result in results:
            boxes = result.boxes.xyxy  
            classes = result.boxes.cls
            if len(boxes) == 0:
                raise NoСarPlateDetected("Символы не обнаружены на изображении.")

            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = map(int, box[:4])
                char = model.names[int(cls)] 
                characters.append((x1, char))

        characters.sort(key=lambda x: x[0])

        plate_number = ""
        for x1, char in characters:
            plate_number += char

        if re.match(self.pattern, plate_number.upper()):
            print('Номер автомобиля:', plate_number)
            kafka_producer.send_plate(plate_number)

            
        return plate_number

    def process_video(self,video_path, output_path):
        carplate = list()  
        kafka_producer = CarPlateKafkaProducer()

        
        model = YOLO(self.YOLO_PATH)

        carplate_haar_cascade = cv2.CascadeClassifier(self.HAAR_CASCADE_PATH)

        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise FileNotFoundError(f"Не удалось открыть видео: {video_path}")

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

            if frame_count % self.FRAME == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                carplate_extract_img = self.carplate_extract(frame_rgb, carplate_haar_cascade)

                if carplate_extract_img is not None:
                    carplate_extract_img = self.enlarge_img(carplate_extract_img, 150)

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

            if re.match(self.pattern, plate_number.upper()):
                frame = cv2.putText(frame, f"Number: {plate_number}", (300, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, self.RED_COLOR, 2, cv2.LINE_AA)

                if plate_number not in carplate:
                    carplate.append(plate_number)
                    kafka_producer.send_plate(plate_number)


                out.write(frame)
                
        cap.release()
        out.release()
        kafka_producer.close()
        print(f"Обработка завершена. Выходное видео сохранено как: {output_path}")
        return carplate

class CarPlateKafkaProducer:
    def __init__(self, kafka_bootstrap_servers='host.docker.internal:9092', topic='car-plates'):
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8')
        )
        self.topic = topic

    def send_plate(self, plate_number):
        message = {
            'plate': plate_number,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.producer.send(self.topic, value=message)
        print(f"[Kafka] Отправлено: {message}")

    def close(self):
        self.producer.flush()
        self.producer.close()

if __name__ == '__main__':
    print(СarPlateDetect().detect_number_from_img())
    #print(СarPlateDetect().process_video(VIDEO_INPUT_PATH, VIDEO_OUTPUT_PATH))
