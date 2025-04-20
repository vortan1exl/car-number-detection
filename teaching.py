from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    model.train(data='*', 
            epochs=100, 
            imgsz=416, 
            device=0,  
            batch=8)