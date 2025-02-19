import os
import yaml
from ultralytics import YOLO

data_path = "data_files/visdrone_yolo.yaml"

# Train YOLOv11 model with modified stride
model = YOLO("./models/yolo11n-p2.yaml").load('./models/yolo11n.pt')  # load pretrained weights
model.train(
            data=data_path,
            batch=16,
            epochs=150,
            imgsz=640,
            patience=15,
            augment=True
            #freeze=10,
            #cos_lr=True
)
