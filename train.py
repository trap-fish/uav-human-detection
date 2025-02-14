import os
import yaml
from ultralytics import YOLO

# Set dataset paths
train_images_path = "/media/citi-ai/matthew/mot-detection-training/datasets/filtered/VisDrone2019-DET-human-train/images"
val_images_path = "/media/citi-ai/matthew/mot-detection-training/datasets/filtered/VisDrone2019-DET-human-train/images"

data_path = "combined.yaml"

# Train YOLOv8 model
model = YOLO("yolo11n.pt")  # Using YOLOv8 Nano pretrained model (smallest size)
model.train(
            data=data_path,
            batch=32,
            epochs=300,
            imgsz=640,
            patience=25,
            augment=True
            #freeze=10,
            #cos_lr=True
)
