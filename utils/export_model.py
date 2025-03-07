from ultralytics import YOLO
import os

wkdir = "/media/citi-ai/matthew/uav-human-detection"
model_rltv_path = "/media/citi-ai/matthew/uav-human-detection/results/experiment_20250219/exp_1_yolo11n_VisDrone_SGD_lr0.01_frzNone_coslrTrue/weights/best.pt"

#model_rltv_path = "yolov8n.pt"
model_path = os.path.join(wkdir, model_rltv_path)
model = YOLO(model_path)
imgsz = (640, 640)
data_path = os.path.join(wkdir, "data_files/VisDrone.yaml")

model.export(format='openvino', int8=True, imgsz=imgsz, data=data_path, batch=1, device='cpu')
model.export(format='openvino', int8=False, imgsz=imgsz, data=data_path, batch=1, device='cpu')
