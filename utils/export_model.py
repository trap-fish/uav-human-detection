from ultralytics import YOLO
import os

wkdir = "/media/citi-ai/matthew/uav-human-detection"
model_rltv_path = "hailo-ai/shared_with_docker/visdrone/models/yolo5_export/exp13/weights/best.pt"

model_path = os.path.join(wkdir, model_rltv_path)
model = YOLO(model_path)
imgsz = (640, 640)
data_path = os.path.join(wkdir, "data_files/VisDrone.yaml")

model.export(format='onnx', opset=16)
# model.export(format='openvino', int8=False, imgsz=imgsz, data=data_path, batch=1, device='cpu')
