from ultralytics import YOLO
import os

wkdir = "/media/citi-ai/matthew/uav-human-detection"
model_rltv_path = "/media/citi-ai/matthew/uav-human-detection/runs/detect/train34/weights/best.pt"

#model_rltv_path = "yolov8n.pt"
model_path = os.path.join(wkdir, model_rltv_path)
model = YOLO(model_path)
imgsz = (640, 640)
data_path = os.path.join(wkdir, "data_files/VisDrone.yaml")

#model.export(format='tflite', int8=True, device='cpu', data=data_path)
#model.export(format='tflite', int8=True, device='cpu', data=data_path, imgsz=imgsz)
#model.export(format='ncnn', half=True, imgsz=imgsz)
model.export(format='onnx', imgsz=imgsz, half=True, simplify=True, nms=True, conf=0.4)
