from ultralytics import YOLO
import os

wkdir = "/media/citi-ai/matthew/uav-human-detection"
expdir = "results/experiment_20250315"
data_path = os.path.join(wkdir, "data_files/VisDrone.yaml")
model_rltv_path = os.path.join(wkdir, expdir)


for experiment in os.listdir(model_rltv_path):

    model_path = os.path.join(model_rltv_path, experiment, "weights/best.pt")
    model = YOLO(model_path)
    imgsz = (640, 640)
    print(f"Exporting {experiment} to onnx:\n")
    model.export(format='onnx', opset=14, device="cpu")
    print(f"Exporting {experiment} to onnx:\n\n")
    model.export(format='openvino', int8=False, imgsz=imgsz, data=data_path, batch=1, device='cpu')
