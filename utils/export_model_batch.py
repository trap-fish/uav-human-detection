from ultralytics import YOLO
import os
import warnings
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--expdir",
                    help="relative path of the experiment directory, e.g. results/experiment_<experiment_type>",
                    type=str)

args = parser.parse_args()
expdir = args.expdir

wkdir = "/media/citi-ai/matthew/uav-human-detection"
expdir = args.expdir
data_path = os.path.join(wkdir, "data_files/VisDrone.yaml")
model_rltv_path = os.path.join(wkdir, expdir)


for experiment in os.listdir(model_rltv_path):
    model_path = os.path.join(model_rltv_path, experiment, "weights/best.pt")
    if not os.path.exists(model_path):
        warnings.warn("This path is either not a model or there is no best.pt")
        continue
    model = YOLO(model_path)
    imgsz = (640, 640)
    print(f"Exporting {experiment} to onnx:\n")
    model.export(format='onnx', opset=16, device="cpu")
    print(f"Exporting {experiment} to onnx:\n\n")
    model.export(format='openvino', int8=False, imgsz=imgsz, data=data_path, batch=1, device='cpu')
