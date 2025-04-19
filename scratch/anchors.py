import torch
import os
root = "/media/citi-ai/matthew/uav-human-detection/hailo-ai/shared_with_docker/visdrone/models"
model = 'yolov5v0_exp11_export/weights/best.pt'
modelpath = os.path.join(root, model)
m = torch.load(modelpath)["model"]
detect = list(m.children())[0][-1]

print(detect.anchor_grid)