from ultralytics import YOLO
import os
root_dir = "/media/citi-ai/matthew/uav-human-detection/"
img_path = "datasets/uavdt/UAV-benchmark-S/S1604/img000003.jpg"
#"datasets/uavdt/UAV-benchmark-S/S0304/img000002.jpg"
            #"datasets/uavdt/UAV-benchmark-S/S1601/img000009.jpg"
img = os.path.join(root_dir, img_path)
model_rltv_path = os.path.join(root_dir, "results/experiment_20250214/exp_2_yolo11n_Okutama_SGD_lr0.01_frz10_coslrTrue/weights/best.pt")
model_path = os.path.join(root_dir, model_rltv_path)
model = YOLO(model_path)

model.predict(img, save=True, imgsz=640, conf=0.3)