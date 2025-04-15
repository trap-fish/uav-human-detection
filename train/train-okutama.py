from ultralytics import YOLO, RTDETR
import os
from datetime import datetime

expdate = datetime.today().strftime('%Y%m%d')

wkdir = "/media/citi-ai/matthew/uav-human-detection"

# yaml files for datasets
data_rltv_path = "data_files/"
datasets =  ["VisDrone.yaml"]

# model paths
model_dir_pth = os.path.join(wkdir, "models/")
yolov5n_pth = "yolov5nu.pt"
yolo5s_pth = "yolov5su.pt"
yolov8n_pth = "yolov8n.pt"
yolo11n_pth = "yolo11n.pt"
yolo11s_pth = "yolo11s.pt"
rtdetr_pth = "rtdetr-l.pt"
yolo11p2_path = "yolo11-p2.yaml"
#yolo5p2_path = "yolo5-p2.yaml"


model_dir = {
    # "yolo5n": {"type": "yolo", "path": yolov5n_pth},
    "yolo5s": {"type": "yolo", "path": yolo5s_pth},
    #"yolop2n": {"type": "yolop2", "path": yolo5p2_path}
    #"yolop2n": {"type": "yolop2", "path": yolo11p2_path},
    #"yolov8n": {"type": "yolo", "path": yolov8n_pth},
    # "yolo11n": {"type": "yolo", "path": yolo11n_pth},
    # "yolo11s": {"type": "yolo", "path": yolo11s_pth},
    # "yolop2s": {"type": "yolop2", "path": yolo11p2_path},
    # "rtdetrl": {"type": "rtdetr", "path": rtdetr_pth},
}

# Define experiments: model, optimizer, and learning rate combinations
# 10 for backbone (9 for rtdetr), 23 for yolo5, 22 for yolo11, 28 for 11p2, 27 rtdet
freeze = [10, 23]
experiments = [
    {"optimizer": "SGD", "lr": 0.01, "freeze": None, "cos_lr": True},
    {"optimizer": "SGD", "lr": 0.01, "freeze": freeze[0], "cos_lr": True},
    {"optimizer": "SGD", "lr": 0.01, "freeze": freeze[1], "cos_lr": True}, 
    {"optimizer": "SGD", "lr": 0.01, "freeze": None, "cos_lr": False},
    {"optimizer": "SGD", "lr": 0.001, "freeze": None, "cos_lr": False},
    {"optimizer": "SGD", "lr": 0.0005, "freeze": None, "cos_lr": False},
    {"optimizer": "SGD", "lr": 0.01, "freeze": freeze[0], "cos_lr": False},
    {"optimizer": "SGD", "lr": 0.01, "freeze": freeze[1], "cos_lr": False},
    {"optimizer": "SGD", "lr": 0.001, "freeze": freeze[0], "cos_lr": False},
    {"optimizer": "SGD", "lr": 0.0005, "freeze": freeze[0], "cos_lr": False},
]

# Paths and parameters
EPOCHS = 150  # Number of training epochs
BATCH_SZ = 16
PATIENCE = 15
IMGSZ = 640  # Image size for training
SINGLE_CLS = True
DEGREES=45        
SCALE=0.9
SHEAR=0 # 2
PERSPECTIVE=0.0005
MOSAIC=0.5
MIXUP=0.5

# Directory to save results
res_rltv_dir = f"results/experiment_{expdate}"
results_dir = os.path.join(wkdir, res_rltv_dir)
print(results_dir)

os.makedirs(results_dir, exist_ok=True)

for data_yml in datasets:
    data_path = os.path.join(wkdir, data_rltv_path, data_yml)

    # loop through each model
    for model_name, model_info in model_dir.items():
        # Train the model with the custom optimizer
        print(f"\nTraining with model: {model_name}")
        model_type = model_info["type"]
        model_file = model_info["path"]
        model_path = os.path.join(model_dir_pth, model_file)

        if not os.path.isfile(model_path):
            raise ValueError(f"Model file not found: {model_path}")

        # initialise model
        if model_type == "yolo":
            model = YOLO(model_path)
        elif model_name == "yolop2n":
            pretrained = os.path.join(model_dir_pth, "yolo11n.pt") # load yaml cfg with pretrained
            model = YOLO(model_path).load(pretrained)
        elif model_name == "yolop2s":
            pretrained = os.path.join(model_dir_pth, "yolo11s.pt") # load yaml cfg with pretrained
            model = YOLO(model_path).load(pretrained)
        elif model_type == "rtdetr":
            model = RTDETR(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Loop through experiments
        for i, exp in enumerate(experiments):
            if i < 4:
                continue
            print(f"\nStarting Experiment {i + 1}: {exp}")
            params = f"{exp['optimizer']}_lr{exp['lr']}_frz{exp['freeze']}_coslr{exp['cos_lr']}"
            variation = data_path.split('/')[-1].split(".")[0]
            exp_name = f"exp_{i + 1}_{model_name}_{variation}_{params}"

            # Train the model
            model.train(
                data=data_path,
                batch=BATCH_SZ,
                epochs=EPOCHS,
                imgsz=IMGSZ,
                optimizer=exp["optimizer"],
                lr0=exp["lr"],
                patience=PATIENCE,
                project=results_dir,
                name=exp_name,
                freeze=exp["freeze"],
                cos_lr=exp["cos_lr"],
                augment=True,
                single_cls=SINGLE_CLS,
                # degrees=DEGREES,
                # scale=SCALE,
                # shear=SHEAR,
                # perspective=PERSPECTIVE,
                # mosaic=MOSAIC,
                # mixup=MIXUP
            )

            print(f"Experiment {i + 1} completed! Results saved in {results_dir}/{exp_name}")

print("\nAll experiments completed!")
