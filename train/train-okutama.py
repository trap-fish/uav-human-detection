from ultralytics import YOLO, RTDETR
import os
from datetime import datetime

expdate = datetime.today().strftime('%Y%m%d')

wkdir = "/media/citi-ai/matthew/uav-human-detection"

# yaml files for datasets
data_rltv_path = "data_files/"
datasets = ['combined.yaml', "Okutama.yaml", "VisDrone.yaml", "hit-uav.yaml"]

# model paths
model_rltv_path = "models/"
yolo11n_pth = "yolo11n.pt"
yolo11s_pth = "yolo11s.pt"

model_dir = {
    "yolo11n": {"type": "yolo", "path": yolo11n_pth},
    #"yolo11s": {"type": "yolo", "path": yolo11s_pth},
}

# Define experiments: model, optimizer, and learning rate combinations
experiments = [
    {"optimizer": "SGD", "lr": 0.01, "freeze": None, "cos_lr": True},
    {"optimizer": "SGD", "lr": 0.01, "freeze": 10, "cos_lr": True},
    {"optimizer": "SGD", "lr": 0.01, "freeze": 22, "cos_lr": True},
    {"optimizer": "SGD", "lr": 0.01, "freeze": None, "cos_lr": False},
    {"optimizer": "SGD", "lr": 0.001, "freeze": None, "cos_lr": False},
    {"optimizer": "SGD", "lr": 0.0005, "freeze": None, "cos_lr": False},
    {"optimizer": "SGD", "lr": 0.01, "freeze": 10, "cos_lr": False},
    {"optimizer": "SGD", "lr": 0.01, "freeze": 22, "cos_lr": False},
    {"optimizer": "SGD", "lr": 0.001, "freeze": 10, "cos_lr": False},
    {"optimizer": "SGD", "lr": 0.0005, "freeze": 10, "cos_lr": False},
]

# Paths and parameters
epochs = 400  # Number of training epochs
imgsz = 640  # Image size for training

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

        # initialise model
        if model_type == "yolo":
            model_path = os.path.join(wkdir, model_rltv_path, model_file)
            if os.path.isfile(model_path):
                model = YOLO(model_path)
            else:
                raise ValueError(f"Model file not found: {model_path}")

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Loop through experiments
        for i, exp in enumerate(experiments):
            print(f"\nStarting Experiment {i + 1}: {exp}")
            
            
            params = f"{exp['optimizer']}_lr{exp['lr']}_frz{exp['freeze']}_coslr{exp['cos_lr']}"
            variation = data_path.split('/')[-1].split(".")[0]
            exp_name = f"exp_{i + 1}_{model_name}_{variation}_{params}"

            # Train the model
            model.train(
                data=data_path,
                batch=16,
                epochs=epochs,
                imgsz=imgsz,
                optimizer=exp["optimizer"],
                lr0=exp["lr"],
                patience=20,
                project=results_dir,
                name=exp_name,
                freeze=exp["freeze"],
                cos_lr=exp["cos_lr"],
                augment=True
            )

            print(f"Experiment {i + 1} completed! Results saved in {results_dir}/{exp_name}")

print("\nAll experiments completed!")
