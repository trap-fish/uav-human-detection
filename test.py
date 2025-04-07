from ultralytics import RTDETR, YOLO
import os

wkdir = "/media/citi-ai/matthew/uav-human-detection"

model_rltv_path = "/media/citi-ai/matthew/uav-human-detection/results/experiment_20250219/exp_1_yolo11n_VisDrone_SGD_lr0.01_frzNone_coslrTrue/weights/best.pt"
model_path = os.path.join(wkdir, model_rltv_path)
model = YOLO(model_path)

#data_path = "human-det.yaml"
#data_path = "Okutama.yaml"
#data_path = "hit-uav.yaml"
data_rltv_path = "data_files/VisDrone.yaml"
dataset = data_rltv_path.split(".", maxsplit=1)[0].split("/")[-1].lower().replace("-", "_")
data_path = os.path.join(wkdir, data_rltv_path)

project_dir = f"results/validation/experiment_20250219/exp_1_yolo11n_VisDrone_SGD_lr0.01_frzNone_coslrTrue{dataset}/"
os.makedirs(project_dir, exist_ok=True)

results = model.val(data=data_path, split='val', project=project_dir)  # Use the test split
results_path = project_dir

# # Save results to a file
# output_file = os.path.join(results_path, f"test_results.txt")
# with open(output_file, "w") as f:
#     f.write(str(results))
#     f.close()
