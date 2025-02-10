from ultralytics import RTDETR, YOLO
import os


model = YOLO("/media/citi-ai/matthew/mot-detection-training/results/exp1_yolo11n_results_hituav/yolo11n_cosine/weights/best.pt")
# model = YOLO("/media/citi-ai/matthew/mot-detection-training/runs/detect/train31/weights/best.pt") # this is the combined run
# model = YOLO("/media/citi-ai/matthew/mot-detection-training/results/exp1_yolo11m_results_hituav/yolo11m_cosine8/weights/best.pt")

project_dir = "/media/citi-ai/matthew/mot-detection-training/results/exp1_yolo11M_results_hituav_thermal_val_on_niicu/"
#data_path = "Okutama.yaml"
# data_path = "hit-uav.yaml"
data_path = "niicu_test.yaml"
os.makedirs(project_dir, exist_ok=True)

# Evaluate on the test dataset
results = model.val(data=data_path, split='test', project=project_dir)  # Use the test split
results_path = project_dir

# Save results to a file
output_file = os.path.join(results_path, f"test_results.txt")
with open(output_file, "w") as f:
    f.write(str(results))
    f.close()
