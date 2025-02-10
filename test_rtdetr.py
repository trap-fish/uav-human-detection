from ultralytics import RTDETR, YOLO
import os

# # Load a COCO-pretrained RT-DETR-l model
model = RTDETR("/media/citi-ai/matthew/visdrone-train/results/exp1_training_results/exp_5_yolo11n_SGD_lr0.001/weights/best.pt")
#model = RTDETR("rtdetr-l.pt")
model = RTDETR("/media/citi-ai/matthew/visdrone-train/results/exp3_ft_training_results/exp_1_rtdetr-l_SGD_lr0.01_Okutama_finetuned/weights/best.pt")

model = YOLO("yolo11n.pt")
model = YOLO("runs/detect/train17/weights/best.pt") # YOLO("/media/citi-ai/matthew/mot-detection-training/runs/detect/train31/weights/best.pt")
model = YOLO("/media/citi-ai/matthew/mot-detection-training/results/exp1_yolo11n_results_hituav/yolo11n_cosine/weights/best.pt")

model = YOLO("/media/citi-ai/matthew/mot-detection-training/runs/detect/train31/weights/best.pt")

#model = YOLO("/media/citi-ai/matthew/visdrone-train/results/exp1_training_results/exp_5_yolo11n_SGD_lr0.001/weights/best.pt")
#model = YOLO("/media/citi-ai/matthew/visdrone-train/results/exp3_ft_training_results/exp_1_yolo11n_SGD_lr0.01_Okutama_finetuned/weights/best.pt")
project_dir = "/media/citi-ai/matthew/visdrone-train/results/exp1_yolo11n_results_combined_test_on_humandet/"
data_path = "human-det.yaml"
#data_path = "Okutama.yaml"
#data_path = "hit-uav.yaml"
os.makedirs(project_dir, exist_ok=True)

# Evaluate on the test dataset
results = model.val(data=data_path, split='test', project=project_dir)  # Use the test split
results_path = project_dir

# Save results to a file
output_file = os.path.join(results_path, f"test_results.txt")
with open(output_file, "w") as f:
    f.write(str(results))
    f.close()
