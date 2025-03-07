from ultralytics import RTDETR, YOLO
import os

wkdir = "/media/citi-ai/matthew/uav-human-detection"


# # Load a COCO-pretrained RT-DETR-l model
# model_path = "results/exp1_training_results/exp_5_yolo11n_SGD_lr0.001/weights/best.pt"
# model_path = "results/exp1_training_results/exp_5_yolo11n_SGD_lr0.001/weights/best.pt"
# model_path = "results/exp3_ft_training_results/exp_1_rtdetr-l_SGD_lr0.01_Okutama_finetuned/weights/best.pt"
# model_path = "media/citi-ai/matthew/mot-detection-training/runs/detect/train31/weights/best.pt"
# model_path = "runs/detect/train17/weights/best.pt"
# model_path = "/media/citi-ai/matthew/mot-detection-training/results/exp1_yolo11n_results_hituav/yolo11n_cosine/weights/best.pt"
# model_path = "/media/citi-ai/matthew/mot-detection-training/runs/detect/train31/weights/best.pt"
# model_path = "results/exp1_training_results/exp_5_yolo11n_SGD_lr0.001/weights/best.pt"
# model_rltv_path = "models/exp_1_yolo11n_SGD_lr0.01_Okutama_finetuned.pt"
# model_rltv_path = "runs/detect/train31/weights/best.pt" # train28 is trained on visdrone, 31 is combined visdrone/okutama
model_rltv_path = "/media/citi-ai/matthew/uav-human-detection/runs/detect/train34/weights/best.pt"
model_path = os.path.join(wkdir, model_rltv_path)
dtype='int8'
model = YOLO(f"/media/citi-ai/matthew/uav-human-detection/runs/detect/train34/weights/best_saved_model/best_{dtype}.tflite", task='detect')

#data_path = "human-det.yaml"
#data_path = "Okutama.yaml"
#data_path = "hit-uav.yaml"
data_rltv_path = "data_files/VisDrone.yaml"
dataset = data_rltv_path.split(".", maxsplit=1)[0].split("/")[-1].lower().replace("-", "_")
data_path = os.path.join(wkdir, data_rltv_path)

project_dir = f"results/exp_1_yolo11n_SGD_lr0.01_Okutama_finetuned_test_on_{dataset}/"
os.makedirs(project_dir, exist_ok=True)
imgpath = '/media/citi-ai/matthew/uav-human-detection/datasets/filtered/visdrone_humans/val/images/0000001_02999_d_0000005.jpg'
imgsz = (640,640)
res = model.predict(imgpath, project=project_dir, device='cpu', imgsz=imgsz)
res[0].plot(show=True)

# # Evaluate on the test dataset
# results = model.val(data=data_path, split='test', project=project_dir, device='cpu')  # Use the test split
# results_path = project_dir

# # Save results to a file
# output_file = os.path.join(results_path, f"test_results.txt")
# with open(output_file, "w") as f:
#     f.write(str(results))
#     f.close()
