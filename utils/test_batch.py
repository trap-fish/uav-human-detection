from ultralytics import YOLO
import os

# # Load a pretrained YOLO11n model
#model = YOLO("/media/citi-ai/matthew/visdrone-train/results/exp1_training_results/exp_5_yolo11n-vd_SGD_lr0.001/weights/best.pt")

# # Run inference on 'bus.jpg' with arguments
# model.predict("/media/citi-ai/matthew/visdrone-train/datasets/Okutama/yolo-format/test/images/1.1.8.33.jpg", save=True, imgsz=640, conf=0.5)

exp_dir = "/media/citi-ai/matthew/visdrone-train/results/exp1_training_results/"
data_path = "Okutama.yaml"

# Evaluate on the test dataset
#results = model.val(data="Okutama.yaml", split='test', project=project_dir)  # Use the test split

skipdirs = [] # add any directories to skip over

for dirpath in os.listdir(exp_dir):
    exp_path = os.path.join(exp_dir, dirpath)
    results_path = os.path.join(exp_path, "evaluation")
    os.makedirs(results_path, exist_ok=True)
    
    model_path = os.path.join(exp_path, "weights", "best.pt")
    if "rtdetr" in model_path.split('/')[-3]:
        continue
    elif dirpath in skipdirs:
        continue

    # Load the model
    print(model_path)
    model = YOLO(model_path)

    # Evaluate the model
    results = model.val(data=data_path, split='test', project=results_path)

    # Save results to a file
    output_file = os.path.join(results_path, f"{exp_path}_results.txt")
    with open(output_file, "w") as f:
        f.write(f"Evaluation Results for {exp_path}:\n")
        f.write(str(results))
        f.close()
