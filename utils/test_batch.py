import os
import csv
from ultralytics import YOLO
import time

# Define the root directory and dataset
root_dir = "/media/citi-ai/matthew/uav-human-detection/"
data_list = ["VisDrone.yaml"]
results_dir = os.path.join(root_dir, "results/experiment_20250315")
output_csv = os.path.join(results_dir, "visdrone2cls_results_val_all.csv")
formats = ["onnx", "openvino"]
device = "cpu"

    # Initialize CSV
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['model_type', 'experiment_name', 'dataset', 'precision', 'recall', 'mAP50', 'mAP95','fitness', 'processing_time', 'fps', 'preprocess', 'inference', 'postprocess', 'loss','', 'model_path'])
    for datayml in data_list:

        dataset = os.path.join(root_dir, "data_files", datayml)

        for exp in os.listdir(results_dir):
            if not os.path.isdir(os.path.join(results_dir, exp)):
                ValueError(f"Warning: {exp} is not a directory. Skipping...")
                continue

            exp_dir = os.path.join(results_dir, exp)
            if not os.path.exists(os.path.join(exp_dir, "weights")):
                ValueError(f"Warning: {exp} does not contain a weights directory. Skipping...")
                continue

            # if 'okutama' not in model_path.lower():
            #     continue
            experiment_name = exp
            for model_format in formats:
                if model_format == 'openvino':
                    model_path = os.path.join(exp_dir, "weights/best_openvino_model")
                else:
                    model_path = os.path.join(exp_dir, f"weights/best.{model_format}")
                
                print(f"\n\nTesting {experiment_name} with model {model_path}\n\n")

                # Load model and run validation
                model = YOLO(model_path)
                start_time = time.perf_counter()
                results = model.val(data=dataset, split='val', device=device)
                end_time = time.perf_counter()

                processing_time = end_time - start_time
                fps = 532 / processing_time  # visdrone val has 532 images

                # Extract relevant metrics
                mAP50 = results.results_dict['metrics/mAP50(B)'].round(6)
                precision = results.results_dict['metrics/precision(B)'].round(6)
                recall = results.results_dict['metrics/recall(B)'].round(6)
                mAP95 = results.results_dict['metrics/mAP50-95(B)'].round(6)
                fitness = results.results_dict['fitness'].round(6)
                preprocess = round(results.speed['preprocess'], 6)
                inference = round(results.speed['inference'], 6)
                postprocess = round(results.speed['postprocess'], 6)
                loss = round(results.speed['loss'], 6)

                model_type = str.split(experiment_name, '_')[2]

                # Write to CSV
                writer.writerow([model_type, experiment_name, datayml, precision, recall, mAP50, mAP95, fitness , processing_time, fps, preprocess, inference, postprocess, loss, '', model_path])

print(f"Results saved to {output_csv}")
