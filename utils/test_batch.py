import os
import csv
from ultralytics import YOLO

# Define the root directory and dataset
root_dir = "/media/citi-ai/matthew/uav-human-detection/"
data_list = ["Okutama.yaml"]
results_dir = os.path.join(root_dir, "results/experiment_20250214")
output_csv = os.path.join(results_dir, "multidata_okutama_results_test.csv")

    # Initialize CSV
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['experiment_name', 'dataset', 'precision', 'recall', 'mAP50', 'mAP95','fitness','', 'model_path'])
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

            model_path = os.path.join(exp_dir, "weights/best.pt")

            if 'okutama' not in model_path.lower():
                continue
            experiment_name = exp

            print(f"Testing {experiment_name} with model {model_path}")

            # Load model and run validation
            model = YOLO(model_path)
            results = model.val(data=dataset, split='test')

            # Extract relevant metrics
            mAP50 = results.results_dict['metrics/mAP50(B)'].round(6)
            precision = results.results_dict['metrics/precision(B)'].round(6)
            recall = results.results_dict['metrics/recall(B)'].round(6)
            mAP95 = results.results_dict['metrics/mAP50-95(B)'].round(6)
            fitness = results.results_dict['fitness'].round(6)

            # Write to CSV
            writer.writerow([experiment_name, datayml, precision, recall, mAP50, mAP95, fitness,'', model_path])

print(f"Results saved to {output_csv}")
