import os
import csv
from ultralytics import YOLO, RTDETR
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--results_pth",
                    help="relative path of the results file, e.g. results/experiment_<experiment_type>",
                    type=str)
parser.add_argument("-j", "--save_json",
                    help="to also save the results to a coco formated json file",
                    type=bool, default=False)
parser.add_argument("-n", "--model_name",
                    help="e.g yolo, rtdetr",
                    type=str)
parser.add_argument("-m", "--model_format",
                    help="e.g ['onnx', 'openvino']",
                    type=list, default=['onnx', 'openvino'])
parser.add_argument("-s", "--single_cls",
                    help="single class detection",
                    type=bool, default=False)

args = parser.parse_args()
results_pth = args.results_pth
save_json = args.save_json
model_name = args.model_name
model_format = args.model_format
single_cls = args.single_cls

# Define the root directory and dataset
root_dir = "/media/citi-ai/matthew/uav-human-detection/"
data_list = ["VisDrone.yaml"]
results_dir = os.path.join(root_dir, results_pth)
val_dir = os.path.join(results_dir, "validation_runs_FIXED")
output_csv = os.path.join(val_dir, "visdrone2cls_results_val_all.csv")

formats = model_format
device = "cpu"

# create validation dir if not exists
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

# initialise the csv
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['model_type', 'model_format', 'experiment_name', 'dataset', 'precision', 'recall', 'mAP50', 'mAP95','fitness', 'processing_time', 'fps', 'preprocess', 'inference', 'postprocess', 'loss', '', 'model_path'])
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
                
                val_subdir= experiment_name + "_" + model_format
                print(f"\n\nTesting {experiment_name} with model {model_path}\n\n")

                # Load model and run validation
                if model_name == 'rtdetr':
                    model = RTDETR(model_path)
                else:
                    model = YOLO(model_path, task='detect')
                start_time = time.perf_counter()
                results = model.val(data=dataset, split='val', device=device,
                                    save_json=save_json, project=val_dir,
                                    name=val_subdir, single_cls=single_cls)
                end_time = time.perf_counter()

                processing_time = round(end_time - start_time, 6)
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
                writer.writerow([model_type, model_format, experiment_name, datayml, precision, recall, mAP50, mAP95, fitness , processing_time, fps, preprocess, inference, postprocess, loss, '', model_path])

print(f"Results saved to {output_csv}")
