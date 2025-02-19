from ultralytics import RTDETR, YOLO
import os


model = YOLO('yolo11n.pt')
# model = YOLO("/media/citi-ai/matthew/mot-detection-training/runs/detect/train31/weights/best.pt") # this is the combined run
# model = YOLO("/media/citi-ai/matthew/mot-detection-training/results/exp1_yolo11m_results_hituav/yolo11m_cosine8/weights/best.pt")

#data_path = "Okutama.yaml"
# data_path = "hit-uav.yaml"
data_path = "data_files/niicu_test.yaml"


# Evaluate on the test dataset
results = model.val(data=data_path, split='test')  # Use the test split
results = results.results_dict

print(results)


# # Save results to a file
# output_file = os.path.join(results_path, f"test_results.txt")
# with open(output_file, "w") as f:
#     f.write(str(results))
#     f.close()
