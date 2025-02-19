from sahi import AutoDetectionModel
from sahi.predict import get_prediction
from sahi.predict import get_sliced_prediction
import os

root_dir = "/media/citi-ai/matthew/uav-human-detection/"
img_path = "datasets/uavdt/UAV-benchmark-S/S1604/img000003.jpg"
img = os.path.join(root_dir, img_path)

model_rltv_path = os.path.join(root_dir, "results/experiment_20250214/exp_2_yolo11n_VisDrone_SGD_lr0.01_frz10_coslrTrue/weights/best.pt")
model_path = os.path.join(root_dir, model_rltv_path)

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    confidence_threshold=0.3,
    device="cpu",  # or 'cuda:0'
)

# With an image path
result = get_prediction(img, detection_model)
result.export_visuals(export_dir=f"demo_data/standard_{img.split('/')[-1].replace('jpg', 'jpeg')}")

result.export_visuals(export_dir="demo_data/")

result = get_sliced_prediction(
    img,
    detection_model,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
result.export_visuals(export_dir=f"demo_data/sliced_{img.split('/')[-1].replace('jpg', 'jpeg')}")
