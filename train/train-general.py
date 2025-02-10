from ultralytics import YOLO

data_path = "hit-uav.yaml"
variation = "frozen_bb_cosine"
project = "/media/citi-ai/matthew/visdrone-train/results/exp1_yolo11n_results_hituav"
name = f"yolo11n_{variation}"
# Train YOLOv8 model
model = YOLO("yolo11n.pt")  # Using YOLOv8 Nano pretrained model (smallest size)
model.train(
            data=data_path,
            batch=16,
            epochs=300,
            imgsz=640,
            patience=25,
            augment=True,
            freeze=10,
            cos_lr=True,
            project=project,
            name=name
)