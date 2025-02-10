from ultralytics import YOLO

data_path = "hit-uav.yaml"
variation = "cosine"
project = "/media/citi-ai/matthew/mot-detection-training/results/exp1_yolo11m_results_hituav"
name = f"yolo11m_{variation}"
# Train YOLOv8 model
model = YOLO("yolo11m.pt")  # Using YOLOv8 Nano pretrained model (smallest size)
model.train(
            data=data_path,
            batch=16,
            epochs=150,
            imgsz=512,
            patience=25,
            augment=True,
            #freeze=5,
            cache="disk",
            workers=2,  # Lower workers (default is often 8+)
            device="cuda:0",
            cos_lr=True,
            project=project,
            name=name
)

# # Perform inference on a test image or folder of images
# test_image_path = "/path/to/test/image_or_folder"
# model.predict(source=test_image_path, save=True)