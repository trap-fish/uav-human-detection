import os
import shutil
from PIL import Image

def convert_bbox(size, box):
    """
    Convert (x1, y1, x2, y2) to normalized YOLO (x_center, y_center, width, height).
    """
    x1, y1, x2, y2 = box
    img_w, img_h = size  # Image width and height

    # Ensure no division by zero
    if img_w == 0 or img_h == 0:
        raise ValueError(f"Invalid image size: {size}")

    # Convert to YOLO format
    x_center = (x1 + x2) / 2.0 / img_w
    y_center = (y1 + y2) / 2.0 / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h

    return x_center, y_center, width, height

def convert_labels(images_dir, labels_dir, output_labels_dir, output_imagesrgb_dir, output_imagesthermal_dir, type_filter={0, 1}):
    """
    Converts tab-delimited annotations into YOLO format and copies relevant images.

    Args:
        images_dir (str): Path to the images folder.
        labels_dir (str): Path to the original label files.
        output_labels_dir (str): Path to save converted labels.
        output_imagesrgb_dir (str): Path to save relevant images.
        output_imagesthermal_dir (str): Path to save relevant images.
        type_filter (set): Set of object types to keep (e.g., {0, 1} for RGB+Thermal & Thermal).
    """
    os.makedirs(output_labels_dir, exist_ok=True)
    os.makedirs(output_imagesrgb_dir, exist_ok=True)
    os.makedirs(output_imagesthermal_dir, exist_ok=True)

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue
        
        image_file = os.path.join(images_dir, label_file.replace(".txt", ".jpg"))  # Match image name
        if not os.path.exists(image_file):
            print(f"Warning: Image {image_file} not found for {label_file}, skipping...")
            continue

        thermal_image_file = image_file.replace("/rgb/", "/thermal/")
        if not os.path.exists(thermal_image_file):
            print(f"Warning: Image {thermal_image_file} not found for {label_file}")

        # Get image dimensions
        img = Image.open(image_file)
        img_width, img_height = img.size
              
        output_lines = []
        with open(os.path.join(labels_dir, label_file), "r") as f:
            for line in f:
                parts = line.strip().split("\t")  # Tab-separated values
                
                x1, y1, x2, y2 = map(float, parts[:4])  # obj_type is in the 5th column
                
                # Convert to YOLO format
                yolo_bbox = convert_bbox((img_width, img_height), (x1, y1, x2, y2))
                
                # YOLO format: class_id (always 0) + bbox coordinates
                output_lines.append(f"0 {' '.join(f'{x:.6f}' for x in yolo_bbox)}\n")

        # Save converted labels and copy corresponding image
        if output_lines:
            with open(os.path.join(output_labels_dir, label_file), "w") as f_out:
                f_out.writelines(output_lines)
            
            # Copy corresponding image
            shutil.copy(image_file, os.path.join(output_imagesrgb_dir, os.path.basename(image_file)))
            shutil.copy(thermal_image_file, os.path.join(output_imagesthermal_dir, os.path.basename(thermal_image_file)))

            print(f"Processed {label_file} ✅ Copied: {os.path.basename(image_file)}")
        else:
            print(f"Skipped {label_file} (No relevant annotations) ⚠️")

# Example Usage
images_dir = "/media/citi-ai/matthew/mot-detection-training/datasets/niicu_test/images/rgb/"
labels_dir = "/media/citi-ai/matthew/mot-detection-training/datasets/niicu_test/labels"
output_labels_dir = "/media/citi-ai/matthew/mot-detection-training/datasets/niicu_test/yolo/labels"
output_imagesrgb_dir = "/media/citi-ai/matthew/mot-detection-training/datasets/niicu_test/yolo/images/rgb"
output_images_tir_dir = "/media/citi-ai/matthew/mot-detection-training/datasets/niicu_test/yolo/images/thermal"


convert_labels(images_dir, labels_dir, output_labels_dir, output_imagesrgb_dir,
               output_images_tir_dir, type_filter={0, 1})  # Keep only types 0 & 1
