import os
import shutil

"""
Okutama-Action does not contain a separated validation and test dataset. 
This script will split a part of the training data to use for the validation.
Sequences were selected to ensure no overlap with the training data - similar to how the
test data sequences were selected (see Okutama-Actions docs for info on this)

It additionally converts the data to YOLO format
"""

# List of sequences to use for validation
val_data = ['1.2.2', '2.2.2', 
            '1.2.11', '2.2.11', 
            '1.1.5', '2.1.5']

# Paths
image_root_dir = "/media/citi-ai/matthew/visdrone-train/datasets/Okutama/train"
label_root_dir = "/media/citi-ai/matthew/visdrone-train/datasets/Okutama/train/Labels/SingleActionTrackingLabels/3840x2160"
output_dir = "/media/citi-ai/matthew/visdrone-train/datasets/Okutama/yolo-format"

# Create train and val directories for images and labels
output_image_train_dir = os.path.join(output_dir, "train/images")
output_label_train_dir = os.path.join(output_dir, "train/labels")
output_image_val_dir = os.path.join(output_dir, "val/images")
output_label_val_dir = os.path.join(output_dir, "val/labels")

os.makedirs(output_image_train_dir, exist_ok=True)
os.makedirs(output_label_train_dir, exist_ok=True)
os.makedirs(output_image_val_dir, exist_ok=True)
os.makedirs(output_label_val_dir, exist_ok=True)

# Original resolution and target resolution
original_width, original_height = 3840, 2160
target_width, target_height = 1280, 720
scale_x = target_width / original_width
scale_y = target_height / original_height

# Process each label file
for label_file in os.listdir(label_root_dir):
    if not label_file.endswith(".txt"):
        continue  # Skip non-label files

    # Extract video identifier (n.n.n) from label file name
    video_id = label_file.replace(".txt", "")
    
    # Determine if this video belongs to the validation set
    is_validation = video_id in val_data
    
    # Set appropriate output directories
    output_image_dir = output_image_val_dir if is_validation else output_image_train_dir
    output_label_dir = output_label_val_dir if is_validation else output_label_train_dir

    # Parse the label file
    frame_annotations = {}  # Dictionary to group annotations by frame
    label_path = os.path.join(label_root_dir, label_file)

    with open(label_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            data = line.strip().split()
            if len(data) < 10:
                continue  # Skip malformed lines

            xmin, ymin, xmax, ymax = map(int, data[1:5])
            frame = int(data[5])
            lost, occluded, generated = map(int, data[6:9])
            label = data[9].strip('"')

            if lost == 1:  # Skip annotations where the object is outside the screen
                continue

            # Rescale bounding box to target resolution
            x_center = (xmin + xmax) / 2 * scale_x / target_width
            y_center = (ymin + ymax) / 2 * scale_y / target_height
            box_width = (xmax - xmin) * scale_x / target_width
            box_height = (ymax - ymin) * scale_y / target_height

            # Determine the YOLO class ID
            if label == "Person":
                yolo_cls = 0  # Class ID for Person
            else:
                print(f"Warning: Unknown label '{label}' encountered. Skipping...")
                continue  # Skip any unknown labels

            # YOLO format: class_id x_center y_center width height
            yolo_annotation = f"{yolo_cls} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"

            # Add annotation to the corresponding frame
            if frame not in frame_annotations:
                frame_annotations[frame] = []
            frame_annotations[frame].append(yolo_annotation)

    # Write YOLO label files for each frame
    image_dir = os.path.join(image_root_dir, f"Drone{video_id.split('.')[0]}", 
                             "Morning" if video_id.split('.')[1] == "1" else "Noon", 
                             "Extracted-Frames-1280x720", video_id)
    
    for frame, annotations in frame_annotations.items():
        frame_name = f"{frame}.jpg"
        frame_path = os.path.join(image_dir, frame_name)

        if not os.path.exists(frame_path):
            print(f"Warning: Frame {frame_name} not found in {image_dir}")
            continue

        # Copy the image to the output directory and rename it
        output_image_name = f"{video_id}.{frame}.jpg"
        output_image_path = os.path.join(output_image_dir, output_image_name)
        shutil.copy(frame_path, output_image_path)

        # Write annotations to the corresponding label file
        label_output_path = os.path.join(output_label_dir, f"{video_id}.{frame}.txt")
        with open(label_output_path, "w") as label_out_file:
            label_out_file.write("\n".join(annotations))

    print(f"Processed {label_file}: YOLO labels and images saved in {'val' if is_validation else 'train'}")

print(f"YOLO dataset generated in:\nTrain Images: {output_image_train_dir}\nTrain Labels: {output_label_train_dir}\nVal Images: {output_image_val_dir}\nVal Labels: {output_label_val_dir}")
