import os
import shutil

dataset = "train" 

# Paths to the original dataset
root_dir = "/media/citi-ai/matthew/uav-human-detection/datasets/"
dataset_dir = os.path.join(root_dir, f"visdrone/VisDrone2019-DET-{dataset}")
annotations_dir = os.path.join(dataset_dir, "annotations")
images_dir = os.path.join(dataset_dir, "images")
output_dir = os.path.join(root_dir, f"filtered/visdrone_humans/{dataset}")

# Output directories
output_images_dir = os.path.join(output_dir, "images")
output_labels_dir = os.path.join(output_dir, "annotations")
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# Define human-related class IDs based on the VisDrone dataset specification
HUMAN_CLASS_IDS = [0, 1, 2]  # Adjust this based on the VisDrone class mapping

def filter_annotation(input_annotation_file, output_annotation_file, mergeClass=False):
    """
    Filters the .txt annotation file for human objects only.
    """
    filtered_lines = []
    with open(input_annotation_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            x, y, w, h, score, class_id, truncation, occulsion = line.strip().split(",")
            if int(class_id) in HUMAN_CLASS_IDS:
                if mergeClass:
                    class_id = 1  # Map all human classes to 1
                filtered_lines.append(f"{x}, {y}, {w}, {h}, {score}, {class_id}, {truncation}, {occulsion}\n")
    
    if not filtered_lines:
        return False  # No human-related annotations found
    
    # Write filtered annotations to the output file
    with open(output_annotation_file, "w") as f:
        f.writelines(filtered_lines)
    
    return True

def process_dataset():
    """
    Process the VisDrone dataset to retain only human-related data.
    """
    print(f"processing data: {annotations_dir}")
    annotation_files = os.listdir(annotations_dir)
    for annotation_file in annotation_files:
        input_annotation_path = os.path.join(annotations_dir, annotation_file)
        output_annotation_path = os.path.join(output_labels_dir, annotation_file)
        
        # Filter annotation and copy image if relevant
        if filter_annotation(input_annotation_path, output_annotation_path):
            # Copy the corresponding image
            image_file = annotation_file.replace(".txt", ".jpg")
            input_image_path = os.path.join(images_dir, image_file)
            output_image_path = os.path.join(output_images_dir, image_file)
            
            if os.path.exists(input_image_path):
                shutil.copy(input_image_path, output_image_path)

if __name__ == "__main__":
    process_dataset()
    print(f"Filtered dataset saved at {output_dir}")
