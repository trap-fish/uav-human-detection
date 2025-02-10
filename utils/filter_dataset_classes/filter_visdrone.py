import os
import shutil

# Paths to the original dataset
dataset_dir = "/media/citi-ai/matthew/mot-detection-training/datasets/VisDrone2019-DET-train"
annotations_dir = os.path.join(dataset_dir, "annotations")
images_dir = os.path.join(dataset_dir, "images")
output_dir = "/media/citi-ai/matthew/mot-detection-training/datasets/filtered/VisDrone2019-DET-human-train"

# Output directories
output_images_dir = os.path.join(output_dir, "images")
output_annotations_dir = os.path.join(output_dir, "annotations")
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_annotations_dir, exist_ok=True)

# Define human-related class IDs based on the VisDrone dataset specification
HUMAN_CLASS_IDS = [0, 1]  # Adjust this based on the VisDrone class mapping

def filter_annotation(input_annotation_file, output_annotation_file):
    """
    Filters the .txt annotation file for human objects only.
    """
    filtered_lines = []
    with open(input_annotation_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            components = line.strip().split(",")
            class_id = int(components[5])  # 6th column is class ID
            if class_id in HUMAN_CLASS_IDS:
                filtered_lines.append(line)
                
    
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
    print("processing data")
    annotation_files = os.listdir(annotations_dir)
    for annotation_file in annotation_files:
        input_annotation_path = os.path.join(annotations_dir, annotation_file)
        output_annotation_path = os.path.join(output_annotations_dir, annotation_file)
        
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
