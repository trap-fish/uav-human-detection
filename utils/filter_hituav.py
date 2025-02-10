import os
import shutil


def filter_annotation(input_annotation_file, output_annotation_file, filtered_classes):
    """
    Filters the .txt annotation file for human objects only.
    """
    filtered_lines = []
    with open(input_annotation_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            components = line.strip().split(" ")
            class_id = int(components[0])  # First column is class ID
            
            if class_id in filtered_classes:
                # Change class 4 to 1
                if class_id == 4:
                    components[0] = "1"  # Modify class ID
                
                filtered_lines.append(" ".join(components) + "\n")
    
    if not filtered_lines:
        return False  # No human-related annotations found
    
    # Write filtered annotations to the output file
    with open(output_annotation_file, "w") as f:
        f.writelines(filtered_lines)
    
    return True


def process_dataset(images_dir, output_images_dir, annotations_dir, output_annotations_dir, class_ids):
    """
    Process the VisDrone dataset to retain only human-related data.
    """
    print("processing data")
    annotation_files = os.listdir(annotations_dir)
    for annotation_file in annotation_files:
        input_annotation_path = os.path.join(annotations_dir, annotation_file)
        output_annotation_path = os.path.join(output_annotations_dir, annotation_file)
        
        # Filter annotation and copy image if relevant
        if filter_annotation(input_annotation_path, output_annotation_path, class_ids):
            # Copy the corresponding image
            image_file = annotation_file.replace(".txt", ".jpg")
            input_image_path = os.path.join(images_dir, image_file)
            output_image_path = os.path.join(output_images_dir, image_file)
            
            if os.path.exists(input_image_path):
                shutil.copy(input_image_path, output_image_path)


if __name__ == "__main__":
    datasets = ['train', 'val', 'test']
    
    human_class_ids = [0]  # class ID that will remain in the dataset

    for data_dir in datasets:
        output_dir = f"/media/citi-ai/matthew/mot-detection-training/datasets/filtered/hit-uav/{data_dir}/"
        dataset_dir = "/media/citi-ai/matthew/visdrone-train/datasets/hit-uav/"
        annotations_dir = os.path.join(dataset_dir, "labels", data_dir)
        images_dir = os.path.join(dataset_dir, "images", data_dir)

        # Output directories
        output_images_dir = os.path.join(output_dir, "images")
        output_annotations_dir = os.path.join(output_dir, "labels")
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_annotations_dir, exist_ok=True)

        process_dataset(images_dir, output_images_dir,
                        annotations_dir, output_annotations_dir,
                        human_class_ids)

        print(f"Filtered dataset saved at {output_dir}")
