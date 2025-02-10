from pathlib import Path
from PIL import Image
from tqdm import tqdm

def convert_box(size, box):
    """
    Convert VisDrone bounding box format to YOLO format.
    Args:
        size (tuple): Image dimensions (width, height).
        box (tuple): VisDrone box format (x_min, y_min, width, height).
    Returns:
        tuple: YOLO format (x_center, y_center, width, height).
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (box[0] + box[2] / 2.0) * dw
    y_center = (box[1] + box[3] / 2.0) * dh
    width = box[2] * dw
    height = box[3] * dh
    return x_center, y_center, width, height

def visdrone2yolo(filtered_dir):
    """
    Convert VisDrone annotations to YOLO format in the filtered directory.
    Args:
        filtered_dir (Path): Directory containing filtered VisDrone dataset.
    """
    annotations_dir = filtered_dir / 'annotations'
    images_dir = filtered_dir / 'images'
    labels_dir = filtered_dir / 'labels'
    labels_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(annotations_dir.glob('*.txt'), desc=f'Converting {filtered_dir}')
    for annotation_file in pbar:
        image_file = images_dir / annotation_file.with_suffix('.jpg').name
        if not image_file.exists():
            print(f"Warning: Image {image_file} not found. Skipping.")
            continue
        
        # Get image size
        img_size = Image.open(image_file).size
        
        # Process annotation file
        lines = []
        with open(annotation_file, 'r') as file:
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                if row[4] == '0':  # Ignore regions with class 0
                    continue
                cls = int(row[5]) - 1  # Adjust class index to 0-based
                box = tuple(map(int, row[:4]))
                yolo_box = convert_box(img_size, box)
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in yolo_box)}\n")
        
        # Save YOLO labels
        label_file = labels_dir / annotation_file.name
        with open(label_file, 'w') as out_file:
            out_file.writelines(lines)

# Update the path to your filtered dataset
filtered_dir = Path("/media/citi-ai/matthew/mot-detection-training/datasets/filtered/VisDrone2019-DET-human-val")
visdrone2yolo(filtered_dir)
print("Conversion completed successfully!")
