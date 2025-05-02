import os
import json
import shutil
from tqdm import tqdm

def convert_bbox_coco2yolo(img_width, img_height, bbox):
    """
    Convert bounding box from COCO  format to YOLO format

    Parameters
    ----------
    img_width : int
        width of image
    img_height : int
        height of image
    bbox : list[int]
        bounding box annotation in COCO format: 
        [top left x position, top left y position, width, height]

    Returns
    -------
    list[float]
        bounding box annotation in YOLO format: 
        [x_center_rel, y_center_rel, width_rel, height_rel]
    """
    
    # YOLO bounding box format: [x_center, y_center, width, height]
    # (float values relative to width and height of image)
    x_tl, y_tl, w, h = bbox

    dw = 1.0 / img_width
    dh = 1.0 / img_height

    x_center = x_tl + w / 2.0
    y_center = y_tl + h / 2.0

    x = x_center * dw
    y = y_center * dh
    w = w * dw
    h = h * dh

    return [x, y, w, h]
    
def convert_coco_json_to_yolo_txt(output_path, json_file):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    path = os.makedirs(output_path)

    with open(json_file) as f:
        json_data = json.load(f)

    label_file = os.path.join(output_path, "labels")
    categories_list = ["person", "car"]
    with open(label_file, "w") as f:
        for category in tqdm(categories_list, desc="Categories"):
            f.write(f"{category}\n")

    for image in tqdm(json_data["images"], desc="Annotation txt for each image"):
        
        img_id = image["id"]
        img_name = image["file_name"]
        img_width = image["width"]
        img_height = image["height"]

        anno_in_image = [anno for anno in json_data["annotations"] if anno["image_id"] == img_id]
        anno_txt = os.path.join(output_path, img_name.split(".")[0] + ".txt")
        with open(anno_txt, "w") as f:
            for anno in anno_in_image:
                cur_cat = anno["category_id"]
                cat_dict = {1:0, 2:1}   # creates a dict for the classes 
                if cur_cat in [1,2]:       # only keep relevant labels
                    bbox_COCO = anno["bbox"]
                    x, y, w, h = convert_bbox_coco2yolo(img_width, img_height, bbox_COCO)
                    f.write(f"{cat_dict[cur_cat]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print("Converting COCO Json to YOLO txt finished!")
    
convert_coco_json_to_yolo_txt("train_labels_output", "/media/citi-ai/matthew/uav-human-detection/datasets/coco-2017/train/labels.json")
convert_coco_json_to_yolo_txt("val_labels_output","/media/citi-ai/matthew/uav-human-detection/datasets/coco-2017/validation/labels.json")