import json
import os

# get a list of images (filename) and the corresponding image_id from annotations file
rootdir = "/media/citi-ai/matthew/uav-human-detection"
annotations_file = os.path.join(rootdir, 'hailo-ai/shared_with_docker/visdrone/annotations_VisDroneHumans_val.json')
images_path = os.path.join(rootdir, 'hailo-ai/shared_with_docker/visdrone/VisDrone2019-DET-val/images')

with open(annotations_file, 'r') as f:
    val_gt = json.load(f)
    f.close()
    
images = val_gt['images']
image_list = [(x['file_name'], x['id']) for x in images] 


# need to add an annotation ID and segmentation to json output from ultralytics to compare with original model
def map_imageid_to_json(detection_file):
    with open(detection_file, 'r') as f:
        annotations = json.load(f)
        f.close()


    # Create a mapping from images using file_name as key and image_id as value
    mapping = {item["file_name"].split('.')[0]: item["id"] for item in images}

    # Update predictions dict by mapping image_id
    for item in annotations:
        item["image_id"] = mapping.get(item["image_id"], item["image_id"])
        
    return annotations

        
preds_file = os.path.join(rootdir, 'results/experiment_20250414/validation_runs_FIXED/exp_1_yolo11n_VisDrone_SGD_lr0.01_frzNone_coslrTrue_onnx/predictions.json')
remapped_file = map_imageid_to_json(preds_file)
outputfile = os.path.join(rootdir, "hailo-ai/shared_with_docker/visdrone/exp20250414_yolo11n_exp1_ul.json")
with open(outputfile, 'w') as f:
    json.dump(remapped_file, f)
    f.close()

print(f"DETECTIONS SAVED TO {outputfile}")