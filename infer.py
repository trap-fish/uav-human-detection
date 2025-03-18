import cv2
import os
import json

def xywh2xyxy(xywh):
    x,y,w,h = xywh
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    return x1, y1, x2, y2

def xywhn2xyxy(xywh):
    x,y,w,h = xywh
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    return x1, y1, x2, y2

def x1y1whn2xcywh(xywh):
    x,y,w,h = xywh
    x1 = x - w / 2
    y1 = y - h / 2


    return x1, y1, w, h

def get_annotations(annotations_path, img_id):
    with open(annotations_path, 'r') as file:
        gt_data = json.load(file)
        file.close()
    annotations = gt_data['annotations']
    images = gt_data['images']
    gt = []
    for ann in annotations:
        if ann['image_id'] == img_id:
            gt.append([ann['category_id'], *ann['bbox']])
    for img in images:
        if img['id'] == img_id:
            file_name = img['file_name']
    return gt, file_name

def get_detections(results_path, img_id=None):
    with open(results_path, 'r') as file:
        results = json.load(file)
        file.close()
    preds = []
    for res in results:
        if img_id is not None:
            if res['image_id'] != img_id:
                continue
        img_id = res['image_id']
        preds.append([res['category_id'], *res['bbox'], res['score'], img_id])
        assert img_id is not None, f"Image ID has not been found in {results}"
    return preds

root_dir = "/media/citi-ai/matthew/uav-human-detection"
annotations_path = "hailo-ai/shared_with_docker/visdrone/annotations_VisDroneHumans_val.json"
results_path = f"hailo-ai/shared_with_docker/visdrone/detections_VisDrone_quant_dets.json"
images_dir = "/media/citi-ai/matthew/uav-human-detection/hailo-ai/shared_with_docker/VisDrone2019-DET-val/images/"


img_id = 234 # known imageid as per the dataset
gt, file_name = get_annotations(os.path.join(root_dir, annotations_path), img_id=img_id)
detections = get_detections(os.path.join(root_dir, results_path), img_id=img_id)

# Load the image
imagefilename = file_name #"0000001_02999_d_0000005.jpg" #"0000001_03999_d_0000007.jpg" 
imgpth = os.path.join(images_dir, imagefilename)
img = cv2.imread(imgpth)
img_name = os.path.basename(imgpth).split('.')[0]
print(img)

colorgt = {
    1: (66, 75, 245),
    2: (20, 240, 100)
}

thickness = 1

class_colors = {
    1: (0, 251, 255),  # Cyan
    2: (119, 0, 255)  # Dark blue
}

class_names = {
    1: 'pedestrian',
    2: 'person'
}

for idx, labels in enumerate(detections):
    # Unpack the labels
    clsid, x1_prd, y1_prd, x2_prd, y2_prd, conf, _ = labels
    x1_prd, y1_prd, x2_prd, y2_prd = xywh2xyxy([x1_prd, y1_prd, x2_prd, y2_prd])
    x1_prd, y1_prd, x2_prd, y2_prd = int(x1_prd), int(y1_prd), int(x2_prd), int(y2_prd)
    print(f"detections: {[x1_prd, y1_prd, x2_prd, y2_prd]}")

    # Filter by confidence threshold
    if conf > 0.20:
        # Select color based on class
        color = class_colors[clsid]  # Default to white if unknown class

        # Draw bounding box
        cv2.rectangle(img, (x1_prd, y1_prd), (x2_prd, y2_prd), color, thickness)

        # Display class name and confidence
        text = f"{class_names[clsid]}: {conf:.2f}"
        cv2.putText(img, text, (x1_prd, y1_prd - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


# # Loop through labels and draw bounding boxes
for idx, labels in enumerate(gt):
    clsid, x1, y1, w, h = labels
    if clsid not in [1,2]:
        continue
    x2 = x1 + w
    y2 = y1 + h
    color = colorgt[clsid]
    print(f"ground truth: {[x1, y1, x2, y2]}")
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    # Display class name and confidence
    text = f"{class_names[clsid]}"
    cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)



# Display the image with bounding boxes
cv2.imwrite('visdrone_inferCOCO_quant_img1.jpg', img)
cv2.waitKey(0)  # Wait for key press
cv2.destroyAllWindows()  # Close window