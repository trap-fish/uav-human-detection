import os
import json
import csv
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Configs
annType = 'bbox'  # change to 'segm' or 'keypoints' if needed
rootdir = '/local/shared_with_docker/visdrone/'
annFile = os.path.join(rootdir, 'annotations_VisDroneHumans_val.json')
detections_dir = os.path.join(rootdir, 'model_eval')  # where all your detection JSONs are
output_csv = os.path.join(rootdir, 'evaluation_results.csv')

# Get list of detection files (adjust pattern if needed)
detection_files = [f for f in os.listdir(detections_dir) if f.endswith('.json') and 'detections' in f]

# Load ground truth
cocoGt = COCO(annFile)

# Prepare CSV output
headers = [
    'File', 'AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
    'AR_1', 'AR_10', 'AR_100', 'AR_small', 'AR_medium', 'AR_large'
]
rows = []

# Loop through detection files and evaluate
for det_file in detection_files:
    print(f"Evaluating: {det_file}")
    resFile = os.path.join(detections_dir, det_file)
    cocoDt = cocoGt.loadRes(resFile)

    imgIds = sorted(cocoGt.getImgIds())
    catIDs = [1, 2]  # adjust as needed
    maxDets = [1, 10, 100]

    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.params.catIds = catIDs
    cocoEval.params.maxDets = maxDets
    cocoEval.params.useCats = 0
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # Redirect summarize output to variables
    stats = cocoEval.stats  # 12 metrics

    # Save to CSV row
    rows.append([det_file] + list(stats))
    print(list(stats))

# Write CSV
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(rows)

print(f"Done! Results saved to: {output_csv}")
