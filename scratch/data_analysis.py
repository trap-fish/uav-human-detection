import json
from collections import defaultdict
import matplotlib.pyplot as plt


# List the paths to your COCO annotation files
files = [
    '/media/citi-ai/matthew/uav-human-detection/cocoformat_annotations_VisDroneHumans_train.json',
    '/media/citi-ai/matthew/uav-human-detection/cocoformat_annotations_VisDroneHumans_val.json',
    '/media/citi-ai/matthew/uav-human-detection/cocoformat_annotations_VisDroneHumans_test-dev.json'
]

# Initialize counters
total_images = 0
categories = {}
annotations_count = defaultdict(int)

# Parse each annotation file
for file_path in files:
    with open(file_path, 'r') as f:
        data = json.load(f)

    images = data.get('images', [])
    total_images += len(images)

    for cat in data.get('categories', []):
        if cat['id'] <3:  # Exclude background
            categories[cat['id']] = cat['name']

    for ann in data.get('annotations', []):
        if ann['category_id'] <3:  # Exclude background
            annotations_count[ann['category_id']] += 1

# Prepare data for plotting
category_names = [categories[cat_id] for cat_id in sorted(categories.keys())]
counts = [annotations_count[cat_id] for cat_id in sorted(categories.keys())]

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(category_names, counts, color='skyblue')
plt.xlabel("Category")
plt.ylabel("Annotation Count")
plt.title("Label Counts per Category (excluding background)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('visdrone_labels.png')
