import json

# Paths to original and filtered COCO annotations
original_annotation_path = "annotations_VisDrone_dev.json"
filtered_annotation_path = "filtered_annotations_VisDrone_dev.json"

# Load the original COCO annotations
with open(original_annotation_path, "r") as f:
    coco_data = json.load(f)

# Filter and remap annotations for pedestrian (1) and people (2)
human_category_id = 1
filtered_annotations = []

for ann in coco_data["annotations"]:
    if ann["category_id"] in [1, 2]:  # Keep only pedestrian and people
        ann["category_id"] = human_category_id  # Remap to single human category
        filtered_annotations.append(ann)

# Keep only images that have valid annotations
# valid_image_ids = {ann["image_id"] for ann in filtered_annotations}
# filtered_images = [img for img in coco_data["images"] if img["id"] in valid_image_ids]

# Use all original images
filtered_images = coco_data["images"]

# Update the COCO data with filtered annotations and images
coco_data["annotations"] = filtered_annotations
coco_data["images"] = filtered_images

# Update categories to a single "human" category
coco_data["categories"] = [{"id": human_category_id, "name": "human"}]

# Save the filtered and merged COCO annotations
with open(filtered_annotation_path, "w") as f:
    json.dump(coco_data, f)

print(f"Filtered and merged annotations saved to {filtered_annotation_path}")
