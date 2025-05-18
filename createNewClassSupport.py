import json

with open('C:/Users/User/Desktop/few_shot_detection/kaggle/input/oxalis-coco/_annotations.coco.json', 'r') as f:
    data = json.load(f)


first_10_images = data['images'][:10]


first_10_image_ids = set(img['id'] for img in first_10_images)


filtered_annotations = [ann for ann in data['annotations'] if ann['image_id'] in first_10_image_ids]

new_data = {
    "info": data.get("info", {}),
    "licenses": data.get("licenses", []),
    "categories": data.get("categories", []),
    "images": first_10_images,
    "annotations": filtered_annotations
}


with open('C:/Users/User/Desktop/few_shot_detection/kaggle/input/oxalis-coco/_annotations_first_10.coco.json', 'w') as f:
    json.dump(new_data, f, indent=4)
