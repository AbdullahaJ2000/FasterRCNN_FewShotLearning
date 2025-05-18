import json

with open('C:/Users/User/Desktop/few_shot_detection/kaggle/input/oxalis-coco/_annotations.coco.json', 'r') as f:
    data = json.load(f)

# نأخذ أول 10 صور فط
first_10_images = data['images'][:10]

# نحصل على الـids الخاصة بهذه الصور
first_10_image_ids = set(img['id'] for img in first_10_images)

# نفلتر التعليقات التي تخص هذه الصور فقط
filtered_annotations = [ann for ann in data['annotations'] if ann['image_id'] in first_10_image_ids]

# نعيد بناء البيانات مع الصور والتعليقات المفلترة
new_data = {
    "info": data.get("info", {}),
    "licenses": data.get("licenses", []),
    "categories": data.get("categories", []),
    "images": first_10_images,
    "annotations": filtered_annotations
}

# حفظ الملف الجديد
with open('C:/Users/User/Desktop/few_shot_detection/kaggle/input/oxalis-coco/_annotations_first_10.coco.json', 'w') as f:
    json.dump(new_data, f, indent=4)

print("تم إنشاء ملف JSON يحتوي أول 10 صور فقط: _annotations_first_10.coco.json")
