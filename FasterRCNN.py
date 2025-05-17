# faster_rcnn_pipeline.py
import os
import json
import torch
import torchvision
from PIL import Image, ImageDraw
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision.ops import box_convert, box_iou
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

# Dataset class for COCO format
class CocoDetectionDataset(Dataset):
    def __init__(self, json_file, images_dir, transforms=None):
        with open(json_file) as f:
            coco = json.load(f)

        self.images_dir = images_dir
        self.transforms = transforms

        self.image_id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
        self.id_map = list(self.image_id_to_filename.keys())

        anns_per_image = defaultdict(list)
        for ann in coco['annotations']:
            anns_per_image[ann['image_id']].append(ann)

        self.samples = []
        for img_id in self.id_map:
            anns = anns_per_image[img_id]
            boxes = []
            labels = []
            for ann in anns:
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['category_id'])

            self.samples.append({
                'file_name': self.image_id_to_filename[img_id],
                'bboxes': boxes,
                'labels': labels
            })

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.images_dir, sample['file_name'])
        image = Image.open(img_path).convert("RGB")

        boxes = torch.tensor(sample['bboxes'], dtype=torch.float32)
        labels = torch.tensor(sample['labels'], dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}

        if self.transforms:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.samples)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

def compute_map(model, dataloader):
    model.eval()
    aps = []
    print("\nDetailed validation results:")
    with torch.no_grad():
        for images, targets in dataloader:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for i in range(len(images)):
                gt_boxes = targets[i]['boxes'].to(device)
                pred_boxes = outputs[i]['boxes']
                image_id = targets[i]['image_id'].item()

                if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
                    print(f"Image {image_id}: Skipped (empty boxes)")
                    continue

                ious = box_iou(gt_boxes, pred_boxes)
                max_ious = ious.max(dim=1)[0]

                ap_50 = (max_ious > 0.5).float().mean().item()
                ap_75 = (max_ious > 0.75).float().mean().item()

                print(f"Image {image_id}: GT={len(gt_boxes)}, Pred={len(pred_boxes)}, AP@0.5={ap_50:.2f}, AP@0.75={ap_75:.2f}, Mean IoU={max_ious.mean().item():.2f}")

                aps.append(ap_50)
    return sum(aps) / len(aps) if aps else 0.0

def inference(image_path, model_path, save_path="output.jpg"):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img_tensor)[0]

    draw = ImageDraw.Draw(image)
    for box, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
        if score > 0.5:
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1), f"{label.item()}:{score:.2f}", fill="red")

    image.save(save_path)
    print(f"Saved prediction image to {save_path}")

# ---------- MAIN ----------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    choice = input("Type 'train' to train a model or 'infer' to run inference: ").strip().lower()

    if choice == 'train':
        json_path = input("Enter path to COCO JSON annotation: ").strip()
        images_path = input("Enter path to images folder: ").strip()
        num_epochs = int(input("Enter number of training epochs: ").strip())

        dataset = CocoDetectionDataset(json_path, images_path, transforms=transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        indices = list(range(len(dataset)))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, val_indices)

        train_loader = DataLoader(train_set, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
        val_loader = DataLoader(val_set, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)
        model.to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        model.train()

        losses_list = []
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            for images, targets in tqdm(train_loader):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            loss_value = losses.item()
            losses_list.append(loss_value)
            print(f"Epoch {epoch+1} finished with loss {loss_value:.4f}")

        plt.plot(range(1, len(losses_list)+1), losses_list, marker='o')
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig("training_loss_curve.png")
        plt.close()
        print("Saved training loss curve to training_loss_curve.png")

        torch.save(model.state_dict(), "fasterrcnn.pth")
        map_score = compute_map(model, val_loader)
        print(f"\nFinal Validation mAP@0.5: {map_score:.4f}")

    elif choice == 'infer':
        model_path = input("Enter path to trained model (.pth): ").strip()
        image_path = input("Enter path to image for inference: ").strip()
        inference(image_path, model_path)

    else:
        print("Invalid choice. Type 'train' or 'infer'.")

