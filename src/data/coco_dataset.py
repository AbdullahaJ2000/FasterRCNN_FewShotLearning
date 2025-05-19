import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict


class BaseDataset(Dataset):
    """Base class for object detection datasets."""
    
    def __init__(self, json_file, images_dir, transforms=None):
        """
        Args:
            json_file (str): Path to the COCO JSON file
            images_dir (str): Path to images directory
            transforms: Optional transformations to apply
        """
        self.images_dir = images_dir
        self.transforms = transforms
        
        # Load COCO annotations
        with open(json_file) as f:
            self.coco_data = json.load(f)
            
        # Initialize dataset-specific attributes
        self._initialize_dataset()
        
    def _initialize_dataset(self):
        """Initialize dataset-specific attributes. To be implemented by child classes."""
        raise NotImplementedError("Subclasses must implement _initialize_dataset")
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        raise NotImplementedError("Subclasses must implement __getitem__")
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)


class CocoDetectionDataset(BaseDataset):
    """Dataset for standard COCO-formatted object detection data."""
    
    def _initialize_dataset(self):
        """Initialize COCO dataset attributes."""
        # Get all category IDs
        self.category_ids = sorted({ann['category_id'] for ann in self.coco_data['annotations']})
        self.class_to_index = {cid: i + 1 for i, cid in enumerate(self.category_ids)}
        
        # Map image IDs to filenames
        self.image_id_to_filename = {img['id']: img['file_name'] for img in self.coco_data['images']}
        self.id_map = list(self.image_id_to_filename.keys())
        
        # Group annotations by image
        anns_per_image = defaultdict(list)
        for ann in self.coco_data['annotations']:
            anns_per_image[ann['image_id']].append(ann)
        
        # Create samples list
        self.samples = []
        for img_id in self.id_map:
            anns = anns_per_image[img_id]
            boxes = []
            labels = []
            
            for ann in anns:
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(self.class_to_index[ann['category_id']])
            
            self.samples.append({
                'file_name': self.image_id_to_filename[img_id],
                'bboxes': boxes,
                'labels': labels
            })
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        sample = self.samples[idx]
        img_path = os.path.join(self.images_dir, sample['file_name'])
        img_path=img_path.replace("\\","/")
        img_path=img_path.replace('"',"")
        image = Image.open(img_path).convert("RGB")
        
        boxes = torch.tensor(sample['bboxes'], dtype=torch.float32)
        labels = torch.tensor(sample['labels'], dtype=torch.int64)
        
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        
        if self.transforms:
            image = self.transforms(image)
        
        return image, target
    
    def get_num_classes(self):
        """Return the number of classes in the dataset (including background)."""
        return len(self.category_ids) + 1  # include background class
    
    def get_class_to_indices(self):
        """Create a mapping from classes to sample indices."""
        class_to_indices = defaultdict(list)
        
        for idx, sample in enumerate(self.samples):
            for label in set(sample['labels']):
                class_to_indices[label].append(idx)
        return class_to_indices


class NewClassDataset(BaseDataset):
    """Dataset for fine-tuning on a new class."""
    
    def __init__(self, json_file, images_dir, transforms=None, target_class_id=None):
        """
        Args:
            json_file (str): Path to the COCO JSON file
            images_dir (str): Path to images directory
            transforms: Optional transformations to apply
            target_class_id (int, optional): The class ID to use for all annotations
        """
        self.target_class_id = target_class_id
        super().__init__(json_file, images_dir, transforms)
    
    def _initialize_dataset(self):
        """Initialize new class dataset attributes."""
        # Determine target class ID if not provided
        if self.target_class_id is None:
            category_ids = sorted({ann['category_id'] for ann in self.coco_data['annotations']})
            if category_ids:
                self.target_class_id = category_ids[0]
            else:
                raise ValueError("No annotations found in the JSON file")
        
        # Map to a single class (class index 1)
        self.class_to_index = {self.target_class_id: 1}
        
        # Map image IDs to filenames
        self.image_id_to_filename = {img['id']: img['file_name'] for img in self.coco_data['images']}
        
        # Group annotations by image
        anns_per_image = defaultdict(list)
        for ann in self.coco_data['annotations']:
            anns_per_image[ann['image_id']].append(ann)
        
        # Create samples list
        self.samples = []
        for img_id, img_file in self.image_id_to_filename.items():
            anns = anns_per_image[img_id]
            boxes = []
            labels = []
            
            for ann in anns:
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['category_id'])  # Always use class index 1 for our new class
            
            self.samples.append({
                'file_name': img_file,
                'bboxes': boxes,
                'labels': labels
            })
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        sample = self.samples[idx]
        img_path = os.path.join(self.images_dir, sample['file_name'])
        img_path= img_path.replace("\\","/")
        image = Image.open(img_path).convert("RGB")
        
        boxes = torch.tensor(sample['bboxes'], dtype=torch.float32)
        labels = torch.tensor(sample['labels'], dtype=torch.int64)
        
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        
        if self.transforms:
            image = self.transforms(image)
        
        return image, target