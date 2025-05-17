
import os
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
import time


class Inferencer:
    """Class for running inference with object detection models."""
    
    def __init__(self, model, transform, device="cuda", conf_threshold=0.5):
        """
        Initialize the inferencer.
        
        Args:
            model: The detection model
            transform: Transformation to apply to images
            device (str): Device to use for inference
            conf_threshold (float): Confidence threshold for detections
        """
        self.model = model
        self.model.to(device)
        self.model.eval()
        self.transform = transform
        self.device = device
        self.conf_threshold = conf_threshold
    
    def predict_single_image(self, image_path, draw_on_image=True):
        """
        Run inference on a single image.
        
        Args:
            image_path (str): Path to the image
            draw_on_image (bool): Whether to draw bounding boxes on the image
            
        Returns:
            tuple: (prediction, output_image) if draw_on_image=True, else (prediction, original_image)
        """
        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            prediction = self.model(img_tensor)[0]
        
        if draw_on_image:
            # Create a copy for drawing
            output_img = image.copy()
            draw = ImageDraw.Draw(output_img)
            
            # Draw bounding boxes
            for box, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
                if score > self.conf_threshold:
                    x1, y1, x2, y2 = box.tolist()
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                    
                    # Class 1 is the newly fine-tuned class
                    class_name = "new_class" if label.item() == 1 else f"class_{label.item()}"
                    draw.text((x1, y1 - 10), f"{class_name}: {score:.2f}", fill="red")
            
            return prediction, output_img
        else:
            return prediction, image
    
    def run_inference_on_images(self, image_paths, output_dir, batch_size=1):
        """
        Run inference on a list of images and save the outputs.
        
        Args:
            image_paths (list): List of paths to images
            output_dir (str): Directory to save outputs
            batch_size (int): Number of images to process in a batch
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Track stats
        start_time = time.time()
        total_images = len(image_paths)
        processed_count = 0
        error_count = 0
        
        print(f"Processing {total_images} images...")
        
        # Process images one by one for reliability
        for img_path in tqdm(image_paths, desc="Running inference"):
            try:
                _, output_img = self.predict_single_image(img_path, draw_on_image=True)
                
                # Save the output image
                output_path = os.path.join(output_dir, f"output_{os.path.basename(img_path)}")
                output_img.save(output_path)
                processed_count += 1
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                error_count += 1
        
        # Print summary
        elapsed_time = time.time() - start_time
        print(f"Processed {processed_count} of {total_images} images in {elapsed_time:.2f} seconds")
        if processed_count > 0:
            print(f"Average: {elapsed_time/processed_count:.3f} seconds per image")
        if error_count > 0:
            print(f"Errors: {error_count} images could not be processed")
        print(f"All outputs saved to {output_dir}")


    def _batch_inference(self, image_paths, output_dir, batch_size):
        """
        Run inference in batches for improved performance.
        
        Args:
            image_paths (list): List of paths to images
            output_dir (str): Directory to save outputs
            batch_size (int): Number of images to process in a batch
        """
        from torch.utils.data import Dataset, DataLoader
        
        # Define a simple dataset for loading images
        class ImageDataset(Dataset):
            def __init__(self, image_paths, transform):
                self.image_paths = image_paths
                self.transform = transform
            
            def __len__(self):
                return len(self.image_paths)
            
            def __getitem__(self, idx):
                image_path = self.image_paths[idx]
                image = Image.open(image_path).convert("RGB")
                if self.transform:
                    image_tensor = self.transform(image)
                return image_tensor, image_path, image
        
        # Create dataset and dataloader
        dataset = ImageDataset(image_paths, self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Process batches
        for batch in tqdm(dataloader, desc=f"Processing batches of {batch_size} images"):
            image_tensors, paths, original_images = batch
            
            # Move to device
            image_tensors = image_tensors.to(self.device)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(image_tensors)
            
            # Process each prediction in the batch
            for i, (prediction, img_path, original_image) in enumerate(zip(predictions, paths, original_images)):
                try:
                    # Create a copy for drawing
                    output_img = original_image.copy()
                    draw = ImageDraw.Draw(output_img)
                    
                    # Draw bounding boxes
                    for box, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
                        if score > self.conf_threshold:
                            x1, y1, x2, y2 = box.tolist()
                            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                            
                            # Class 1 is the newly fine-tuned class
                            class_name = "new_class" if label.item() == 1 else f"class_{label.item()}"
                            draw.text((x1, y1 - 10), f"{class_name}: {score:.2f}", fill="red")
                    
                    # Save the output image
                    output_path = os.path.join(output_dir, f"output_{os.path.basename(img_path)}")
                    output_img.save(output_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")