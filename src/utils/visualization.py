import matplotlib.pyplot as plt
import os
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np


class Visualizer:
    """Class for visualization utilities."""
    
    @staticmethod
    def plot_losses(losses, title="Training Losses", save_path=None):
        """
        Plot training losses.
        
        Args:
            losses (list): List of loss values
            title (str): Plot title
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Saved loss plot to {save_path}")
        
        plt.show()
    
    @staticmethod
    def draw_bounding_boxes(image, boxes, labels=None, scores=None, class_names=None, 
                           color=(255, 0, 0), thickness=2, font_size=12):
        """
        Draw bounding boxes on an image.
        
        Args:
            image (PIL.Image or numpy.ndarray): The image to draw on
            boxes (torch.Tensor or list): Bounding boxes in format [x1, y1, x2, y2]
            labels (torch.Tensor or list, optional): Class labels for each box
            scores (torch.Tensor or list, optional): Confidence scores for each box
            class_names (dict, optional): Mapping from class indices to names
            color (tuple): RGB color for the boxes
            thickness (int): Line thickness
            font_size (int): Font size for labels
            
        Returns:
            PIL.Image: The image with bounding boxes drawn
        """
        # Convert to PIL Image if numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Make a copy so we don't modify the original
        output_img = image.copy()
        
        # Create a drawing context
        draw = ImageDraw.Draw(output_img)
        
        # Convert boxes to list if tensor
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy().tolist()
        
        # Convert labels to list if tensor
        if labels is not None and isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy().tolist()
        
        # Convert scores to list if tensor
        if scores is not None and isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy().tolist()
        
        # Draw each box
        for i, box in enumerate(boxes):
            # Get coordinates
            x1, y1, x2, y2 = box
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
            
            # Add label and score if available
            if labels is not None:
                label = labels[i]
                label_text = ""
                
                # Convert label index to name if class_names provided
                if class_names and label in class_names:
                    label_text = class_names[label]
                else:
                    label_text = f"class_{label}"
                
                # Add score if available
                if scores is not None:
                    score = scores[i]
                    label_text += f": {score:.2f}"
                
                # Draw text
                draw.text((x1, y1 - font_size - 4), label_text, fill=color)
        
        return output_img
    
    @staticmethod
    def visualize_dataset_samples(dataset, num_samples=5, class_names=None, figsize=(15, 10)):
        """
        Visualize random samples from a dataset.
        
        Args:
            dataset: The dataset to visualize
            num_samples (int): Number of samples to visualize
            class_names (dict, optional): Mapping from class indices to names
            figsize (tuple): Figure size for the plot
        """
        # Create a figure
        fig, axes = plt.subplots(1, num_samples, figsize=figsize)
        
        # Handle the case where num_samples=1
        if num_samples == 1:
            axes = [axes]
        
        # Generate random indices
        indices = np.random.randint(0, len(dataset), num_samples)
        
        # Plot each sample
        for i, idx in enumerate(indices):
            # Get a sample
            image, target = dataset[idx]
            
            # Convert tensor to PIL Image
            if isinstance(image, torch.Tensor):
                # Convert CxHxW to HxWxC
                image = image.permute(1, 2, 0).cpu().numpy()
                
                # Normalize to [0, 255]
                image = (image * 255).astype(np.uint8)
                
                # Convert to PIL Image
                image = Image.fromarray(image)
            
            # Draw bounding boxes
            image_with_boxes = Visualizer.draw_bounding_boxes(
                image, 
                target["boxes"], 
                target["labels"], 
                class_names=class_names
            )
            
            # Plot
            axes[i].imshow(image_with_boxes)
            axes[i].set_title(f"Sample {idx}")
            axes[i].axis("off")
        
        plt.tight_layout()
        plt.show()