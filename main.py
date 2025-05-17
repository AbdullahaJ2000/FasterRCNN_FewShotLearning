import os
import sys
import torch
import torchvision
import argparse
import glob
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from src.utils.config_utils import Config
from src.models.detector import ModelFactory
from src.data.coco_dataset import CocoDetectionDataset, NewClassDataset
from src.training.trainers import MetaTrainer, FineTuner
from src.inference.inference import Inferencer
from src.utils.visualization import Visualizer

import os
import sys
import torch
import torchvision
import argparse
import glob
import time
from datetime import datetime
from tqdm import tqdm
from PIL import Image

# Project imports
from src.utils.config_utils import Config
from src.models.detector import ModelFactory
from src.inference.inference import Inferencer


def setup_arg_parser():
    """Set up the argument parser."""
    parser = argparse.ArgumentParser(description='Few-Shot Object Detection')
    
    # General arguments
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['meta-train', 'new-class', 'infer'],
                        help='Operation mode')
    
    # Meta-training arguments
    parser.add_argument('--meta_train_json', type=str,
                        help='Path to COCO JSON for meta-training')
    parser.add_argument('--meta_train_images', type=str,
                        help='Path to images for meta-training')
    
    # New class fine-tuning arguments
    parser.add_argument('--model_path', type=str,
                        help='Path to pretrained model')
    parser.add_argument('--new_class_json', type=str,
                        help='Path to COCO JSON for new class')
    parser.add_argument('--new_class_images', type=str,
                        help='Path to images for new class')
    parser.add_argument('--test_images', type=str,
                        help='Path to test images folder')
    
    # Inference arguments
    parser.add_argument('--image_path', type=str,
                        help='Path to image file or directory containing images for inference')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                        help='Confidence threshold for detections (0.0-1.0)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference (when processing multiple images)')
    
    return parser

def meta_train(config):
    """Run meta-training."""
    # Get configuration values
    num_epochs = config.get('meta_learning', 'num_epochs', default=10)
    ways = config.get('meta_learning', 'ways', default=5)
    shots = config.get('meta_learning', 'shots', default=5)
    json_path = config.get('dataset', 'meta_train', 'json_path')
    images_path = config.get('dataset', 'meta_train', 'images_path')
    device = config.get('general', 'device', default='cuda')
    output_model_path = config.get('paths', 'meta_trained_model')
    
    # Create dataset
    transform = torchvision.transforms.ToTensor()
    dataset = CocoDetectionDataset(json_path, images_path, transforms=transform)
    num_classes = dataset.get_num_classes()
    
    # Create model
    model = ModelFactory.create_model(
        config.get('model', 'backbone', default='fasterrcnn_resnet50_fpn'),
        num_classes
    )
    
    # Create trainer
    trainer = MetaTrainer(model, dataset, config.config, device=device)
    
    # Train model
    print(f"Starting meta-training with {ways}-way, {shots}-shot learning for {num_epochs} epochs...")
    losses = trainer.train()
    
    # Save model
    trainer.save_model(output_model_path)
    
    # Plot losses
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Visualizer.plot_losses(
        losses, 
        title="Meta-Training Losses", 
        save_path=os.path.join(config.get('paths', 'output_dir', default='outputs'), f"meta_losses_{timestamp}.png")
    )



def fine_tune_new_class(config):
    """Fine-tune on a new class."""
    # Get configuration values
    model_path = config.get('paths', 'meta_trained_model')
    new_class_json = config.get('dataset', 'new_class', 'json_path')
    new_class_images = config.get('dataset', 'new_class', 'images_path')
    test_images_folder = config.get('dataset', 'new_class', 'test_images_folder')
    num_fine_tune_epochs = config.get('fine_tuning', 'num_epochs', default=20)
    device = config.get('general', 'device', default='cuda')
    output_model_path = config.get('paths', 'fine_tuned_model')
    conf_threshold = config.get('inference', 'conf_threshold', default=0.5)
    output_dir = os.path.join(config.get('paths', 'output_dir', default='outputs'), 'new_class_results')
    
    # Number of background classes + 1 new class
    num_classes = 2  # Background (0) + new class (1)
    
    # Create model
    meta_model = ModelFactory.create_model(
        config.get('model', 'backbone', default='fasterrcnn_resnet50_fpn'),
        num_classes
    )
    
    # Load the pre-trained weights
    meta_model = ModelFactory.load_model(meta_model, model_path, num_classes=num_classes)
    print("Loaded meta-trained model weights")
    
    # Create dataset for the new class
    transform = torchvision.transforms.ToTensor()
    new_class_dataset = NewClassDataset(
        new_class_json, 
        new_class_images,
        transforms=transform
    )
    
    print(f"Created dataset with {len(new_class_dataset)} examples of the new class")
    
    # Fine-tune the model on the new class
    fine_tuner = FineTuner(meta_model, new_class_dataset, config.config, device=device)
    fine_tuned_model = fine_tuner.train()
    
    # Save the fine-tuned model
    fine_tuner.save_model(output_model_path)
    
  
    # Get paths of all test images
    test_image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']:
        test_image_paths.extend(glob.glob(os.path.join(test_images_folder, ext)))
    
    if not test_image_paths:
        print(f"No images found in {test_images_folder}")
    else:
        print(f"Found {len(test_image_paths)} test images")
        
        # Create a specific output directory for inference results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        inference_output_dir = os.path.join(output_dir, f"new_class_results_{timestamp}")
        os.makedirs(inference_output_dir, exist_ok=True)
        
        # Run inference on the test images
        inferencer = Inferencer(
            fine_tuned_model,
            transform,
            device=device,
            conf_threshold=conf_threshold
        )
        
        inferencer.run_inference_on_images(test_image_paths, inference_output_dir)


def run_inference(config):
    """Run inference on a single image or a directory of images."""
    # Get configuration values
    model_path = config.get('paths', 'fine_tuned_model')
    device = config.get('general', 'device', default='cuda')
    conf_threshold = config.get('inference', 'conf_threshold', default=0.5)
    image_path = config.get('inference', 'image_path')
    output_dir = config.get('paths', 'output_dir', default='outputs')
    batch_size = config.get('inference', 'batch_size', default=1)
    
    print(f"Starting inference with path: {image_path}")
    
    # Check if image_path exists
    if not os.path.exists(image_path):
        print(f"Error: The path '{image_path}' does not exist")
        return
    
    # Debug information about the path
    print(f"Image path is a directory: {os.path.isdir(image_path)}")
    print(f"Image path is a file: {os.path.isfile(image_path)}")
    
    # Load the state_dict to examine it
    state_dict = torch.load(model_path, map_location=device)
    
    # Determine number of classes from the model weights
    if 'roi_heads.box_predictor.cls_score.weight' in state_dict:
        cls_weight = state_dict['roi_heads.box_predictor.cls_score.weight']
        num_classes = cls_weight.size(0)
        print(f"Detected {num_classes} classes in model file")
    else:
        # Default to config value or 2 if we can't determine
        num_classes = config.get('model', 'num_classes', default=2)
        print(f"Using configured number of classes: {num_classes}")
    
    # Create model with correct number of classes
    model = ModelFactory.create_model(
        config.get('model', 'backbone', default='fasterrcnn_resnet50_fpn'),
        num_classes
    )
    
  
    # Option 1: If load_model is a method of ModelFactory
    model = ModelFactory.load_model(model, model_path)
  
    
    print(f"Loaded model from {model_path}")
    
    # Create transform
    transform = torchvision.transforms.ToTensor()
    
    # Create inferencer
    inferencer = Inferencer(
        model,
        transform,
        device=device,
        conf_threshold=conf_threshold
    )
    
    # Process differently based on whether image_path is a directory or file
    if os.path.isdir(image_path):
        print(f"Processing directory: {image_path}")
        # Get all image files in the directory
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']:
            found_files = glob.glob(os.path.join(image_path, ext))
            print(f"Found {len(found_files)} files with extension {ext}")
            image_paths.extend(found_files)
        
        if not image_paths:
            print(f"No images found in {image_path}")
            return
        
        print(f"Found {len(image_paths)} images in total")
        
        # Create a specific output directory for this batch
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_output_dir = os.path.join(output_dir, f"inference_batch_{timestamp}")
        os.makedirs(batch_output_dir, exist_ok=True)
        
        # Process each image individually to avoid any batch processing issues
        processed_count = 0
        error_count = 0
        start_time = time.time()
        
        for img_path in tqdm(image_paths, desc="Processing images"):
            try:
                _, output_img = inferencer.predict_single_image(img_path, draw_on_image=True)
                output_filename = f"output_{os.path.basename(img_path)}"
                output_path = os.path.join(batch_output_dir, output_filename)
                output_img.save(output_path)
                processed_count += 1
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                error_count += 1
        
        elapsed_time = time.time() - start_time
        print(f"Processed {processed_count} images with {error_count} errors in {elapsed_time:.2f} seconds")
        print(f"Results saved to {batch_output_dir}")
    
    elif os.path.isfile(image_path):
        print(f"Processing single image: {image_path}")
        try:
            _, output_img = inferencer.predict_single_image(image_path, draw_on_image=True)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"inference_output_{os.path.basename(image_path)}")
            output_img.save(output_path)
            print(f"Saved output to {output_path}")
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
    
    else:
        print(f"Error: '{image_path}' is neither a valid file nor a directory")
        
def main():
    """Main entry point."""
    # Parse arguments
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Override config with command line arguments
    if args.meta_train_json:
        config.update(args.meta_train_json, 'dataset', 'meta_train', 'json_path')
    if args.meta_train_images:
        config.update(args.meta_train_images, 'dataset', 'meta_train', 'images_path')
    if args.model_path:
        config.update(args.model_path, 'paths', 'meta_trained_model')
    if args.new_class_json:
        config.update(args.new_class_json, 'dataset', 'new_class', 'json_path')
    if args.new_class_images:
        config.update(args.new_class_images, 'dataset', 'new_class', 'images_path')
    if args.test_images:
        config.update(args.test_images, 'dataset', 'new_class', 'test_images_folder')
        
    # Add this block for image_path - Point 5
    if args.image_path:
        # Convert to absolute path
        image_path = os.path.abspath(args.image_path)
        print(f"Using absolute image path: {image_path}")
        config.update(image_path, 'inference', 'image_path')
    # End of added block
        
    if args.conf_threshold:
        config.update(args.conf_threshold, 'inference', 'conf_threshold')
    if hasattr(args, 'batch_size') and args.batch_size:
        config.update(args.batch_size, 'inference', 'batch_size')
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.update(str(device), 'general', 'device')
    print(f"Using device: {device}")
    
    # Run the specified mode
    if args.mode == 'meta-train':
        meta_train(config)
    elif args.mode == 'new-class':
        fine_tune_new_class(config)
    elif args.mode == 'infer':
        run_inference(config)
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()