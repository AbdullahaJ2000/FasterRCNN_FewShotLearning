# Few-Shot Object Detection with Faster R-CNN

This project implements a meta-learning approach for few-shot object detection using Faster R-CNN. It enables the model to learn new object classes with very few training examples by leveraging meta-learning techniques.

## Overview

The system consists of two main phases:
1. **Generalization Phase (Meta-Training)**: The model learns to quickly adapt to new object classes by learning general patterns and features that are transferable across different object categories
2. **Adaptation Phase (Few-Shot Fine-tuning)**: The generalized model is adapted to recognize new classes with very few examples (typically 1-5 shots)

## Features

- Meta-learning based few-shot object detection
- Support for COCO format datasets
- Easy adaptation to new classes with minimal examples
- Inference capabilities with visualization

## Requirements

- Python 3.8+
- PyTorch 2.7.0
- CUDA 11.8 (for GPU support)
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
FasterRCNN_FewShotLearning/
├── config/             # Configuration files
├── models/            # Model definitions
├── src/               # Source code
│   ├── data/         # Dataset handling
│   ├── models/       # Model implementations
│   ├── training/     # Training logic
│   ├── inference/    # Inference code
│   └── utils/        # Utility functions
├── outputs/          # Training outputs and results
├── NewClassData/     # Data for new classes
├── main.py           # Main entry point
└── requirements.txt  # Project dependencies
```

## Usage

### 1. Generalization Phase (Meta-Training)

Train the model to learn generalizable features:

```bash
python main.py --mode meta-train --config config/config.yaml
```

### 2. Adaptation Phase (Few-Shot Fine-tuning)

Adapt the generalized model to new classes:

```bash
python main.py --mode new-class --config config/config.yaml
```

### 3. Inference

Run inference on new images:

```bash
python main.py --mode infer --config config/config.yaml
```

## Configuration

The project uses YAML configuration files to manage parameters. Here are the required parameters for each mode:

### General Settings (All Modes)
```yaml
general:
  device: "cuda"  # or "cpu"
  random_seed: 42
```

### 1. Generalization Phase (Meta-Training)
```yaml
paths:
  meta_trained_model: "models/fasterrcnn_meta_trained.pth"

dataset:
  meta_train:
    json_path: "path/to/coco/annotations.json"
    images_path: "path/to/training/images"

model:
  backbone: "fasterrcnn_resnet50_fpn"

meta_learning:
  num_epochs: 50
  ways: 20  # N-way classification
  shots: 15  # K-shot learning
  batch_size: 2
  lr: 0.0001
```

### 2. Adaptation Phase (Few-Shot Fine-tuning)
```yaml
paths:
  meta_trained_model: "models/fasterrcnn_meta_trained.pth"

dataset:
  new_class:
    json_path: "path/to/new_class/annotations.json" #support data annotation
    images_path: "path/to/new_class/images" #support data images
    test_images_folder: "path/to/test/images" #query data images

fine_tuning:
  num_epochs: 10
  batch_size: 2
  lr: 0.005
  momentum: 0.9
  num_classes: 12
```

### 3. Inference Mode
```yaml
paths:
  fine_tuned_model: "models/fasterrcnn_finetuned_new_class.pth"

inference:
  image_path: "path/to/image.jpg"  # Can be a single image or directory
  conf_threshold: 0.5
  batch_size: 1
  save_visualizations: true
```

### Important Notes:
1. For Generalization Phase:
   - Requires COCO format dataset
   - `ways` and `shots` parameters determine the few-shot learning setup
   - Higher `ways` and `shots` values require more GPU memory
   - Focuses on learning transferable features across different object categories

2. For Adaptation Phase:
   - Requires new class data in COCO format
   - `num_classes` should match your new class dataset
   - Support data (few examples) is used for adaptation
   - Query data without annotation

3. For inference:
   - Can process single images or directories
   - Adjust `conf_threshold` to control detection confidence
   - Set `save_visualizations` to save annotated images

## How It Works

1. **Generalization Phase**:
   - The model learns to quickly adapt to new object classes
   - Uses episodic training with N-way K-shot learning
   - Learns general patterns and features that are transferable across different object categories
   - Leverages Faster R-CNN architecture with meta-learning capabilities

2. **Adaptation Phase**:
   - Takes the generalized model
   - Adapts to new classes using very few examples (support data)
   - Evaluates performance on unseen examples (query data)

3. **Inference**:
   - Performs object detection on new images
   - Supports batch processing
   - Visualizes detection results

## Performance

The system is designed to achieve good detection performance even with very few training examples.

