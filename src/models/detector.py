import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights


class BaseDetector:
    """Base class for all detectors."""
    
    @classmethod
    def create(cls, num_classes, pretrained=True):
        """
        Create a detector model.
        
        Args:
            num_classes (int): Number of classes (including background)
            pretrained (bool): Whether to use pretrained weights
            
        Returns:
            model: The created model
        """
        raise NotImplementedError("Subclasses must implement create")
    
    @staticmethod
    def load_model(model, model_path, num_classes=None):
        """
        Load model weights from a checkpoint.
        
        Args:
            model: The model to load weights into
            model_path (str): Path to the model checkpoint
            num_classes (int, optional): If provided, adapt the final layers
                                        to match the new number of classes
        
        Returns:
            model: The model with loaded weights
        """
        state_dict = torch.load(model_path)
        
        # If num_classes is provided and different from original model
        if num_classes is not None:
            # Filter out incompatible final layer weights if needed
            if "roi_heads.box_predictor.cls_score.weight" in state_dict:
                orig_cls_weight = state_dict["roi_heads.box_predictor.cls_score.weight"]
                orig_cls_bias = state_dict["roi_heads.box_predictor.cls_score.bias"]
                
                # If the original model had more classes, adapt the weights
                if orig_cls_weight.size(0) > num_classes:
                    state_dict["roi_heads.box_predictor.cls_score.weight"] = torch.cat([
                        orig_cls_weight[0:1],  # Background
                        orig_cls_weight[1:num_classes]  # Keep only needed classes
                    ])
                    state_dict["roi_heads.box_predictor.cls_score.bias"] = torch.cat([
                        orig_cls_bias[0:1],  # Background
                        orig_cls_bias[1:num_classes]  # Keep only needed classes
                    ])
                
                # Similarly for the box regressor
                orig_bbox_weight = state_dict["roi_heads.box_predictor.bbox_pred.weight"]
                orig_bbox_bias = state_dict["roi_heads.box_predictor.bbox_pred.bias"]
                
                if orig_bbox_weight.size(0) > num_classes * 4:
                    state_dict["roi_heads.box_predictor.bbox_pred.weight"] = torch.cat([
                        orig_bbox_weight[0:4],  # Background (should be zeros)
                        orig_bbox_weight[4:num_classes*4]  # Keep only needed classes
                    ])
                    state_dict["roi_heads.box_predictor.bbox_pred.bias"] = torch.cat([
                        orig_bbox_bias[0:4],  # Background (should be zeros)
                        orig_bbox_bias[4:num_classes*4]  # Keep only needed classes
                    ])
        
        # Load state dict with strict=False to ignore missing keys
        model.load_state_dict(state_dict, strict=False)
        return model


class FasterRCNNDetector(BaseDetector):
    """Faster R-CNN detector implementation."""
    
    @classmethod
    def create(cls, num_classes, pretrained=True):
        """
        Create a Faster R-CNN detector.
        
        Args:
            num_classes (int): Number of classes (including background)
            pretrained (bool): Whether to use pretrained weights
            
        Returns:
            model: The Faster R-CNN model
        """
        # Create model with pretrained weights
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_resnet50_fpn(weights=weights)
        
        # Replace the classifier with a new one
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
        
        return model


class ModelFactory:
    """Factory class for creating object detection models."""
    
    @staticmethod
    def create_model(model_type, num_classes, pretrained=True):
        """
        Create a model of the specified type.
        
        Args:
            model_type (str): Model type ("fasterrcnn_resnet50_fpn" supported)
            num_classes (int): Number of classes (including background)
            pretrained (bool): Whether to use pretrained weights
            
        Returns:
            model: The created model
        """
        if model_type == "fasterrcnn_resnet50_fpn":
            return FasterRCNNDetector.create(num_classes, pretrained)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def load_model(model, model_path, num_classes=None):
        """
        Load model weights from a checkpoint.
        
        Args:
            model: The model to load weights into
            model_path (str): Path to the model checkpoint
            num_classes (int, optional): If provided, adapt the final layers
                                        to match the new number of classes
        
        Returns:
            model: The model with loaded weights
        """
        return BaseDetector.load_model(model, model_path, num_classes)