import torch
import copy
import random
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import os
import numpy as np


class BaseTrainer:
    """Base class for model trainers."""
    
    def __init__(self, model, device="cuda"):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            device (str): Device to use for training ("cuda" or "cpu")
        """
        self.model = model
        self.device = device
        self.model.to(self.device)
    
    def train(self):
        """Train the model. To be implemented by child classes."""
        raise NotImplementedError("Subclasses must implement train")
    
    def save_model(self, output_path):
        """
        Save the model to disk.
        
        Args:
            output_path (str): Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), output_path)
        print(f"Saved model to {output_path}")


class MetaTrainer(BaseTrainer):
    """Trainer for meta-learning on few-shot object detection."""
    
    def __init__(
        self, 
        model, 
        dataset, 
        config, 
        device="cuda"
    ):
        """
        Initialize the meta-trainer.
        
        Args:
            model: The model to train
            dataset: The dataset to train on
            config (dict): Configuration parameters
            device (str): Device to use for training
        """
        super().__init__(model, device)
        self.dataset = dataset
        self.config = config
        
        # Initialize class-to-indices mapping
        self.class_to_indices = self._build_class_to_indices()
        
        # Validate that we have enough examples per class
        self._validate_dataset()
    
    def _build_class_to_indices(self):
        """Build a mapping from classes to sample indices."""
        class_to_indices = defaultdict(list)
        for idx, sample in enumerate(self.dataset.samples):
            for label in set(sample['labels']):
                class_to_indices[label].append(idx)
        return class_to_indices
    
    def _validate_dataset(self):
        """Validate that the dataset is suitable for meta-learning."""
        ways = self.config['meta_learning']['ways']
        shots = self.config['meta_learning']['shots']
        
        # Check if we have enough classes
        if len(self.class_to_indices) < ways:
            raise ValueError(
                f"Dataset only contains {len(self.class_to_indices)} classes, "
                f"but {ways} requested."
            )
        
        # Check if we have enough examples per class
        insufficient = [
            cls for cls, ids in self.class_to_indices.items() 
            if len(ids) < shots + 1
        ]
        if insufficient:
            raise ValueError(
                f"Classes with insufficient samples (require ≥ {shots+1}): {insufficient}"
            )
    
    def _sample_task(self):
        """
        Sample a few-shot task from the dataset.
        
        Returns:
            tuple: (support_indices, query_indices)
        """
        ways = self.config['meta_learning']['ways']
        shots = self.config['meta_learning']['shots']
        
        # Sample N classes
        selected_classes = random.sample(list(self.class_to_indices.keys()), ways)
        
        support_indices, query_indices = [], []
        for cls in selected_classes:
            # Sample K+3 examples per class (K for support, 3 for query)
            indices = random.sample(self.class_to_indices[cls], shots + 3)
            support_indices += indices[:shots]
            query_indices += indices[shots:]
        
        return support_indices, query_indices
    
    def train(self):
        """Train the model using meta-learning."""
        num_epochs = self.config['meta_learning']['num_epochs']
        batch_size = self.config['meta_learning']['batch_size']
        lr = self.config['meta_learning']['lr']
        
        # Use Adam optimizer for meta-learning
        meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Keep track of losses
        losses_list = []
        
        # Meta-training loop
        for epoch in range(num_epochs):
            outer_loss = 0.0
            print(f"Meta-Epoch {epoch+1}/{num_epochs}")
            
            # Sample a task
            support_ids, query_ids = self._sample_task()
            support_set = Subset(self.dataset, support_ids)
            query_set = Subset(self.dataset, query_ids)
            
            # Create data loaders
            support_loader = DataLoader(
                support_set, 
                batch_size=batch_size, 
                shuffle=True, 
                collate_fn=lambda x: tuple(zip(*x))
            )
            
            # Create a copy of the model for inner loop updates
            inner_model = copy.deepcopy(self.model)
            inner_model.train()
            
            # Inner loop optimization with SGD
            inner_optimizer = torch.optim.SGD(inner_model.parameters(), lr=0.01)
            
            # Train on support set
            for images, targets in support_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                loss_dict = inner_model(images, targets)
                if isinstance(loss_dict, list):
                    loss_dict = loss_dict[0]
                
                # Display loss
                print("Outer loop loss_dict:", {
                    k: v.detach().cpu().item() if v.numel() == 1 else v.detach().cpu().mean().item() 
                    for k, v in loss_dict.items()
                })
                
                # Calculate total loss
                loss = sum(
                    v.float().mean() if v.dim() > 0 else v.float() 
                    for v in loss_dict.values()
                )
                
                # Update meta-model
                meta_optimizer.zero_grad()
                loss.backward()
                meta_optimizer.step()
                
                outer_loss += loss.item()
            
            # Record loss for this epoch
            losses_list.append(outer_loss)
            print(f"Epoch {epoch+1} Loss: {outer_loss:.4f}")
        
        return losses_list



class FineTuner(BaseTrainer):
    '''Trainer for fine-tuning a model on a new class.'''
    
    def __init__(
        self, 
        model, 
        dataset, 
        config, 
        device="cuda"
    ):
        '''
        Initialize the fine-tuner.
        
        Args:
            model: The model to fine-tune
            dataset: The dataset to fine-tune on
            config (dict): Configuration parameters
            device (str): Device to use for training
        '''
        super().__init__(model, device)
        self.dataset = dataset
        self.config = config
    
    def train(self):
        '''Fine-tune the model on a new class.'''
        num_epochs = self.config['fine_tuning']['num_epochs']
        batch_size = self.config['fine_tuning']['batch_size']
        lr = self.config['fine_tuning']['lr']
        momentum = self.config['fine_tuning']['momentum']
        
        # Set model to training mode
        self.model.train()
        
        # Create a dataloader for the new class examples
        dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=lambda x: tuple(zip(*x))
        )
        
        # Use SGD with momentum for fine-tuning
        optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=lr, 
            momentum=momentum
        )
        
        print(f"Fine-tuning on {len(self.dataset)} examples for {num_epochs} epochs...")
        
        # Fine-tuning loop
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for images, targets in dataloader:
                # Move data to device
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                loss_dict = self.model(images, targets)
                if isinstance(loss_dict, list):
                    loss_dict = loss_dict[0]
                
                # Calculate total loss
                loss = sum(
                    v.float().mean() if v.dim() > 0 else v.float() 
                    for v in loss_dict.values()
                )
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        
        return self.model



class FineTuner2(BaseTrainer):
    """Trainer for fine-tuning a model on a new class with given ways and shots."""
    
    def __init__(
        self, 
        model, 
        dataset, 
        config, 
        device="cuda"
    ):
        super().__init__(model, device)
        self.dataset = dataset
        self.config = config
        
        # Build class to indices mapping for sampling
        self.class_to_indices = self._build_class_to_indices()
        
        # Read ways and shots from config
        self.ways = config['fine_tuning'].get('ways', 1)
        self.shots = config['fine_tuning'].get('shots', 1)

        
        self._validate_dataset()
    
    def _build_class_to_indices(self):
        class_to_indices = defaultdict(list)
        for idx, sample in enumerate(self.dataset.samples):
            print(sample['labels'])
            for label in set(sample['labels']):
                class_to_indices[label].append(idx)
        return class_to_indices
    
    def _validate_dataset(self):
        if len(self.class_to_indices) < self.ways:
            raise ValueError(
                f"Dataset only has {len(self.class_to_indices)} classes, but {self.ways} ways requested."
            )
        insufficient = [
            cls for cls, ids in self.class_to_indices.items() 
            if len(ids) < self.shots
        ]
        if insufficient:
            raise ValueError(
                f"Classes with insufficient samples (need at least {self.shots}): {insufficient}"
            )
    
    def _sample_few_shot(self):
        # Sample classes and shots per class
        selected_classes = random.sample(list(self.class_to_indices.keys()), self.ways)
        indices = []
        for cls in selected_classes:
            cls_indices = random.sample(self.class_to_indices[cls], self.shots)
            indices.extend(cls_indices)
        return indices
    
    def train(self):
        num_epochs = self.config['fine_tuning']['num_epochs']
        batch_size = self.config['fine_tuning']['batch_size']
        lr = self.config['fine_tuning']['lr']
        momentum = self.config['fine_tuning']['momentum']
        
        self.model.train()
        
        losses = []
        
        for epoch in range(num_epochs):
            # Sample few-shot data for this epoch
            sample_indices = self._sample_few_shot()
            subset = Subset(self.dataset, sample_indices)
            dataloader = DataLoader(
                subset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
            )
            
            epoch_loss = 0.0
            for images, targets in dataloader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = self.model(images, targets)
                if isinstance(loss_dict, list):
                    loss_dict = loss_dict[0]
                
                loss = sum(
                    v.float().mean() if v.dim() > 0 else v.float()
                    for v in loss_dict.values()
                )
                
                optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=lr,
                    momentum=momentum
                )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
            losses.append(epoch_loss)
        
        return self.model
