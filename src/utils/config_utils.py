import os
import yaml
import random
import torch
import numpy as np


class Config:
    """Class for loading and managing configuration."""
    
    def __init__(self, config_path=None):
        """
        Initialize the configuration.
        
        Args:
            config_path (str, optional): Path to the configuration file
        """
        self.config = {}
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path):
        """
        Load configuration from a YAML file.
        
        Args:
            config_path (str): Path to the configuration file
        """
        # Check if file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load YAML file
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set random seed for reproducibility
        if 'general' in self.config and 'random_seed' in self.config['general']:
            self.set_random_seed(self.config['general']['random_seed'])
    
    def get(self, *keys, default=None):
        """
        Get a value from the configuration using nested keys.
        
        Args:
            *keys: Keys to navigate the configuration dictionary
            default: Default value to return if keys not found
            
        Returns:
            The value from the configuration or the default
        """
        # Start with the full config
        result = self.config
        
        # Navigate through the keys
        for key in keys:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                return default
        
        return result
    
    def update(self, value, *keys):
        """
        Update a value in the configuration using nested keys.
        
        Args:
            value: The new value
            *keys: Keys to navigate the configuration dictionary
        """
        if not keys:
            return
        
        # Navigate to the parent of the leaf node
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Update the leaf node
        config[keys[-1]] = value
    
    def save_config(self, config_path):
        """
        Save the configuration to a YAML file.
        
        Args:
            config_path (str): Path to save the configuration file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Save YAML file
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    @staticmethod
    def set_random_seed(seed):
        """
        Set random seed for reproducibility.
        
        Args:
            seed (int): Random seed
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False