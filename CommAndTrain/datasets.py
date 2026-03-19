#!/usr/bin/env python
"""
DFL Dataset Processing
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class CIFAR10Partitioner:
    """CIFAR-10 partitioner, supports non-IID partitioning."""
    
    def __init__(self, config):
        self.config = config
        self.num_agents = config.num_agents
        self.weights = config.weights
        
        # Ensure weights sum to 1
        if abs(sum(self.weights) - 1.0) > 1e-6:
            self.weights = [w / sum(self.weights) for w in self.weights]
        
        # Data transforms
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        print(f"CIFAR10 partitioner initialized: {self.num_agents} agents, weights={self.weights}")
    
    def load_datasets(self):
        """Load CIFAR-10 datasets."""
        print("Loading CIFAR-10 dataset...")
        
        train_dataset = datasets.CIFAR10(
            root=self.config.data_dir,
            train=True,
            download=True,
            transform=self.train_transform
        )
        
        test_dataset = datasets.CIFAR10(
            root=self.config.data_dir,
            train=False,
            download=True,
            transform=self.test_transform
        )
        
        print(f"Training set: {len(train_dataset)} images")
        print(f"Test set: {len(test_dataset)} images")
        
        return train_dataset, test_dataset
    
    def partition_data_iid(self, dataset):
        """IID data partitioning."""
        num_samples = len(dataset)
        indices = list(range(num_samples))
        np.random.shuffle(indices)
        
        # Calculate number of samples per agent based on weights
        partition_sizes = [int(w * num_samples) for w in self.weights]
        
        # Handle possible rounding errors
        diff = num_samples - sum(partition_sizes)
        if diff > 0:
            for i in range(diff):
                partition_sizes[i % self.num_agents] += 1
        
        partitions = []
        start_idx = 0
        for size in partition_sizes:
            end_idx = start_idx + size
            partitions.append(indices[start_idx:end_idx])
            start_idx = end_idx
        
        return partitions
    
    def partition_data_non_iid(self, dataset, degree=0.8):
        """
        Non-IID data partitioning.
        
        Args:
            degree: Non-IID degree, between 0 and 1.
                      0: completely IID
                      1: extreme non-IID (each agent has only a few classes)
        """
        # Get class labels
        targets = np.array(dataset.targets)
        num_classes = 10
        
        # For each class, assign to agents according to weights
        class_indices = [np.where(targets == i)[0] for i in range(num_classes)]
        
        # Number of non-IID classes per terminal
        non_iid_classes_per_terminal = max(1, int(num_classes * (1 - degree)))
        if non_iid_classes_per_terminal == num_classes:
            # If non-IID degree is 0, fall back to IID
            return self.partition_data_iid(dataset)
        
        # Initialize data indices for each agent
        terminal_indices = [[] for _ in range(self.num_agents)]
        
        # Allocate data for each class
        for class_idx in range(num_classes):
            class_data = class_indices[class_idx]
            np.random.shuffle(class_data)
            
            # Allocate this class data according to agent weights
            class_weights = self.weights.copy()
            
            # Give more of this class data to a few main holders
            num_main_holders = non_iid_classes_per_terminal
            main_holders = np.random.choice(
                range(self.num_agents),
                size=num_main_holders,
                replace=False
            )
            
            # Adjust weights: main holders get double weight
            for holder in main_holders:
                class_weights[holder] *= 2.0
            
            # Normalize weights
            class_weights = [w / sum(class_weights) for w in class_weights]
            
            # Allocate data
            class_partition_sizes = [int(w * len(class_data)) for w in class_weights]
            
            # Handle possible rounding errors
            diff = len(class_data) - sum(class_partition_sizes)
            if diff > 0:
                for i in range(diff):
                    class_partition_sizes[i % self.num_agents] += 1
            
            start_idx = 0
            for terminal_idx, size in enumerate(class_partition_sizes):
                if size > 0:
                    end_idx = start_idx + size
                    terminal_indices[terminal_idx].extend(class_data[start_idx:end_idx])
                    start_idx = end_idx
        
        # Shuffle each agent's data
        for i in range(self.num_agents):
            np.random.shuffle(terminal_indices[i])
        
        # Analyze partition statistics
        self._analyze_partition(terminal_indices, targets)
        
        return terminal_indices
    
    def _analyze_partition(self, partitions, targets):
        """Analyze data partition results."""
        print("\nData partition analysis:")
        for terminal_idx, indices in enumerate(partitions):
            if len(indices) == 0:
                print(f"  Agent {terminal_idx}: 0 samples")
                continue
            
            terminal_targets = targets[indices]
            unique_classes, class_counts = np.unique(terminal_targets, return_counts=True)
            
            print(f"  Agent {terminal_idx}: {len(indices)} samples")
            print(f"    Class distribution: ", end="")
            for cls, count in zip(unique_classes, class_counts):
                print(f"Class {cls}: {count} ({count/len(indices)*100:.1f}%) ", end="")
            print()
    
    def create_dataloaders(self):
        """Create data loaders for all agents."""
        train_dataset, test_dataset = self.load_datasets()
        
        # Partition training data
        if self.config.data_distribution == 'iid':
            partitions = self.partition_data_iid(train_dataset)
        else:
            partitions = self.partition_data_non_iid(
                train_dataset, 
                degree=self.config.non_iid_degree
            )
        
        # Create training data loaders
        train_loaders = []
        for terminal_idx, indices in enumerate(partitions):
            if len(indices) == 0:
                print(f"Warning: Agent {terminal_idx} has no training data")
                continue
            
            subset = Subset(train_dataset, indices)
            dataloader = DataLoader(
                subset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=2,          # Enable multi-process data loading
                pin_memory=True
            )
            train_loaders.append(dataloader)
        
        # Test loader uses full test dataset
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=2,              # Enable multi-process data loading
            pin_memory=True
        )
        
        return train_loaders, test_loader


class MNISTPartitioner:
    """MNIST partitioner (simplified version)."""
    
    def __init__(self, config):
        self.config = config
        self.num_agents = config.num_agents
        self.weights = config.weights
        
        # Data transforms
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def create_dataloaders(self):
        """Create MNIST data loaders."""
        train_dataset = datasets.MNIST(
            root=self.config.data_dir,
            train=True,
            download=True,
            transform=self.train_transform
        )
        
        test_dataset = datasets.MNIST(
            root=self.config.data_dir,
            train=False,
            download=True,
            transform=self.test_transform
        )
        
        # Simple partitioning (simplified to IID here)
        train_loaders = []
        partition_size = len(train_dataset) // self.num_agents
        
        for i in range(self.num_agents):
            start_idx = i * partition_size
            end_idx = (i + 1) * partition_size if i < self.num_agents - 1 else len(train_dataset)
            
            indices = list(range(start_idx, end_idx))
            subset = Subset(train_dataset, indices)
            
            dataloader = DataLoader(
                subset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=2           # Enable multi-process data loading
            )
            train_loaders.append(dataloader)
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=2               # Enable multi-process data loading
        )
        
        return train_loaders, test_loader


class DatasetFactory:
    """Dataset factory."""
    
    @staticmethod
    def create_partitioner(config):
        """Create data partitioner."""
        if config.dataset_name == 'cifar10':
            return CIFAR10Partitioner(config)
        elif config.dataset_name == 'mnist':
            return MNISTPartitioner(config)
        else:
            raise ValueError(f"Unknown dataset: {config.dataset_name}")