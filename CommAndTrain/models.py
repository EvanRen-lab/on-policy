#!/usr/bin/env python
"""
DFL Model Definitions
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, mobilenet_v2
import warnings
warnings.filterwarnings('ignore')

# ---------- Linear Models ----------
class LinearClassifier(nn.Module):
    """Pure linear classifier (logistic regression) - suitable for flattened images like CIFAR-10."""
    def __init__(self, input_dim, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        print(f"Initialized linear model: input_dim={input_dim}, num_classes={num_classes}")

    def forward(self, x):
        x = x.view(x.size(0), -1)   # flatten [batch, channels, height, width] -> [batch, input_dim]
        return self.fc(x)


# ---------- MobileNetV2 adapted for CIFAR-10 ----------
class ModifiedMobileNetV2(nn.Module):
    """Modified MobileNetV2 to adapt to 32x32 input of CIFAR-10."""
    def __init__(self, num_classes=10, pretrained=False):
        super().__init__()
        # Load standard MobileNetV2
        self.model = mobilenet_v2(pretrained=pretrained)
        
        # Modify the first convolution layer: original is 3x3 stride=2 padding=1, output 112x112
        # Change to stride=1, padding=1, keep feature map size 32x32 -> 32x32
        self.model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Remove the first downsampling (since stride has been changed to 1, subsequent structure unchanged)
        # MobileNetV2 still has further downsampling, 32->16->8... final feature map size 1x1, which is acceptable
        
        # Modify classifier
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
        
        print(f"Initialized ModifiedMobileNetV2: CIFAR-10 version, pretrained={pretrained}")

    def forward(self, x):
        return self.model(x)



class ModifiedResNet18(nn.Module):
    """Modified ResNet18 adapted to 32x32 input of CIFAR-10."""
    
    def __init__(self, num_classes=10, pretrained=False):
        super(ModifiedResNet18, self).__init__()
        
        # Load pretrained ResNet18
        self.model = resnet18(pretrained=pretrained)
        
        # Modify the first convolution layer to adapt to CIFAR-10 (32x32)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Remove the first max pooling layer (because CIFAR-10 images are small)
        self.model.maxpool = nn.Identity()
        
        # Modify the fully connected layer
        self.model.fc = nn.Linear(512, num_classes)
        
        print(f"Initialized ResNet18: CIFAR-10 version, pretrained={pretrained}")
    
    def forward(self, x):
        return self.model(x)


class ModifiedResNet34(nn.Module):
    """Modified ResNet34 adapted to 32x32 input of CIFAR-10."""
    
    def __init__(self, num_classes=10, pretrained=False):
        super(ModifiedResNet34, self).__init__()
        
        # Load pretrained ResNet34
        self.model = resnet34(pretrained=pretrained)
        
        # Modify the first convolution layer to adapt to CIFAR-10 (32x32)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Remove the first max pooling layer
        self.model.maxpool = nn.Identity()
        
        # Modify the fully connected layer
        self.model.fc = nn.Linear(512, num_classes)
        
        print(f"Initialized ResNet34: CIFAR-10 version, pretrained={pretrained}")
    
    def forward(self, x):
        return self.model(x)


class SimpleCNN(nn.Module):
    """Simple CNN model suitable for fast testing."""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Input: 3x32x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 32x32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 32x32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x16x16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64x16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 64x16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x8x8
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128x8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 128x8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x4x4
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        print(f"Initialized SimpleCNN: {num_classes} classes")
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class LinearSVM(nn.Module):
    """Linear multi-class SVM (using Hinge Loss)."""
    def __init__(self, input_dim, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        print(f"Initialized linear SVM: input_dim={input_dim}, num_classes={num_classes}")

    def forward(self, x):
        x = x.view(x.size(0), -1)   # flatten
        return self.fc(x)            # output raw scores (no softmax)

# ---------- Updated ModelFactory ----------
class ModelFactory:
    @staticmethod
    def create_model(model_name, num_classes=10, pretrained=False, input_dim=None):
        """
        Create a model.

        Args:
            model_name: Model name ('resnet18', 'resnet34', 'simple_cnn', 'linear', 'mobilenet_v2', 'linear_svm')
            num_classes: Number of classes.
            pretrained: Whether to use pretrained weights (only applicable for models that support pretraining).
            input_dim: Input dimension (required only for linear models).
        """
        if model_name == 'resnet18':
            return ModifiedResNet18(num_classes, pretrained)
        elif model_name == 'resnet34':
            return ModifiedResNet34(num_classes, pretrained)
        elif model_name == 'simple_cnn':
            return SimpleCNN(num_classes)
        elif model_name == 'linear':
            if input_dim is None:
                raise ValueError("Linear model requires input_dim to be specified.")
            return LinearClassifier(input_dim, num_classes)
        elif model_name == 'mobilenet_v2':
            return ModifiedMobileNetV2(num_classes, pretrained)
        elif model_name == 'linear_svm':
            if input_dim is None:
                raise ValueError("Linear SVM model requires input_dim to be specified.")
            return LinearSVM(input_dim, num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")