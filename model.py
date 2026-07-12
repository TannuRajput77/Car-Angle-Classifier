"""
model.py
ResNet34 with transfer learning for car angle classification.

Why ResNet34 (interview talking point):
- Pretrained on ImageNet -> already knows edges, textures, shapes
- Residual connections solve vanishing gradient -> can go deeper than plain CNN
- 34 layers is a good balance: enough capacity for angle classification,
  not as heavy as ResNet50/101 -> faster training/inference, good for
  a cataloguing pipeline that needs to process images at scale
"""

import torch.nn as nn
from torchvision import models


def build_model(num_classes, freeze_backbone=True):
    """
    freeze_backbone=True -> only train the final classifier layer (faster,
    less data needed, good when your angle dataset is small)
    freeze_backbone=False -> fine-tune the whole network (better accuracy
    if you have a large enough dataset, but slower and can overfit on
    small data)
    """
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model