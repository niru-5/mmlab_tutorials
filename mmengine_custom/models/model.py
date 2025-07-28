"""This module is used to implement and register the custom model.

Follow the `guide <https://mmengine.readthedocs.io/en/latest/tutorials/model.html>`
in MMEngine to implement CustomModel

The default implementation only does the register process. Users need to rename
the ``CustomModel`` to the real name of the model and implement it.
"""  # noqa: E501
from mmengine.model import BaseModel

from mmengine_custom.registry import MODELS
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


@MODELS.register_module()
class CustomModel(BaseModel):
    ...

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Residual connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

@MODELS.register_module()
class SimpleConvModel(BaseModel):
    def __init__(self, num_classes=10, data_preprocessor=None,
                 init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        # size is 128*128*3 
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # output: 128*128*64
        
        # First stage - 64 channels
        self.stage1 = nn.Sequential(
            ConvBlock(64, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 128, stride=2)  # downsample
        )
        # output: 64*64*128
        
        # Second stage - 128 channels
        self.stage2 = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 128),
            ConvBlock(128, 256, stride=2)  # downsample
        )
        # output: 32*32*256
        
        # Third stage - 256 channels
        self.stage3 = nn.Sequential(
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 512, stride=2)  # downsample
        )
        # output: 16*16*512
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, inputs, labels, mode, data_samples=None):
        x = self.initial_conv(inputs)
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.classifier(x)
        
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels, ignore_index=0)}
        elif mode == 'predict':
            # Apply softmax only during inference
            return F.softmax(x, dim=1), labels

@MODELS.register_module()
class MobileNetV2Model(BaseModel):
    def __init__(self, num_classes=10, data_preprocessor=None,
                 init_cfg=None, pretrained=False):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        
        # Load pretrained MobileNetV2
        if pretrained:
            self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        else:
            self.backbone = mobilenet_v2(weights=None)
        
        # Replace the classifier with a new one
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, inputs, labels, mode, data_samples=None):
        x = self.backbone(inputs)
        
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels, ignore_index=0)}
        elif mode == 'predict':
            # Apply softmax only during inference
            return F.softmax(x, dim=1), labels