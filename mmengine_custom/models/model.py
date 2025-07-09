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


@MODELS.register_module()
class CustomModel(BaseModel):
    ...

@MODELS.register_module()
class SimpleConvModel(BaseModel):
    def __init__(self, num_classes=10, data_preprocessor=None,
                 init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        # size is 128*128*3 
        
        # one conv layer with 3*3 and output is 32 channels, with a stride of 2, with padding 1
        self.conv = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        # add a batchnorm layer
        self.bn = nn.BatchNorm2d(32)
        # add a relu activation function
        self.relu = nn.ReLU()
        
        # output is 64*64*32
        
        # add another conv layer with 3*3 and output is 64 channels, with a stride of 2
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        # add a batchnorm layer
        self.bn2 = nn.BatchNorm2d(64)
        # add a relu activation function
        self.relu2 = nn.ReLU()
        # output is 32*32*64
        
        # add another conv layer with 3*3 and output is 64 channels, with a stride of 2
        self.conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        # add a batchnorm layer
        self.bn3 = nn.BatchNorm2d(64)
        # add a relu activation function
        self.relu3 = nn.ReLU()
        # output is 16*16*64
        
        
        # one maxpool layer with kernel size 2 and stride 2
        self.maxpool = nn.MaxPool2d(4, 4)
        
        
        # one linear layer with 64*4*4 input and 10 output
        self.linear = nn.Linear(64*4*4, num_classes)
        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, inputs, labels, mode, data_samples=None):
        
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.maxpool(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.linear(x)
        
        x = self.softmax(x)
        
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels