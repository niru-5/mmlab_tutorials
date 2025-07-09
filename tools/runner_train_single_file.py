import torch.nn.functional as F
import torchvision
from mmengine.model import BaseModel
import torch.nn as nn
import json
from torch.utils.data import Dataset, DataLoader
from mmengine.dataset import BaseDataset
from PIL import Image
import os   
from torch.optim import SGD
from mmengine.runner import Runner
from mmengine.evaluator import BaseMetric
import sys
from mmengine.registry import DefaultScope
sys.path.append('/hdd/side_projects/mmlab_tutorials/')
# from mmengine_custom.datasets.datasets import Fruits360Dataset
from mmengine_custom.models.model import SimpleConvModel as CustomModel


# @DATASETS.register_module()
class Fruits360Dataset(BaseDataset):
    def __init__(self, meta_file, ann_file, pipeline):
        # load the meta file here. 
        with open(meta_file, 'r') as f:
            self.meta_info = json.load(f)
        meta_info = {"classes": list(self.meta_info.keys())}
        super().__init__(ann_file=ann_file, metainfo=meta_info, pipeline=pipeline)
        self.classes = list(self.meta_info.keys())
    
    def load_data_list(self):
        # load the ann_file here. 
        with open(self.ann_file, 'r') as f:
            self.annotations = json.load(f)
        self.ann_img_paths = list(self.annotations.keys())
        self.ann_img_idx = list(self.annotations.values())
        data_list = []
        for idx, raw_data_info in enumerate(self.ann_img_paths):
            data_info = {"img_path": raw_data_info, "img_label": self.ann_img_idx[idx]}
            data_list.append(data_info)
        return data_list

class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        # save the middle result of a batch to `self.results`
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        # return the dict containing the eval results
        # the key is the name of the metric name
        return dict(accuracy=100 * total_correct / total_size)


class SimpleConvModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
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
        

    def forward(self, imgs, labels, mode):
        
        x = self.conv(imgs)
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



# Default transforms for training
def get_default_transforms():
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])



pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(128, 128), keep_ratio=False),
            dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            dict(type='ToTensor', keys=['img']),
            dict(type='CustomPackClsInputs'),
        ]

meta_file_path = "/hdd/side_projects/mmlab_tutorials/data/class_names_to_indices.json"
train_ann_file_path = "/hdd/side_projects/mmlab_tutorials/data/train_list.json"
test_ann_file_path = "/hdd/side_projects/mmlab_tutorials/data/test_list.json"
val_ann_file_path = "/hdd/side_projects/mmlab_tutorials/data/val_list.json"


# Set default scope before creating datasets
DefaultScope.get_instance('mmengine_custom', scope_name='mmengine_custom')

# train_dataset = Fruits360Dataset(meta_file_path, train_ann_file_path, get_default_transforms())
# test_dataset = Fruits360Dataset(meta_file_path, test_ann_file_path, get_default_transforms())
# val_dataset = Fruits360Dataset(meta_file_path, val_ann_file_path, get_default_transforms())

train_dataset = Fruits360Dataset(meta_file_path, train_ann_file_path, pipeline)
test_dataset = Fruits360Dataset(meta_file_path, test_ann_file_path, pipeline)
val_dataset = Fruits360Dataset(meta_file_path, val_ann_file_path, pipeline)

batch_size = 64
num_classes = len(train_dataset.classes)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



runner = Runner(
    # the model used for training and validation.
    # Needs to meet specific interface requirements
    model= CustomModel(num_classes=num_classes),
    # working directory which saves training logs and weight files
    work_dir='./work_dir',
    # train dataloader needs to meet the PyTorch data loader protocol
    train_dataloader=train_dataloader,
    # optimize wrapper for optimization with additional features like
    # AMP, gradtient accumulation, etc
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.01, momentum=0.9)),
    # trainging coinfs for specifying training epoches, verification intervals, etc
    train_cfg=dict(by_epoch=True, max_epochs=24, val_interval=1),
    # validation dataloaer also needs to meet the PyTorch data loader protocol
    val_dataloader=val_dataloader,
    # validation configs for specifying additional parameters required for validation
    val_cfg=dict(),
    # validation evaluator. The default one is used here
    val_evaluator=dict(type=Accuracy),
    # default_scope='mmengine_custom'
)

runner.train()
