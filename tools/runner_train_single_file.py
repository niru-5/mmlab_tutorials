import torch.nn.functional as F
import torchvision
from mmengine.model import BaseModel
import torch.nn as nn
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os   
from torch.optim import SGD
from mmengine.runner import Runner
from mmengine.evaluator import BaseMetric



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

# dataset builder

# build a custom dataset of fruits 360 using torchvision

class Fruits360Dataset(Dataset):
    def __init__(self, meta_file, ann_file, transform=None):
        """
        Args:
            meta_file (str): Path to JSON file containing class information
            ann_file (str): Path to JSON file containing image paths and labels
            transform (callable, optional): Optional transform to be applied on an image
        """
        self.transform = transform
        
        # Load class information
        with open(meta_file, 'r') as f:
            self.classes = json.load(f)
            
        # Load annotations
        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
            self.ann_img_paths = list(self.annotations.keys())
            self.ann_img_idx = list(self.annotations.values())
            

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get annotation for the index
        img_path = self.ann_img_paths[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get class label
        label = self.ann_img_idx[idx]
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
            
        return image, label

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


meta_file_path = "/hdd/side_projects/mmlab_tutorials/data/class_names_to_indices.json"
train_ann_file_path = "/hdd/side_projects/mmlab_tutorials/data/train_list.json"
test_ann_file_path = "/hdd/side_projects/mmlab_tutorials/data/test_list.json"
val_ann_file_path = "/hdd/side_projects/mmlab_tutorials/data/val_list.json"


train_dataset = Fruits360Dataset(meta_file_path, train_ann_file_path, get_default_transforms())
test_dataset = Fruits360Dataset(meta_file_path, test_ann_file_path, get_default_transforms())
val_dataset = Fruits360Dataset(meta_file_path, val_ann_file_path, get_default_transforms())

batch_size = 64
num_classes = len(train_dataset.classes)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



runner = Runner(
    # the model used for training and validation.
    # Needs to meet specific interface requirements
    model=SimpleConvModel(num_classes=num_classes),
    # working directory which saves training logs and weight files
    work_dir='./work_dir',
    # train dataloader needs to meet the PyTorch data loader protocol
    train_dataloader=train_dataloader,
    # optimize wrapper for optimization with additional features like
    # AMP, gradtient accumulation, etc
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.01, momentum=0.9)),
    # trainging coinfs for specifying training epoches, verification intervals, etc
    train_cfg=dict(by_epoch=True, max_epochs=24, val_interval=3),
    # validation dataloaer also needs to meet the PyTorch data loader protocol
    val_dataloader=val_dataloader,
    # validation configs for specifying additional parameters required for validation
    val_cfg=dict(),
    # validation evaluator. The default one is used here
    val_evaluator=dict(type=Accuracy),
)

runner.train()
