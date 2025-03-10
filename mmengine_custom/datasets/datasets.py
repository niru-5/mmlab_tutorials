"""This module is used to implement and register the custom datasets.

If OpenMMLab series repositries have supported the target dataset, for example,
CocoDataset. You can simply use it by setting ``type=mmdet.CocoDataset`` in the
config file.

If you want to do some small modifications to the existing dataset,
you can inherit from it and override its methods:

Examples:
    >>> from mmdet.datasets import CocoDataset as MMDetCocoDataset
    >>>
    >>> class CocoDataset(MMDetCocoDataset):
    >>>     def load_data_list(self):
    >>>         ...

Don't worry about the duplicated name of the custom ``CocoDataset`` and the
mmdet ``CocoDataset``, they are registered into different registry nodes.

The default implementation only does the register process. Users need to rename
the ``CustomDataset`` to the real name of the target dataset, for example,
``WiderFaceDataset``, and then implement it.
"""

from mmengine.dataset import BaseDataset

from mmengine_custom.registry import DATASETS
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image



@DATASETS.register_module()
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
    


class Fruits360DatasetTorch(Dataset):
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