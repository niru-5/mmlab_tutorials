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
import pandas as pd
import os

@DATASETS.register_module()
class CUB200_2011Dataset(BaseDataset):
    """CUB-200-2011 Dataset class compatible with MMEngine.
    
    This dataset class handles the Caltech-UCSD Birds-200-2011 dataset, which contains
    11,788 images of 200 bird species. The class inherits from MMEngine's BaseDataset
    for seamless integration with MMEngine frameworks.
    
    Args:
        ann_file (str): Path to the annotation file in CSV format containing image paths
            and class IDs.
        pipeline (list): List of data transformations to apply.
        data_root (str): Root directory of the dataset containing images and metadata.
    
    Attributes:
        img_class_id (list): List of class IDs for each image.
        img_paths (list): List of relative image paths.
        data_root (str): Root directory of the dataset.
        meta_info (dict): Dictionary mapping class IDs to class names.
        classes (list): List of class IDs.
    """
    
    def __init__(self, ann_file, pipeline, data_root):
        # load the meta file here. 
        df = pd.read_csv(ann_file)
        self.img_class_id = df['img_class_id'].to_list()
        self.img_paths = df['img_path'].to_list()
        self.data_root = data_root
        self.meta_info = {}
        with open(os.path.join(self.data_root, "classes.txt"), "r") as f:
            class_names = f.readlines()
            for idx, class_name in enumerate(class_names):
                tokens = class_name.strip().split()
                self.meta_info[int(tokens[0])] = tokens[1]
        
        meta_info = {"classes": list(self.meta_info.values())}
        super().__init__(ann_file=ann_file, metainfo=meta_info, pipeline=pipeline,
                         data_root=data_root)
        self.classes = list(self.meta_info.keys())
    
    def load_data_list(self):
        """Load data list from annotation file.
        
        Returns:
            list[dict]: List of data info dictionaries. Each dictionary contains:
                - img_path (str): Absolute path to the image.
                - img_label (int): Class label for the image.
        """
        data_list = []
        for idx, raw_data_info in enumerate(self.img_paths):
            data_info = {"img_path": os.path.join(self.data_root, "images", raw_data_info), "img_label": self.img_class_id[idx]}
            data_list.append(data_info)
        return data_list

@DATASETS.register_module()
class Fruits360Dataset(BaseDataset):
    """Fruits 360 Dataset class compatible with MMEngine.
    
    This dataset class handles the Fruits 360 dataset, which contains images of various
    fruits and vegetables. The class inherits from MMEngine's BaseDataset for seamless
    integration with MMEngine frameworks.
    
    Args:
        meta_file (str): Path to the JSON file containing class information.
        ann_file (str): Path to the JSON file containing image annotations.
        pipeline (list): List of data transformations to apply.
    
    Attributes:
        meta_info (dict): Dictionary containing class information.
        classes (list): List of class names.
        annotations (dict): Dictionary mapping image paths to class indices.
        ann_img_paths (list): List of image paths.
        ann_img_idx (list): List of class indices.
    """
    
    def __init__(self, meta_file, ann_file, pipeline):
        # load the meta file here. 
        with open(meta_file, 'r') as f:
            self.meta_info = json.load(f)
        meta_info = {"classes": list(self.meta_info.keys())}
        super().__init__(ann_file=ann_file, metainfo=meta_info, pipeline=pipeline)
        self.classes = list(self.meta_info.keys())
    
    def load_data_list(self):
        """Load data list from annotation file.
        
        Returns:
            list[dict]: List of data info dictionaries. Each dictionary contains:
                - img_path (str): Path to the image.
                - img_label (int): Class label for the image.
        """
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
    

@DATASETS.register_module()
class Fruits360DatasetTorch(Dataset):
    """Fruits 360 Dataset class compatible with PyTorch.
    
    This dataset class provides a PyTorch-native implementation for the Fruits 360 dataset.
    It implements the standard PyTorch Dataset interface and can be used with PyTorch
    DataLoader for efficient data loading.
    
    Args:
        meta_file (str): Path to JSON file containing class information.
        ann_file (str): Path to JSON file containing image paths and labels.
        transform (callable, optional): Optional transform to be applied on an image.
    
    Attributes:
        classes (dict): Dictionary containing class information.
        annotations (dict): Dictionary mapping image paths to class indices.
        ann_img_paths (list): List of image paths.
        ann_img_idx (list): List of class indices.
        transform (callable, optional): Transform to be applied to images.
    """
    
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
        """Return the total number of images in the dataset.
        
        Returns:
            int: Total number of images.
        """
        return len(self.annotations)

    def __getitem__(self, idx):
        """Get a single data sample.
        
        Args:
            idx (int): Index of the sample to fetch.
            
        Returns:
            tuple: (image, label) where image is the transformed PIL Image and
                label is the class index.
        """
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