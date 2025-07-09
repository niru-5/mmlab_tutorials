"""This module is used to implement and register the custom data
transformation.

MMCV and OpenMMLab series repositories have provided rich transformation. You
can customize the transformation based on the existed ones.

The default implementation only does the register process. Users need to rename
the ``CustomTransform`` to the real name of the transformation and then
implement it.
"""

try:
    from mmcv.transforms import BaseTransform
except ImportError as e:
    import warnings
    warnings.warn(
        f'mmcv is not installed or cannot be loaded correctly: {e}\n'
        'Using `object` as the base class of the custom transform. and you '
        'must implement the `__call__` method by yourself.')
    BaseTransform = object
import warnings
from mmengine_custom.registry import TRANSFORMS
import numpy as np

@TRANSFORMS.register_module()
class CustomPackClsInputs(BaseTransform):

    def __init__(self,
                 meta_keys=('sample_idx', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data."""
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            # img is a torch tensor
            # img shape is W*H*C
            # change it to C, W, H
            img = img.permute(2, 0, 1)
            # packed_results['imgs'] = img
            packed_results['inputs'] = img
        else:
            raise ValueError("img is not in the results")
        
        if 'img_label' in results:
            packed_results['labels'] = results['img_label']
        else:
            raise ValueError("img_label is not in the results")

        meta_info = {}
        for meta_key in self.meta_keys:
            if meta_key in results:
                meta_info[meta_key] = results[meta_key]
            else:
                warnings.warn(f"{meta_key} is not in the results")
        packed_results['data_samples'] = meta_info

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
