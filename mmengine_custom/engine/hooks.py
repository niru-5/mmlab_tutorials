"""This module is used to implement and register the custom Hooks.

The default implementation only does the register process. Users need to rename
the ``CustomHook`` to the real name of the hook and implement it.
"""

from mmengine.hooks import Hook

from mmengine_custom.registry import HOOKS


from mmengine.visualization import Visualizer
from mmengine.runner import Runner
from mmengine.fileio import get
from typing import List
import os.path as osp
import mmcv

@HOOKS.register_module()
class CustomHook(Hook):
    """Subclass of `mmengine.Hook`.

    Warning:
        The class attribute ``priority`` will influence the excutation sequence
        of other hooks.
    """
    priority = 'NORMAL'
    ...



@HOOKS.register_module()
class CustomVisualizationHook(Hook):
# simple visualization hook for custom model. 

    def __init__(self,
                 val_interval: int = 50,
                 test_interval: int = 50):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.val_interval = val_interval
        self.test_interval = test_interval

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs) -> None:


        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx

        

        if total_curr_iter % self.val_interval == 0:
            # Visualize only the first data
            img_path = data_batch['data_samples']['img_path'][0]
            img_bytes = get(img_path, None)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            
            self._visualizer.add_datasample(
                "val_img_iter_" + str(total_curr_iter)+".png",
                img,
                data_sample=outputs,
                draw_gt=True,
                draw_pred=True,
                show=False,
                wait_time=0,
                step=total_curr_iter)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: List) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]): A batch of data samples
                that contain annotations and predictions.
        """
        total_curr_iter = runner.iter + batch_idx
        if total_curr_iter % self.test_interval == 0:
            # for data_sample in outputs:

            img_path = data_batch['data_samples']['img_path'][0]
            img_bytes = get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

            self._visualizer.add_datasample(
                "test_img_iter_" + str(total_curr_iter)+".png",
                img,
                data_sample=outputs,
                draw_gt=True,
                draw_pred=True,
                show=False,
                wait_time=0,
                step=total_curr_iter)