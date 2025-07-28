from torch.optim import Optimizer
import torch
import numpy as np

from mmengine_custom.registry import VISUALIZERS
from mmengine.visualization import Visualizer

@VISUALIZERS.register_module()
class CustomVisualizer(Visualizer):
    # overwrite the add_datasample method to add the data_sample to the visualizer
    def add_datasample(self, name, img, data_sample,
                       draw_gt=True, draw_pred=True,
                       show=False, wait_time=0, step=None):
        # add the image
        self.set_image(img)
        
        # from data_sample get the prediction and ground truth
        predictions, ground_truth = data_sample
        
        # ground truth = data_sample[1][0]
        gt_label_idx = ground_truth[0].item()
        
        # predictions = data_sample[0][0,:]
        pred_scores = predictions[0, :]
        
        # Take the top 5 max values in predictions and use them to get the class labels
        top5_values, top5_indices = torch.topk(pred_scores, 5)
        
        # Format probabilities to 2 decimal places
        top5_probs = [f"{val:.2%}" for val in torch.softmax(top5_values, dim=0)]
        
        # Combine indices with their probabilities
        pred_info = [f"{idx}({prob})" for idx, prob in zip(top5_indices.tolist(), top5_probs)]
        
        # Format the text
        text = f"GT: {gt_label_idx} | Top 5: [{', '.join(pred_info)}]"
        
        # add the text to the image
        self.draw_texts(
            texts=[text],
            positions=np.array([[10, 30]]),  # Position at top-left
            font_sizes=6,
            colors='white',
            bboxes=[dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7)]
        )
        
        # and then send it to vis backend
        if show:
            self.show(wait_time=wait_time)
        
        # Save the visualization
        drawn_img = self.get_image()
        self.add_image(name, drawn_img, step=step)