"""This module is used to patch the default Evaluator in MMEngine.

Follow the `Guide <https://mmeval.readthedocs.io/en/latest/tutorials/custom_metric.html>_`
in MMEval to customize a Metric.

The default implementation only does the register process. Users need to rename
the ``CustomMetric`` to the real name of the metric and implement it.
"""  # noqa: E501

# from mmeval import BaseMetric
from mmengine.evaluator import BaseMetric

from mmengine_custom.registry import METRICS


@METRICS.register_module()
class CustomMetric(BaseMetric):

    def process(self, data_batch, data_samples):
        ...

    # NOTE for evaluator
    def compute_metrics(self, size):
        ...

@METRICS.register_module()
class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        # Calculate top-1 and top-5 accuracy
        pred_scores = score.detach()
        top1_correct = (pred_scores.argmax(dim=1) == gt).sum().cpu()
        
        # For top-5, first ensure we have at least 5 classes
        num_classes = pred_scores.size(1)
        if num_classes >= 5:
            _, top5_pred = pred_scores.topk(5, dim=1)
            top5_correct = sum(gt.unsqueeze(1) == top5_pred).sum().cpu()
        else:
            top5_correct = top1_correct  # If less than 5 classes, top5 = top1
        
        # save the middle result of a batch to `self.results`
        self.results.append({
            'batch_size': len(gt),
            'top1_correct': top1_correct,
            'top5_correct': top5_correct,
        })

    def compute_metrics(self, results):
        total_top1_correct = sum(item['top1_correct'] for item in results)
        total_top5_correct = sum(item['top5_correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        
        # return the dict containing both top-1 and top-5 accuracy
        return {
            'accuracy/top1': 100 * total_top1_correct / total_size,
            'accuracy/top5': 100 * total_top5_correct / total_size
        }
