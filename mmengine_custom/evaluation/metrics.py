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
