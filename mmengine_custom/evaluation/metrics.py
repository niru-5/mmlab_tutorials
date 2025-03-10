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
