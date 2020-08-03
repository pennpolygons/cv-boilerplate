from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
import torch
import pdb
import numpy as np

# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class Mape(Metric):
    def __init__(self, output_transform=lambda x: x):

        self._num_examples = None
        self._sum_percentages = None
        super(Mape, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self._num_examples = 0
        self._sum_percentages = 0
        super(Mape, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output

        errors = torch.abs(y_pred - y.view_as(y_pred))
        errors_divided = errors / y.view_as(y_pred)

        self._num_examples += torch.sum(y > 0.0).item()
        self._sum_percentages += torch.sum(errors_divided[y > 0.0]).item()

    @sync_all_reduce("_num_examples", "_sum_percentages")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                "CustomAccuracy must have at least one example before it can be computed."
            )

        return (self._sum_percentages / self._num_examples) * 100
