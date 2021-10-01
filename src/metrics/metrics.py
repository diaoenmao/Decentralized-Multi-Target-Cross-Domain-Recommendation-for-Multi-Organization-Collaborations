import math
import torch
import torch.nn.functional as F
from config import cfg
from utils import recur


def RMSE(output, target):
    with torch.no_grad():
        rmse = F.mse_loss(output, target).sqrt().item()
    return rmse

def Accuracy(output, target):
    with torch.no_grad():
        batch_size = output.size(0)
        p = output.sigmoid()
        pred = torch.stack([1-p, p], dim=-1)
        pred = pred.topk(1, 1, True, True)[1]
        correct = pred.eq(target.long().view(-1, 1).expand_as(pred)).float().sum()
        acc = (correct * (100.0 / batch_size)).item()
    return acc


class Metric(object):
    def __init__(self, metric_name):
        self.metric_name = self.make_metric_name(metric_name)
        self.pivot, self.pivot_name, self.pivot_direction = self.make_pivot()
        self.metric = {'Loss': (lambda input, output: output['loss'].item()),
                       'RMSE': (lambda input, output: RMSE(output['target_rating'], input['target_rating'])),
                       'Accuracy': (lambda input, output: Accuracy(output['target_rating'], input['target_rating']))}

    def make_metric_name(self, metric_name):
        return metric_name

    def make_pivot(self):
        if cfg['data_name'] in ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'NFP']:
            if cfg['target_mode'] == 'explicit':
                pivot = float('inf')
                pivot_direction = 'down'
                pivot_name = 'RMSE'
            elif cfg['target_mode'] == 'implicit':
                pivot = -float('inf')
                pivot_direction = 'up'
                pivot_name = 'Accuracy'
            else:
                raise ValueError('Not valid target mode')
        else:
            raise ValueError('Not valid data name')
        return pivot, pivot_name, pivot_direction

    def evaluate(self, metric_names, input, output):
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output)
        return evaluation

    def compare(self, val):
        if self.pivot_direction == 'down':
            compared = self.pivot > val
        elif self.pivot_direction == 'up':
            compared = self.pivot < val
        else:
            raise ValueError('Not valid pivot direction')
        return compared

    def update(self, val):
        self.pivot = val
        return
