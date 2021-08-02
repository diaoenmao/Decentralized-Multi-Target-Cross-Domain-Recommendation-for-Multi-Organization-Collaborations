import math
import torch
import torch.nn.functional as F
from config import cfg
from utils import recur


def RMSE(output, target):
    with torch.no_grad():
        rmse = F.mse_loss(output, target).sqrt().item()
    return rmse


def HR(output, target, topk=10):
    output = output.reshape(-1, 101)
    target = target.reshape(-1, 101)
    sorted, indices = torch.sort(output, dim=-1, descending=True)
    topk_indices = indices[:, :topk]
    topk_target = target[torch.arange(target.size(0)).view(-1, 1), topk_indices]
    hr = torch.any(topk_target, dim=-1).float().mean().item()
    return hr


def NDCG(output, target, topk=10):
    output = output.reshape(-1, 101)
    target = target.reshape(-1, 101)
    sorted, indices = torch.sort(output, dim=-1, descending=True)
    topk_indices = indices[:, :topk]
    topk_target = target[torch.arange(target.size(0)).view(-1, 1), topk_indices]
    nonzero_items = torch.nonzero(topk_target)
    ndcg = output.new_zeros(output.size(0))
    ndcg[nonzero_items[:, 0]] = math.log(2) / torch.log(2 + nonzero_items[:, 1])
    ndcg = ndcg.mean().item()
    return ndcg


def ConfidenceRate(num_confident, num_unknown):
    with torch.no_grad():
        confidence = (num_confident / num_unknown).item()
    return confidence


class Metric(object):
    def __init__(self, metric_name):
        self.metric_name = self.make_metric_name(metric_name)
        self.pivot, self.pivot_name, self.pivot_direction = self.make_pivot()
        self.metric = {'Loss': (lambda input, output: output['loss'].item()),
                       'RMSE': (lambda input, output: RMSE(output['target'], input['target'])),
                       'HR': (lambda input, output: HR(output['target'], input['target'])),
                       'NDCG': (lambda input, output: NDCG(output['target'], input['target'])),
                       'Confidence': (lambda input, output: input['num_confident'].item()),
                       'Confidence Rate': (
                           lambda input, output: ConfidenceRate(input['num_confident'], input['num_unknown']))}

    def make_metric_name(self, metric_name):
        return metric_name

    def make_pivot(self):
        if cfg['data_name'] in ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'NFP']:
            if cfg['data_mode'] == 'explicit':
                pivot = float('inf')
                pivot_direction = 'down'
                pivot_name = 'RMSE'
            elif cfg['data_mode'] == 'implicit':
                pivot = -float('inf')
                pivot_direction = 'up'
                pivot_name = 'HR'
            else:
                raise ValueError('Not valid data mode')
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
