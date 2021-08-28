import math
import torch
import torch.nn.functional as F
from config import cfg
from utils import recur


def RMSE(output, target):
    with torch.no_grad():
        rmse = F.mse_loss(output, target).sqrt().item()
    return rmse


def MAP(output, target, topk=10):
    topk = min(topk, target.size(-1))
    idx = torch.sort(output, dim=-1, descending=True)[1]
    topk_idx = idx[:, :topk]
    topk_target = target[torch.arange(target.size(0), device=output.device).view(-1, 1), topk_idx]
    precision = torch.cumsum(topk_target, dim=-1) / torch.arange(1, topk + 1, device=output.device).float()
    m = torch.sum(topk_target, dim=-1)
    ap = (precision * topk_target).sum(dim=-1) / (m + 1e-10)
    map = ap.mean().item()
    return map


class Metric(object):
    def __init__(self, metric_name):
        self.metric_name = self.make_metric_name(metric_name)
        self.pivot, self.pivot_name, self.pivot_direction = self.make_pivot()
        self.metric = {'Loss': (lambda input, output: output['loss'].item()),
                       'RMSE': (lambda input, output: RMSE(output['target_rating'], input['target_rating'])),
                       'MAP': (lambda input, output: MAP(output['target_rating'], input['target_rating']))}

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
                pivot_name = 'MAP'
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
