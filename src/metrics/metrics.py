import torch
import torch.nn.functional as F
from config import cfg
from utils import recur


def RMSE(output, target):
    with torch.no_grad():
        rmse = F.mse_loss(output, target).sqrt().item()
    return rmse


def NDCG(ranklist, K, positive_item):
    if positive_item in ranklist[:K]:
        ranking_of_positive_item = np.where(ranklist == positive_item)[0][0]
        return math.log(2) / math.log(2 + ranking_of_positive_item)
    else:
        return 0


def HitRatio(ranklist, K, positive_item):
    if positive_item in ranklist[:K]:
        return 1
    else:
        return 0


class Metric(object):
    def __init__(self, metric_name):
        self.metric_name = self.make_metric_name(metric_name)
        self.pivot, self.pivot_name, self.pivot_direction = self.make_pivot()
        self.metric = {'Loss': (lambda input, output: output['loss'].item()),
                       'Accuracy': (lambda input, output: recur(Accuracy, output['target'], input['target']))}

    def make_metric_name(self, metric_name):
        return metric_name

    def make_pivot(self):
        if cfg['data_name'] in ['CIFAR10', 'CIFAR100', 'SVHN']:
            pivot = -float('inf')
            pivot_direction = 'up'
            pivot_name = 'Accuracy'
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
