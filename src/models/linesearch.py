import torch
import torch.nn as nn
from .utils import loss_fn
from config import cfg


class LineSearch(nn.Module):
    def __init__(self, iter, num_organizations):
        super().__init__()
        # self.assist_rate = nn.Parameter(1e-1 * torch.ones(1))
        self.assist_rate = 0.1
        if iter > 5:
            self.assist_rate = 0.01

    def forward(self, input):
        output = {}
        # output['target'] = (input['history'].view(-1, 1) + self.assist_rate * input['output']).mean(dim=-1)
        output['target'] = (input['history'] + self.assist_rate * input['output'])
        if 'target' in input:
            output['loss'] = loss_fn(output['target'], input['target'])
        return output


def linesearch(iter):
    num_organizations = cfg['num_organizations']
    model = LineSearch(iter, num_organizations)
    return model
