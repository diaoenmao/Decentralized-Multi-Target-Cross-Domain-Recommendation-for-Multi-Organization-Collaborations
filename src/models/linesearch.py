import torch
import torch.nn as nn
from .utils import loss_fn
from config import cfg


class LineSearch(nn.Module):
    def __init__(self, num_organizations):
        super().__init__()
        self.assist_rate = nn.Parameter(torch.ones(1))

    def forward(self, input):
        output = {}
        # output['target'] = (input['history'].view(-1, 1) + self.assist_rate * input['output']).mean(dim=-1)
        output['target'] = (input['history'] + self.assist_rate * input['output'].mean(dim=-1))
        if 'target' in input:
            output['loss'] = loss_fn(output['target'], input['target'])
        return output


def linesearch():
    num_organizations = cfg['num_organizations']
    model = LineSearch(num_organizations)
    return model
