import torch
import torch.nn as nn
from .utils import loss_fn


class LineSearch(nn.Module):
    def __init__(self, num_organizations):
        super().__init__()
        self.assist_rate = nn.Parameter(torch.ones(num_organizations))

    def forward(self, input):
        output = {}
        output['target'] = (input['history'] + self.assist_rate.view(-1, 1) * input['output']).sum(dim=0)
        if 'target' in input:
            output['loss'] = loss_fn(output['target'], input['target'])
        return output


def linesearch():
    num_organizations = cfg['num_organizations']
    model = LineSearch(num_organizations)
    return model
