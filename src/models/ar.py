import torch
import torch.nn as nn
from .utils import loss_fn
from config import cfg


class AssistRate(nn.Module):
    def __init__(self, num_organizations, lr, factor, milestones, iter):
        super().__init__()
        if milestones is not None:
            exp = (iter > torch.tensor(milestones)).float().sum().item()
            lr = lr * (factor ** exp)
        self.register_buffer('assist_rate', torch.tensor(lr))
        self.stack = nn.Parameter(torch.zeros(num_organizations))

    def forward(self, input):
        output = {}
        output['target'] = input['history'] + self.assist_rate * (input['output'] * self.stack.softmax(-1)).sum(-1)
        # output['target'] = input['history'] + self.assist_rate * input['output'].mean(-1)
        if 'target' in input:
            output['loss'] = loss_fn(output['target'], input['target'])
        return output


def ar(iter):
    num_organizations = cfg['num_organizations']
    lr = cfg['ar']['lr']
    factor = cfg['ar']['factor']
    milestones = cfg['ar']['milestones']
    model = AssistRate(num_organizations, lr, factor, milestones, iter)
    return model
