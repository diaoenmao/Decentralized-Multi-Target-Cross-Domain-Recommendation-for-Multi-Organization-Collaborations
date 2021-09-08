import torch
import torch.nn as nn
from .utils import loss_fn
from config import cfg


class AssistRate(nn.Module):
    def __init__(self, lr, factor, milestones, iter):
        super().__init__()
        if milestones is not None:
            exp = (iter > torch(milestones)).float().sum().item()
            lr = factor ** exp
        self.register_buffer('assist_rate', torch.tensor(lr))

    def forward(self, input):
        output = {}
        output['target'] = input['history'] + self.assist_rate * input['output']
        if 'target' in input:
            output['loss'] = loss_fn(output['target'], input['target'])
        return output


def ar(iter):
    lr = cfg['ar']['lr']
    factor = cfg['ar']['factor']
    milestones = cfg['ar']['milestones']
    model = AssistRate(lr, factor, milestones, iter)
    return model
