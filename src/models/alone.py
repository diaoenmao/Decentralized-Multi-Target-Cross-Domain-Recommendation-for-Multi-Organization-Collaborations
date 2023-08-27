import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import loss_fn
from config import cfg


class Alone(nn.Module):
    def __init__(self, model, model_name, match_ratio):
        super().__init__()
        self.model_name = model_name
        self.match_ratio = match_ratio
        self.model_list = nn.ModuleList(model)


    def forward(self, input, m):
        output = self.model_list[m](input)
        return output


def alone(model):
    model_name = cfg['model_name']
    match_ratio = cfg['match_ratio']
    model = Alone(model, model_name, match_ratio)
    return model
