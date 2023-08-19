import torch
import torch.nn as nn
from .utils import loss_fn
from config import cfg


class Assist(nn.Module):
    def __init__(self, model, model_name, match_ratio):
        super().__init__()
        self.match_ratio = match_ratio
        model_list = []
        for m in range(len(model)):
            model_list.append(model[m])
        self.model_list = nn.ModuleList(model_list)

    def sync(self, data_loader):
        return

    def forward(self, input, m):
        output = self.model_list[m](input)
        return output


def assist(model):
    model_name = cfg['model_name']
    match_ratio = cfg['match_ratio']
    model = Assist(model, model_name, match_ratio)
    return model
