import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import loss_fn
from config import cfg


class POP(nn.Module):
    def __init__(self, num_users, num_items):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.register_buffer('popularity', torch.zeros(self.num_items))

    def forward(self, input):
        output = {}
        user, item, target = input['user'], input['item'], input['target']
        if self.training:
            positive_item = item[torch.nonzero(target)]
            unique_positive_item, counts = torch.unique(positive_item, return_counts=True)
            self.popularity[unique_positive_item] = self.popularity[unique_positive_item] + counts
        pred = self.popularity[item]
        output['target'] = pred
        output['loss'] = loss_fn(pred, target)
        return output


def pop():
    num_users = cfg['num_users']
    num_items = cfg['num_items']
    model = POP(num_users, num_items)
    return model
