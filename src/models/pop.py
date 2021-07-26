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
        pred = []
        for i in range(len(user)):
            if self.training:
                positive_item = torch.nonzero(target[i])
                self.popularity[positive_item] += 1
            pred_i = self.popularity[item[i]]
            pred.append(pred_i)
        output['target'] = pred
        pred = torch.cat(pred, dim=0)
        target = torch.cat(target, dim=0)
        output['loss'] = loss_fn(pred, target)
        return output


def pop():
    num_users = cfg['num_users']
    num_items = cfg['num_items']
    model = POP(num_users, num_items)
    return model
