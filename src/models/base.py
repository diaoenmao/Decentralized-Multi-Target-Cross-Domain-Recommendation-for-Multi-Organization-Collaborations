import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import loss_fn
from config import cfg


class Base(nn.Module):
    def __init__(self, num_users, num_items):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.register_buffer('base', torch.zeros(self.num_items))
        self.register_buffer('count', torch.zeros(self.num_items))

    def forward(self, input):
        output = {}
        if cfg['data_mode'] == 'explicit':
            if self.training:
                base = input['data_rating'].new_zeros(self.num_items, input['data_rating'].size(0))
                base.scatter_(0, input['data_item'].view(1, -1), input['data_rating'].view(1, -1))
                self.base = self.base + base.sum(dim=-1)
                self.count = self.count + (base > 0).float().sum(dim=-1)
            output['target_rating'] = self.base[input['target_item']] / (self.count[input['target_item']] + 1e-10)
            output['loss'] = loss_fn(output['target_rating'], input['target_rating'])
        elif cfg['data_mode'] == 'implicit':
            if self.training:
                base = input['data_rating'].new_zeros(self.num_items, input['data_rating'].size(0))
                base.scatter_(0, input['data_item'].view(1, -1), input['data_rating'].view(1, -1))
                self.base = self.base + base.sum(dim=-1)
                self.count = self.count + torch.unique(input['data_user']).size(0)
            output['target_rating'] = self.base[input['target_item']] / self.count[input['target_item']]
            output['loss'] = loss_fn(output['target_rating'], input['target_rating'])
        else:
            raise ValueError('Not valid data mode')
        return output


def base():
    num_users = cfg['num_users']
    num_items = cfg['num_items']
    model = Base(num_users, num_items)
    return model
