import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import loss_fn, parse_implicit_rating_pair
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
                base = input['rating'].new_zeros(self.num_items, input['rating'].size(0))
                base.scatter_(0, input['item'].view(1, -1), input['rating'].view(1, -1))
                self.base = self.base + base.sum(dim=-1)
                self.count = self.count + (base > 0).float().sum(dim=-1)
            output['target_rating'] = self.base[input['target_item']] / (self.count[input['target_item']] + 1e-10)
            output['target_rating'][self.count[input['target_item']] == 0] = (
                        self.base[self.count != 0] / self.count[self.count != 0]).mean()
            output['loss'] = loss_fn(output['target_rating'], input['target_rating'])
        elif cfg['data_mode'] == 'implicit':
            if self.training:
                base = input['rating'].new_zeros(self.num_items, input['rating'].size(0))
                base.scatter_(0, input['item'].view(1, -1), input['rating'].view(1, -1))
                self.base = self.base + base.sum(dim=-1)
                self.count = self.count + torch.unique(input['user']).size(0)
            output['target_rating'] = self.base[input['target_item']] / self.count[input['target_item']]
            output['loss'] = loss_fn(output['target_rating'], input['target_rating'])
            output['target_rating'], input['target_rating'] = parse_implicit_rating_pair(self.num_items,
                                                                                         input['target_user'],
                                                                                         input['target_item'],
                                                                                         output['target_rating'],
                                                                                         input['target_rating'])
        else:
            raise ValueError('Not valid data mode')
        return output


def base(num_users=None, num_items=None):
    num_users = cfg['num_users']['data'] if num_users is None else num_users
    num_items = cfg['num_items']['data'] if num_items is None else num_items
    model = Base(num_users, num_items)
    return model
