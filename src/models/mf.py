import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import loss_fn
from config import cfg


class MF(nn.Module):
    def __init__(self, num_users, num_items, hidden_size, info_size):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.info_size = info_size
        self.user_weight = nn.Embedding(num_users, hidden_size)
        self.item_weight = nn.Embedding(num_items, hidden_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.bias = nn.Parameter(torch.randn(1))
        if self.info_size is not None:
            self.user_profile = nn.Linear(info_size['user_profile'], hidden_size)
            self.item_attr = nn.Linear(info_size['item_attr'], hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.user_weight.weight, 0.0, 0.01)
        nn.init.normal_(self.item_weight.weight, 0.0, 0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.zeros_(self.bias)
        if self.info_size is not None:
            nn.init.normal_(self.user_profile.weight, 0.0, 0.01)
            nn.init.zeros_(self.user_profile.bias)
            nn.init.normal_(self.item_attr.weight, 0.0, 0.01)
            nn.init.zeros_(self.item_attr.bias)
        return

    def user_embedding(self, user):
        embedding = self.user_weight(user) + self.user_bias(user)
        return embedding

    def item_embedding(self, item):
        embedding = self.item_weight(item) + self.item_bias(item)
        return embedding

    def forward(self, input):
        output = {}
        if self.training:
            user = input['data_user']
            item = input['data_item']
            rating = input['data_rating']
            if self.info_size is not None:
                user_profile = input['data_user_profile']
                item_attr = input['data_item_attr']
        else:
            user = input['target_user']
            item = input['target_item']
            rating = input['target_rating']
            if self.info_size is not None:
                user_profile = input['target_user_profile']
                item_attr = input['target_item_attr']
        user_embedding = self.user_embedding(user)
        item_embedding = self.item_embedding(item)
        pred = (user_embedding * item_embedding).sum(dim=-1) + self.bias
        if self.info_size is not None:
            user_profile = self.user_profile(user_profile)
            user_profile = (user_embedding * user_profile).sum(dim=-1)
            item_attr = self.item_attr(item_attr)
            item_attr = (item_embedding * item_attr).sum(dim=-1)
            pred = pred + user_profile + item_attr
        output['target_rating'] = pred
        output['loss'] = loss_fn(output['target_rating'], rating)
        return output


def mf():
    num_users = cfg['num_users']
    num_items = cfg['num_items']
    hidden_size = cfg['mf']['hidden_size']
    info_size = cfg['info_size']
    model = MF(num_users, num_items, hidden_size, info_size)
    return model
