import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import loss_fn
from config import cfg


class MF(nn.Module):
    def __init__(self, num_users, num_items, hidden_size):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.user_weight = nn.Embedding(num_users, hidden_size)
        self.item_weight = nn.Embedding(num_items, hidden_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.bias = nn.Parameter(torch.randn(1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.user_weight.weight, 0.0, 0.01)
        nn.init.normal_(self.item_weight.weight, 0.0, 0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.zeros_(self.bias)
        return

    def user_embedding(self, user, aug=None):
        embedding = self.user_weight(user) + self.user_bias(user)
        if aug is not None and cfg['sigma'] > 0:
            embedding = embedding + cfg['sigma'] ** 2 * torch.randn(embedding.size(), device=embedding.device)
        return embedding

    def item_embedding(self, item, aug=None):
        embedding = self.item_weight(item) + self.item_bias(item)
        if aug is not None and cfg['sigma'] > 0:
            embedding = embedding + cfg['sigma'] ** 2 * torch.randn(embedding.size(), device=embedding.device)
        return embedding

    def forward(self, input):
        output = {}
        user, item = input['user'], input['item']
        if 'semi_user' in input:
            semi_user, semi_item = input['semi_user'], input['semi_item']
            user_embedding = self.user_embedding(user, input['aug'])
            item_embedding = self.item_embedding(item, input['aug'])
            semi_user_embedding = self.user_embedding(semi_user, input['aug'])
            semi_item_embedding = self.item_embedding(semi_item, input['aug'])
            user_embedding = torch.cat([user_embedding, semi_user_embedding], dim=0)
            item_embedding = torch.cat([item_embedding, semi_item_embedding], dim=0)
            input['target'] = torch.cat([input['target'], input['semi_target']], dim=0)
        else:
            if 'aug' in input:
                user_embedding = self.user_embedding(user, input['aug'])
                item_embedding = self.item_embedding(item, input['aug'])
            else:
                user_embedding = self.user_embedding(user)
                item_embedding = self.item_embedding(item)
        pred = (user_embedding * item_embedding).sum(dim=-1) + self.bias
        output['target'] = pred
        if 'target' in input:
            output['loss'] = loss_fn(output['target'], input['target'])
        return output


def mf():
    num_users = cfg['num_users']
    num_items = cfg['num_items']
    hidden_size = cfg['mf']['hidden_size']
    model = MF(num_users, num_items, hidden_size)
    return model
