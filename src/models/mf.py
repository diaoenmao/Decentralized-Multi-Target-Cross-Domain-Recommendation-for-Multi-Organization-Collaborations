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

    def user_embedding(self, user, tag=None):
        embedding = self.user_weight(user) + self.user_bias(user)
        if tag == 'weak':
            embedding = embedding + torch.normal(0, 0.01, embedding.size()).to(embedding.device)
        elif tag == 'strong':
            embedding = embedding + torch.normal(0, 0.1, embedding.size()).to(embedding.device)
        return embedding

    def item_embedding(self, item, tag=None):
        embedding = self.item_weight(item) + self.item_bias(item)
        if tag == 'weak':
            embedding = embedding + torch.normal(0, 0.01, embedding.size()).to(embedding.device)
        elif tag == 'strong':
            embedding = embedding + torch.normal(0, 0.1, embedding.size()).to(embedding.device)
        return embedding

    def forward(self, input):
        output = {}
        user, item = input['user'], input['item']
        if 'tag' in input:
            user_embedding = self.user_embedding(user, input['tag'])
            item_embedding = self.item_embedding(item, input['tag'])
        else:
            user_embedding = self.user_embedding(user)
            item_embedding = self.item_embedding(item)
        pred = (user_embedding * item_embedding).sum(dim=-1) + self.bias
        output['target'] = pred
        if 'target' in input:
            target = input['target']
            output['loss'] = loss_fn(pred, target)
        return output


def mf():
    num_users = cfg['num_users']
    num_items = cfg['num_items']
    hidden_size = cfg['mf']['hidden_size']
    model = MF(num_users, num_items, hidden_size)
    return model
