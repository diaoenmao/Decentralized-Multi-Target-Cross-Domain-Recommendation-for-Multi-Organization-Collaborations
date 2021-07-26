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

    def user_embedding(self, user):
        return self.user_weight(user) + self.user_bias(user)

    def item_embedding(self, item):
        return self.item_weight(item) + self.item_bias(item)

    def forward(self, input):
        output = {}
        user, item, target = input['user'], input['item'], input['target']
        pred = []
        for i in range(len(user)):
            user_embedding_i = self.user_embedding(user[i]).view(1, -1)
            item_embedding_i = self.item_embedding(item[i])
            pred_i = user_embedding_i.matmul(item_embedding_i.T).view(-1) + self.bias
            pred.append(pred_i)
        output['target'] = pred
        pred = torch.cat(pred, dim=0)
        target = torch.cat(target, dim=0)
        output['loss'] = loss_fn(pred, target)
        return output


def mf():
    num_users = cfg['num_users']
    num_items = cfg['num_items']
    hidden_size = cfg['mf']['hidden_size']
    model = MF(num_users, num_items, hidden_size)
    return model
