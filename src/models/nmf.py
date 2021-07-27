import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import loss_fn
from config import cfg


class NMF(nn.Module):
    def __init__(self, num_users, num_items, hidden_size):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.user_weight_mlp = nn.Embedding(num_users, hidden_size[0])
        self.item_weight_mlp = nn.Embedding(num_items, hidden_size[0])
        self.user_bias_mlp = nn.Embedding(num_users, 1)
        self.item_bias_mlp = nn.Embedding(num_items, 1)
        self.user_weight_mf = nn.Embedding(num_users, hidden_size[0])
        self.item_weight_mf = nn.Embedding(num_items, hidden_size[0])
        self.user_bias_mf = nn.Embedding(num_users, 1)
        self.item_bias_mf = nn.Embedding(num_items, 1)
        fc = []
        for i in range(len(hidden_size) - 1):
            input_size = 2 * hidden_size[i] if i == 0 else hidden_size[i]
            fc.append(torch.nn.Linear(input_size, hidden_size[i + 1]))
            fc.append(nn.ReLU())
        self.fc = nn.Sequential(*fc)
        self.affine = nn.Linear(hidden_size[-1] + hidden_size[0], 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.user_weight_mlp.weight, 0.0, 0.01)
        nn.init.normal_(self.item_weight_mlp.weight, 0.0, 0.01)
        nn.init.zeros_(self.user_bias_mlp.weight)
        nn.init.zeros_(self.item_bias_mlp.weight)
        nn.init.normal_(self.user_weight_mf.weight, 0.0, 0.01)
        nn.init.normal_(self.item_weight_mf.weight, 0.0, 0.01)
        nn.init.zeros_(self.user_bias_mf.weight)
        nn.init.zeros_(self.item_bias_mf.weight)
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)
        return

    def user_embedding_mlp(self, user):
        return self.user_weight_mlp(user) + self.user_bias_mlp(user)

    def user_embedding_mf(self, user):
        return self.user_weight_mf(user) + self.user_bias_mf(user)

    def item_embedding_mlp(self, item):
        return self.item_weight_mlp(item) + self.item_bias_mlp(item)

    def item_embedding_mf(self, item):
        return self.item_weight_mf(item) + self.item_bias_mf(item)

    def forward(self, input):
        output = {}
        user, item = input['user'], input['item']
        pred = []
        for i in range(len(user)):
            user_embedding_mlp_i = self.user_embedding_mlp(user[i]).view(1, -1)
            user_embedding_mf_i = self.user_embedding_mf(user[i]).view(1, -1)
            item_embedding_mlp_i = self.item_embedding_mlp(item[i])
            item_embedding_mf_i = self.item_embedding_mf(item[i])
            mf = torch.mul(user_embedding_mf_i, item_embedding_mf_i)
            mlp = torch.cat([user_embedding_mlp_i.expand(item_embedding_mlp_i.size(0), -1), item_embedding_mlp_i],
                            dim=-1)
            mlp = self.fc(mlp)
            mlp_mf = torch.cat([mlp, mf], dim=-1)
            pred_i = self.affine(mlp_mf).view(-1)
            pred.append(pred_i)
        output['target'] = pred
        pred = torch.cat(pred, dim=0)
        if 'target' in input:
            target = input['target']
            target = torch.cat(target, dim=0)
            output['loss'] = loss_fn(pred, target)
        return output


def nmf():
    num_users = cfg['num_users']
    num_items = cfg['num_items']
    hidden_size = cfg['nmf']['hidden_size']
    model = NMF(num_users, num_items, hidden_size)
    return model
