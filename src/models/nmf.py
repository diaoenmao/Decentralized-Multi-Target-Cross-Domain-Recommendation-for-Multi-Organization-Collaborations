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

    def user_embedding_mlp(self, user, tag=None):
        embedding = self.user_weight_mlp(user) + self.user_bias_mlp(user)
        if tag is not None and cfg['sigma'] > 0:
            embedding = embedding + cfg['sigma'] ** 2 * torch.randn(embedding.size(), device=embedding.device)
        return embedding

    def user_embedding_mf(self, user, tag=None):
        embedding = self.user_weight_mf(user) + self.user_bias_mf(user)
        if tag is not None and cfg['sigma'] > 0:
            embedding = embedding + cfg['sigma'] ** 2 * torch.randn(embedding.size(), device=embedding.device)
        return embedding


    def item_embedding_mlp(self, item, tag=None):
        embedding = self.item_weight_mlp(item) + self.item_bias_mlp(item)
        if tag is not None and cfg['sigma'] > 0:
            embedding = embedding + cfg['sigma'] ** 2 * torch.randn(embedding.size(), device=embedding.device)
        return embedding

    def item_embedding_mf(self, item, tag=None):
        embedding = self.item_weight_mf(item) + self.item_bias_mf(item)
        if tag is not None and cfg['sigma'] > 0:
            embedding = embedding + cfg['sigma'] ** 2 * torch.randn(embedding.size(), device=embedding.device)
        return embedding

    def forward(self, input):
        output = {}
        user, item = input['user'], input['item']
        if 'semi_user' in input:
            semi_user, semi_item = input['semi_user'], input['semi_item']
            user_embedding_mlp = self.user_embedding_mlp(user, input['tag'])
            user_embedding_mf = self.user_embedding_mf(user, input['tag'])
            item_embedding_mlp = self.item_embedding_mlp(item, input['tag'])
            item_embedding_mf = self.item_embedding_mf(item, input['tag'])
            semi_user_embedding_mlp = self.user_embedding_mlp(semi_user, input['tag'])
            semi_user_embedding_mf = self.user_embedding_mf(semi_user, input['tag'])
            semi_item_embedding_mlp = self.item_embedding_mlp(semi_item, input['tag'])
            semi_item_embedding_mf = self.item_embedding_mf(semi_item, input['tag'])
            user_embedding_mlp = torch.cat([user_embedding_mlp, semi_user_embedding_mlp], dim=0)
            user_embedding_mf = torch.cat([user_embedding_mf, semi_user_embedding_mf], dim=0)
            item_embedding_mlp = torch.cat([item_embedding_mlp, semi_item_embedding_mlp], dim=0)
            item_embedding_mf = torch.cat([item_embedding_mf, semi_item_embedding_mf], dim=0)
            input['target'] = torch.cat([input['target'], input['semi_target']], dim=0)
        else:
            if 'tag' in input:
                user_embedding_mlp = self.user_embedding_mlp(user, input['tag'])
                user_embedding_mf = self.user_embedding_mf(user, input['tag'])
                item_embedding_mlp = self.item_embedding_mlp(item, input['tag'])
                item_embedding_mf = self.item_embedding_mf(item, input['tag'])
            else:
                user_embedding_mlp = self.user_embedding_mlp(user)
                user_embedding_mf = self.user_embedding_mf(user)
                item_embedding_mlp = self.item_embedding_mlp(item)
                item_embedding_mf = self.item_embedding_mf(item)
        mf = torch.mul(user_embedding_mf, item_embedding_mf)
        mlp = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        mlp = self.fc(mlp)
        mlp_mf = torch.cat([mlp, mf], dim=-1)
        pred = self.affine(mlp_mf).view(-1)
        output['target'] = pred
        if 'target' in input:
            output['loss'] = loss_fn(output['target'], input['target'])
        return output


def nmf():
    num_users = cfg['num_users']
    num_items = cfg['num_items']
    hidden_size = cfg['nmf']['hidden_size']
    model = NMF(num_users, num_items, hidden_size)
    return model
