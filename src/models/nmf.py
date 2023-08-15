import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import loss_fn, normalize, denormalize
from config import cfg


class NMF(nn.Module):
    def __init__(self, num_users, num_items, hidden_size):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.user_weight_mlp = nn.Embedding(num_users, hidden_size[0])
        self.item_weight_mlp = nn.Embedding(num_items, hidden_size[0])
        self.user_weight_mf = nn.Embedding(num_users, hidden_size[0])
        self.item_weight_mf = nn.Embedding(num_items, hidden_size[0])
        fc = []
        for i in range(len(hidden_size) - 1):
            if i == 0:
                input_size = 2 * hidden_size[i]
            else:
                input_size = hidden_size[i]
            fc.append(nn.Linear(input_size, hidden_size[i + 1], bias=False))
            fc.append(nn.Tanh())
            fc.append(nn.Dropout(0.5))
        fc.append(nn.Linear(hidden_size[-1], 1, bias=False))
        fc.append(nn.Tanh())
        self.fc = nn.Sequential(*fc)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.user_weight_mlp.weight, 0.0, 1e-4)
        nn.init.normal_(self.item_weight_mlp.weight, 0.0, 1e-4)
        nn.init.normal_(self.user_weight_mf.weight, 0.0, 1e-4)
        nn.init.normal_(self.item_weight_mf.weight, 0.0, 1e-4)
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='tanh')
                if m.bias is not None:
                    m.bias.data.zero_()
        return

    def user_embedding_mlp(self, user, num_matched):
        embedding = self.user_weight_mlp(user)
        if num_matched is not None:
            embedding[user < num_matched['user']] = self.share_user_weight_mlp(user[user < num_matched['user']])
        return embedding

    def user_embedding_mf(self, user, num_matched):
        embedding = self.user_weight_mf(user)
        if num_matched is not None:
            embedding[user < num_matched['user']] = self.share_user_weight_mf(user[user < num_matched['user']])
        return embedding

    def item_embedding_mlp(self, item, num_matched):
        embedding = self.item_weight_mlp(item)
        if num_matched is not None:
            embedding[item < num_matched['item']] = self.share_item_weight_mlp(item[item < num_matched['item']])
        return embedding

    def item_embedding_mf(self, item, num_matched):
        embedding = self.item_weight_mf(item)
        if num_matched is not None:
            embedding[item < num_matched['item']] = self.share_item_weight_mf(item[item < num_matched['item']])
        return embedding

    def make_md(self, num_matched, md_mode, weight_mlp, weight_mf):
        self.num_matched = num_matched
        self.md_mode = md_mode
        self.md_weight_mlp = weight_mlp
        self.md_weight_mf = weight_mf
        return

    def forward(self, input, num_matched=None):
        output = {}
        if self.training:
            user = input['user']
            item = input['item']
            rating = input['rating'].clone().detach()
            rating = normalize(rating, cfg['stats']['min'], cfg['stats']['max'])
        else:
            user = input['target_user']
            item = input['target_item']
            rating = input['target_rating'].clone().detach()
            rating = normalize(rating, cfg['stats']['min'], cfg['stats']['max'])

        user_embedding_mlp = self.user_embedding_mlp(user, num_matched)
        item_embedding_mlp = self.item_embedding_mlp(item, num_matched)
        mlp = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        mlp = self.fc(mlp).squeeze()

        user_embedding_mf = self.user_embedding_mf(user, num_matched)
        item_embedding_mf = self.item_embedding_mf(item, num_matched)
        user_embedding_mf = F.normalize(user_embedding_mf - user_embedding_mf.mean(dim=-1, keepdims=True), dim=-1)
        item_embedding_mf = F.normalize(item_embedding_mf - item_embedding_mf.mean(dim=-1, keepdims=True), dim=-1)
        mf = torch.bmm(user_embedding_mf.unsqueeze(1), item_embedding_mf.unsqueeze(-1)).squeeze()

        nmf = 0.5 * mlp + 0.5 * mf
        output['loss'] = loss_fn(nmf, rating)
        output['target_rating'] = denormalize(nmf, cfg['stats']['min'], cfg['stats']['max'])
        return output


def nmf(num_users=None, num_items=None):
    num_users = cfg['num_users']['data'] if num_users is None else num_users
    num_items = cfg['num_items']['data'] if num_items is None else num_items
    hidden_size = cfg['nmf']['hidden_size']
    model = NMF(num_users, num_items, hidden_size)
    return model
