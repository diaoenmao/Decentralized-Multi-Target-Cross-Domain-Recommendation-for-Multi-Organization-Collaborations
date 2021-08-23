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
        if self.info_size is not None:
            if cfg['data_name'] in ['ML100K', 'ML1M']:
                self.user_profile_mf = nn.Linear(info_size['user_profile'], hidden_size)
                self.user_profile_mlp = nn.Linear(info_size['user_profile'], hidden_size)
            self.item_attr_mf = nn.Linear(info_size['item_attr'], hidden_size)
            self.item_attr_mlp = nn.Linear(info_size['item_attr'], hidden_size)
        fc = []
        for i in range(len(hidden_size) - 1):
            if i == 0:
                input_size = 2 * hidden_size[i]
                if self.info_size is not None:
                    if cfg['data_name'] in ['ML100K', 'ML1M']:
                        input_size = input_size + info_size['user_profile']
                    input_size = input_size + info_size['item_attr']
            else:
                input_size = hidden_size[i]
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
        embedding = self.user_weight_mlp(user) + self.user_bias_mlp(user)
        return embedding

    def user_embedding_mf(self, user):
        embedding = self.user_weight_mf(user) + self.user_bias_mf(user)
        return embedding

    def item_embedding_mlp(self, item):
        embedding = self.item_weight_mlp(item) + self.item_bias_mlp(item)
        return embedding

    def item_embedding_mf(self, item):
        embedding = self.item_weight_mf(item) + self.item_bias_mf(item)
        return embedding

    def forward(self, input):
        output = {}
        if self.training:
            user = input['user']
            item = input['item']
            rating = input['rating']
            if self.info_size is not None:
                if cfg['data_name'] in ['ML100K', 'ML1M']:
                    user_profile = input['user_profile']
                item_attr = input['item_attr']
        else:
            user = input['target_user']
            item = input['target_item']
            rating = input['target_rating']
            if self.info_size is not None:
                if cfg['data_name'] in ['ML100K', 'ML1M']:
                    user_profile = input['target_user_profile']
                item_attr = input['target_item_attr']
        user_embedding_mlp = self.user_embedding_mlp(user)
        user_embedding_mf = self.user_embedding_mf(user)
        item_embedding_mlp = self.item_embedding_mlp(item)
        item_embedding_mf = self.item_embedding_mf(item)
        if self.info_size is not None:
            if cfg['data_name'] in ['ML100K', 'ML1M']:
                user_profile_mf = self.user_profile_mf(user_profile)
                user_profile_mf = user_embedding_mf * user_profile_mf
                item_attr_mf = self.item_attr_mf(item_attr)
                item_attr_mf = item_embedding_mf * item_attr_mf
                user_profile_mlp = self.user_profile_mlp(user_profile)
                item_attr_mlp = self.item_attr_mlp(item_attr)
                mf = user_embedding_mf * item_embedding_mf + user_profile_mf + item_attr_mf
                info_mlp = torch.cat([user_profile_mlp, item_attr_mlp], dim=-1)
            else:
                item_attr_mf = self.item_attr_mf(item_attr)
                item_attr_mf = item_embedding_mf * item_attr_mf
                item_attr_mlp = self.item_attr_mlp(item_attr)
                info_mlp = item_attr_mlp
                mf = user_embedding_mf * item_embedding_mf + item_attr_mf
            mlp = torch.cat([user_embedding_mlp, item_embedding_mlp, info_mlp], dim=-1)
        else:
            mf = user_embedding_mf * item_embedding_mf
            mlp = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        mlp = self.fc(mlp)
        mlp_mf = torch.cat([mlp, mf], dim=-1)
        output['target_rating'] = self.affine(mlp_mf).view(-1)
        output['loss'] = loss_fn(output['target_rating'], rating)
        return output


def nmf():
    num_users = cfg['num_users']
    num_items = cfg['num_items']
    hidden_size = cfg['nmf']['hidden_size']
    model = NMF(num_users, num_items, hidden_size)
    return model
