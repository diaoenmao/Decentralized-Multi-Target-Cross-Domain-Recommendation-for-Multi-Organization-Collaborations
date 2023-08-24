import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import loss_fn
from config import cfg


class Fed(nn.Module):
    def __init__(self, model, model_name, match_ratio):
        super().__init__()
        self.model_name = model_name
        self.match_ratio = match_ratio
        model_list = []
        for m in range(len(model)):
            model_list.append(model[m])
        self.model_list = nn.ModuleList(model_list)

    def sync(self, model_name):
        with torch.no_grad():
            if model_name == 'mf':
                self.num_matched = {'user': int(len(self.model_list[0].user_weight.weight) * self.match_ratio['user']),
                                    'item': int(len(self.model_list[0].item_weight.weight) * self.match_ratio['item'])}
                user_weight = 0
                item_weight = 0
                for i in range(len(self.model_list)):
                    user_weight += self.model_list[i].user_weight.weight[:self.num_matched['user']]
                    item_weight += self.model_list[i].item_weight.weight[:self.num_matched['item']]
                user_weight = user_weight / (len(self.model_list))
                item_weight = item_weight / (len(self.model_list))
                for i in range(len(self.model_list)):
                    self.model_list[i].user_weight.weight.data[:self.num_matched['user']].copy_(user_weight.data)
                    self.model_list[i].item_weight.weight.data[:self.num_matched['item']].copy_(item_weight.data)
            elif model_name == 'nmf':
                self.num_matched = {
                    'user': int(len(self.model_list[0].user_weight_mlp.weight) * self.match_ratio['user']),
                    'item': int(len(self.model_list[0].item_weight_mlp.weight) * self.match_ratio['item'])}
                user_weight_mlp = 0
                item_weight_mlp = 0
                user_weight_mf = 0
                item_weight_mf = 0
                for i in range(len(self.model_list)):
                    user_weight_mlp += self.model_list[i].user_weight_mlp.weight[:self.num_matched['user']]
                    item_weight_mlp += self.model_list[i].item_weight_mlp.weight[:self.num_matched['item']]
                    user_weight_mf += self.model_list[i].user_weight_mf.weight[:self.num_matched['user']]
                    item_weight_mf += self.model_list[i].item_weight_mf.weight[:self.num_matched['item']]
                user_weight_mlp = user_weight_mlp / (len(self.model_list))
                item_weight_mlp = item_weight_mlp / (len(self.model_list))
                user_weight_mf = user_weight_mf / (len(self.model_list))
                item_weight_mf = item_weight_mf / (len(self.model_list))
                for i in range(len(self.model_list)):
                    self.model_list[i].user_weight_mlp.weight.data[:self.num_matched['user']].copy_(
                        user_weight_mlp.data)
                    self.model_list[i].item_weight_mlp.weight.data[:self.num_matched['item']].copy_(
                        item_weight_mlp.data)
                    self.model_list[i].user_weight_mf.weight.data[:self.num_matched['user']].copy_(user_weight_mf.data)
                    self.model_list[i].item_weight_mf.weight.data[:self.num_matched['item']].copy_(item_weight_mf.data)
            elif model_name == 'ae':
                self.num_matched = {
                    'user': int(len(self.model_list[0].user_weight_encoder.weight) * self.match_ratio['user']),
                    'item': int(len(self.model_list[0].item_weight_encoder.weight) * self.match_ratio['item'])}
                user_weight_encoder = 0
                item_weight_encoder = 0
                user_weight_decoder = 0
                item_weight_decoder = 0
                for i in range(len(self.model_list)):
                    user_weight_encoder += self.model_list[i].user_weight_encoder.weight[:self.num_matched['user']]
                    item_weight_encoder += self.model_list[i].item_weight_encoder.weight[:self.num_matched['item']]
                    user_weight_decoder += self.model_list[i].user_weight_decoder.weight[:self.num_matched['user']]
                    item_weight_decoder += self.model_list[i].item_weight_decoder.weight[:self.num_matched['item']]
                user_weight_encoder = user_weight_encoder / (len(self.model_list))
                item_weight_encoder = item_weight_encoder / (len(self.model_list))
                user_weight_decoder = user_weight_decoder / (len(self.model_list))
                item_weight_decoder = item_weight_decoder / (len(self.model_list))
                for i in range(len(self.model_list)):
                    self.model_list[i].user_weight_encoder.weight.data[:self.num_matched['user']].copy_(
                        user_weight_encoder.data)
                    self.model_list[i].item_weight_encoder.weight.data[:self.num_matched['item']].copy_(
                        item_weight_encoder.data)
                    self.model_list[i].user_weight_decoder.weight.data[:self.num_matched['user']].copy_(
                        user_weight_decoder.data)
                    self.model_list[i].item_weight_decoder.weight.data[:self.num_matched['item']].copy_(
                        item_weight_decoder.data)
            elif model_name == 'simplex':
                self.num_matched = {'user': int(len(self.model_list[0].user_weight.weight) * self.match_ratio['user']),
                                    'item': int(len(self.model_list[0].item_weight.weight) * self.match_ratio['item'])}
                user_weight = 0
                item_weight = 0
                for i in range(len(self.model_list)):
                    user_weight += self.model_list[i].user_weight.weight[:self.num_matched['user']]
                    item_weight += self.model_list[i].item_weight.weight[:self.num_matched['item']]
                user_weight = user_weight / (len(self.model_list))
                item_weight = item_weight / (len(self.model_list))
                for i in range(len(self.model_list)):
                    self.model_list[i].user_weight.weight.data[:self.num_matched['user']].copy_(user_weight.data)
                    self.model_list[i].item_weight.weight.data[:self.num_matched['item']].copy_(item_weight.data)
        return

    def forward(self, input, m):
        output = self.model_list[m](input)
        return output


def fed(model):
    model_name = cfg['model_name']
    match_ratio = cfg['match_ratio']
    model = Fed(model, model_name, match_ratio)
    return model
