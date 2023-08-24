import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import loss_fn
from config import cfg


class MDR(nn.Module):
    def __init__(self, model, model_name, match_ratio):
        super().__init__()
        self.model_name = model_name
        self.match_ratio = match_ratio
        model_list = []
        for m in range(len(model)):
            model_list.append(self.make_share(model[0], model[m], model_name))
        self.model_list = nn.ModuleList(model_list)

    def make_share(self, model_0, model_m, model_name):
        if model_name == 'mf':
            self.num_matched = {'user': int(len(model_0.user_weight.weight) * self.match_ratio['user']),
                                'item': int(len(model_0.item_weight.weight) * self.match_ratio['item'])}
            model_m.share_user_weight = model_0.user_weight
            model_m.share_item_weight = model_0.item_weight
        elif model_name == 'nmf':
            self.num_matched = {'user': int(len(model_0.user_weight_mlp.weight) * self.match_ratio['user']),
                                'item': int(len(model_0.item_weight_mlp.weight) * self.match_ratio['item'])}
            model_m.share_user_weight_mlp = model_0.user_weight_mlp
            model_m.share_item_weight_mlp = model_0.item_weight_mlp
            model_m.share_user_weight_mf = model_0.user_weight_mf
            model_m.share_item_weight_mf = model_0.item_weight_mf
        elif model_name == 'ae':
            self.num_matched = {'user': int(len(model_0.user_weight_encoder.weight) * self.match_ratio['user']),
                                'item': int(len(model_0.item_weight_encoder.weight) * self.match_ratio['item'])}
            model_m.share_user_weight_encoder = model_0.user_weight_encoder
            model_m.share_item_weight_encoder = model_0.item_weight_encoder
            model_m.share_user_weight_decoder = model_0.user_weight_decoder
            model_m.share_item_weight_decoder = model_0.item_weight_decoder
        elif model_name == 'simplex':
            self.num_matched = {'user': int(len(model_0.user_weight.weight) * self.match_ratio['user']),
                                'item': int(len(model_0.item_weight.weight) * self.match_ratio['item'])}
            model_m.share_user_weight = model_0.user_weight
            model_m.share_item_weight = model_0.item_weight
        return model_m

    def forward(self, input, m):
        output = self.model_list[m](input, self.num_matched)
        return output


def mdr(model):
    model_name = cfg['model_name']
    match_ratio = cfg['match_ratio']
    model = MDR(model, model_name, match_ratio)
    return model
