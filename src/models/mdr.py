import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import loss_fn
from config import cfg


class MDR(nn.Module):
    def __init__(self, model, data_mode, model_name, match_rate):
        super().__init__()
        self.match_rate = match_rate
        model_list = []
        for m in range(len(model)):
            model_list.append(self.make_share(model[0], model[m], data_mode, model_name))
        self.model_list = nn.ModuleList(model_list)

    def make_share(self, model_0, model_m, data_mode, model_name):
        if model_name in ['mf', 'mlp']:
            if data_mode == 'user':
                num_matched = int(len(model_0.user_weight.weight) * self.match_rate)
                model_m.make_md(num_matched, 'user', model_0.user_weight, model_0.user_bias)
                if model_0.info_size is not None:
                    raise ValueError('Not valid info')
            elif data_mode == 'item':
                num_matched = int(len(model_0.item_weight.weight) * self.match_rate)
                model_m.make_md(num_matched, 'item', model_0.item_weight, model_0.item_bias)
                if model_0.info_size is not None:
                    raise ValueError('Not valid info')
            else:
                raise ValueError('Not valid data mode')
        elif model_name == 'nmf':
            if data_mode == 'user':
                num_matched = int(len(model_0.user_weight_mlp.weight) * self.match_rate)
                model_m.make_md(num_matched, 'user', model_0.user_weight_mlp, model_0.user_bias_mlp,
                                model_0.user_weight_mf, model_0.user_bias_mf)
                if model_0.info_size is not None:
                    raise ValueError('Not valid info')
            elif data_mode == 'item':
                num_matched = int(len(model_0.item_weight_mlp.weight) * self.match_rate)
                model_m.make_md(num_matched, 'item', model_0.item_weight_mlp, model_0.item_bias_mlp,
                                model_0.item_weight_mf, model_0.item_bias_mf)
                if model_0.info_size is not None:
                    raise ValueError('Not valid info')
            else:
                raise ValueError('Not valid data mode')
        return model_m

    def forward(self, input, m):
        output = self.model_list[m](input)
        return output


def mdr(model):
    data_mode = cfg['data_mode']
    model_name = cfg['model_name']
    if 'match_rate' in cfg['assist']:
        match_rate = cfg['assist']['match_rate']
    else:
        match_rate = 1.0
    model = MDR(model, data_mode, model_name, match_rate)
    return model
