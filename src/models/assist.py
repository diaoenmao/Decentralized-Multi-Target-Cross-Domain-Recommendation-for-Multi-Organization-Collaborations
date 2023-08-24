import copy
import torch
import torch.nn as nn
from .utils import loss_fn, normalize, denormalize
from config import cfg


class Assist(nn.Module):
    def __init__(self, model, model_name, match_ratio):
        super().__init__()
        self.model_name = model_name
        self.match_ratio = match_ratio
        model_list = []
        parameter_list = []
        for m in range(len(model)):
            model_list.append(model[m])
            parameter_list.append(nn.Parameter(torch.ones(len(model))))
        self.model_list = nn.ModuleList(model_list)
        self.parameter_list = nn.ParameterList(parameter_list)

    def sync(self):
        sync_model_list = []
        for m in range(len(self.model_list)):
            model_list_m = copy.deepcopy(self.model_list)
            for i in range(len(model_list_m)):
                model_list_m[i] = self.make_share(self.model_list[m], model_list_m[i], self.model_name)
            sync_model_list.append(model_list_m)
        self.sync_model_list = nn.ModuleList(sync_model_list)
        return

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
        if self.training:
            user = input['user']
            item = input['item']
            rating = input['rating'].clone().detach()
            if cfg['target_mode'] == 'explicit':
                rating = normalize(rating, cfg['stats']['min'], cfg['stats']['max'])
        else:
            user = input['target_user']
            item = input['target_item']
            rating = input['target_rating'].clone().detach()
            if cfg['target_mode'] == 'explicit':
                rating = normalize(rating, cfg['stats']['min'], cfg['stats']['max'])
        self.make_share(self.model_list[m], self.model_list[m], self.model_name)
        mask = user < self.num_matched['user']
        model_0 = self.model_list[m]
        output_0 = model_0(input, self.num_matched)
        output_target_rating = [None for _ in range(len(self.model_list))]
        output_target_rating[m] = normalize(output_0['target_rating'], cfg['stats']['min'], cfg['stats']['max'])
        for i in range(len(self.sync_model_list[m])):
            if i != m:
                model_m = self.sync_model_list[m][i]
                output_m = model_m(input, self.num_matched)
                output_m_target_rating_ = normalize(output_m['target_rating'], cfg['stats']['min'], cfg['stats']['max'])
                output_m_target_rating_.detach_()
                output_target_rating[i] = output_m_target_rating_
        output_target_rating = torch.stack(output_target_rating, dim=-1) * self.parameter_list[m].softmax(dim=-1)
        output_target_rating = output_target_rating.sum(dim=-1)
        # output_0['target_rating'][mask] = output_target_rating[mask]
        output_0['loss'] = loss_fn(output_0['target_rating'], rating)
        if cfg['target_mode'] == 'explicit':
            output_0['target_rating'] = denormalize(output_0['target_rating'],
                                                    cfg['stats']['min'], cfg['stats']['max'])
        output = output_0
        return output


def assist(model):
    model_name = cfg['model_name']
    match_ratio = cfg['match_ratio']
    model = Assist(model, model_name, match_ratio)
    return model
