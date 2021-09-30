import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import loss_fn, parse_explicit_rating_flat, parse_implicit_rating_flat
from config import cfg


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        blocks = [nn.Linear(input_size, hidden_size[0]),
                  nn.Tanh()]
        for i in range(len(hidden_size) - 1):
            blocks.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            blocks.append(nn.Tanh())
        self.blocks = nn.Sequential(*blocks)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.blocks:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
        return

    def forward(self, x):
        x = self.blocks(x)
        return x


class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        blocks = []
        for i in range(len(hidden_size) - 1):
            blocks.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            blocks.append(nn.Tanh())
        blocks.append(nn.Linear(hidden_size[-1], output_size))
        self.blocks = nn.Sequential(*blocks)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.blocks:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
        return

    def forward(self, x):
        x = self.blocks(x)
        return x


class AE(nn.Module):
    def __init__(self, encoder_num_users, encoder_num_items, decoder_num_users, decoder_num_items, encoder_hidden_size,
                 decoder_hidden_size, info_size):
        super().__init__()
        self.info_size = info_size
        if cfg['data_mode'] == 'user':
            self.encoder = Encoder(encoder_num_items, encoder_hidden_size)
            self.decoder = Decoder(decoder_num_items, decoder_hidden_size)
        elif cfg['data_mode'] == 'item':
            self.encoder = Encoder(encoder_num_users, encoder_hidden_size)
            self.decoder = Decoder(decoder_num_users, decoder_hidden_size)
        else:
            raise ValueError('Not valid data mode')
        self.dropout = nn.Dropout(p=0.5)
        if info_size is not None:
            if 'user_profile' in info_size:
                self.user_profile = Encoder(info_size['user_profile'], encoder_hidden_size)
            if 'item_attr' in info_size:
                self.item_attr = Encoder(info_size['item_attr'], encoder_hidden_size)

    def forward(self, input):
        output = {}
        x = input['rating']
        encoded = self.encoder(x)
        if self.info_size is not None:
            if 'user_profile' in input:
                user_profile = input['user_profile']
                user_profile = self.user_profile(user_profile)
                encoded = encoded + user_profile
            if 'item_attr' in input:
                item_attr = input['item_attr']
                item_attr = self.item_attr(item_attr)
                encoded = encoded + item_attr
        code = self.dropout(encoded)
        decoded = self.decoder(code)
        output['target_rating'] = decoded
        target_mask = ~(input['target_rating'].isnan())
        if 'local' in input and input['local']:
            output['loss'] = F.mse_loss(output['target_rating'][target_mask], input['target_rating'][target_mask])
        else:
            output['loss'] = loss_fn(output['target_rating'][target_mask], input['target_rating'][target_mask])
        if cfg['target_mode'] == 'explicit':
            output['target_rating'], input['target_rating'] = parse_explicit_rating_flat(output['target_rating'],
                                                                                         input['target_rating'])
        elif cfg['target_mode'] == 'implicit':
            output['target_rating'], input['target_rating'] = parse_implicit_rating_flat(output['target_rating'],
                                                                                         input['target_rating'])
        else:
            raise ValueError('Not valid target mode')
        return output


def ae(encoder_num_users=None, encoder_num_items=None, decoder_num_users=None, decoder_num_items=None):
    encoder_num_users = cfg['num_users']['data'] if encoder_num_users is None else encoder_num_users
    encoder_num_items = cfg['num_items']['data'] if encoder_num_items is None else encoder_num_items
    decoder_num_users = cfg['num_users']['target'] if decoder_num_users is None else decoder_num_users
    decoder_num_items = cfg['num_items']['target'] if decoder_num_items is None else decoder_num_items
    encoder_hidden_size = cfg['ae']['encoder_hidden_size']
    decoder_hidden_size = cfg['ae']['decoder_hidden_size']
    info_size = cfg['info_size']
    model = AE(encoder_num_users, encoder_num_items, decoder_num_users, decoder_num_items, encoder_hidden_size,
               decoder_hidden_size, info_size)
    return model
