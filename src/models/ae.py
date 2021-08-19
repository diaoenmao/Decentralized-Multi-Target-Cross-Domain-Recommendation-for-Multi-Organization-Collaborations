import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import loss_fn
from config import cfg


class Encoder(nn.Module):
    def __init__(self, num_users, num_items, hidden_size, info_size):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.info_size = info_size
        if info_size is None:
            blocks = [nn.Linear(num_items, hidden_size[0]),
                      nn.SELU()]
            for i in range(len(hidden_size) - 1):
                blocks.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
                blocks.append(nn.SELU())
            self.blocks = nn.Sequential(*blocks)
        else:
            blocks = [nn.Linear(num_items + info_size['user_profile'] + info_size['item_attr'], hidden_size[0]),
                      nn.SELU()]
            for i in range(len(hidden_size) - 1):
                blocks.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
                blocks.append(nn.SELU())
            self.blocks = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(p=0.8)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.blocks:
            if isinstance(m, nn.Linear):
                if cfg['data_mode'] == 'explicit':
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif cfg['data_mode'] == 'implicit':
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()
        return

    def forward(self, x, user_profile=None, item_attr=None):
        if self.info_size is None:
            x = self.blocks(x)
        else:
            x = torch.cat([x, user_profile, item_attr], dim=-1)
            x = self.blocks(x)
        x = self.dropout(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_users, num_items, hidden_size):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_size = hidden_size
        blocks = []
        for i in range(len(hidden_size) - 1):
            blocks.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            blocks.append(nn.SELU())
        blocks.append(nn.Linear(hidden_size[-1], num_items))
        self.blocks = nn.Sequential(*blocks)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.blocks:
            if isinstance(m, nn.Linear):
                if cfg['data_mode'] == 'explicit':
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif cfg['data_mode'] == 'implicit':
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()
        return

    def forward(self, x):
        x = self.blocks(x)
        return x


class AE(nn.Module):
    def __init__(self, num_users, num_items, encoder_hidden_size, decoder_hidden_size, info_size):
        super().__init__()
        self.info_size = info_size
        self.encoder = Encoder(num_users, num_items, encoder_hidden_size, info_size)
        self.decoder = Decoder(num_users, num_items, decoder_hidden_size)

    def forward(self, input):
        output = {}
        rating = input['rating']
        target_rating = input['target_rating']
        target_mask = ~target_rating.isnan()
        if self.info_size is not None:
            user_profile = input['user_profile']
            item_attr = input['item_attr']
            encoded = self.encoder(rating, user_profile, item_attr)
        else:
            x = rating
            encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        output['target_rating'] = decoded[target_mask]
        input['target_rating'] = input['target_rating'][target_mask]
        output['loss'] = loss_fn(output['target_rating'], input['target_rating'])
        return output


def ae():
    num_users = cfg['num_users']
    num_items = cfg['num_items']
    encoder_hidden_size = cfg['ae']['encoder_hidden_size']
    decoder_hidden_size = cfg['ae']['decoder_hidden_size']
    info_size = cfg['info_size']
    model = AE(num_users, num_items, encoder_hidden_size, decoder_hidden_size, info_size)
    return model