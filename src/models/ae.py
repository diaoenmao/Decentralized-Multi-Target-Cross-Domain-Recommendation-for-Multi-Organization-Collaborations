import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import loss_fn, normalize, denormalize, cusum_size
from config import cfg


class Linear(nn.Linear):
    def __init__(self, input_size, output_size, bias):
        super().__init__(input_size, output_size, bias)
        self.res_size = min(input_size, output_size)

    def forward(self, x):
        output = super().forward(x)
        output[..., :self.res_size] = 0.5 * x[..., :self.res_size] + 0.5 * output[..., :self.res_size]
        return output


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        blocks = [Linear(input_size, hidden_size[0], bias=False),
                  nn.Tanh()]
        for i in range(len(hidden_size) - 1):
            blocks.append(Linear(hidden_size[i], hidden_size[i + 1], bias=False))
            blocks.append(nn.Tanh())
        self.blocks = nn.Sequential(*blocks)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.blocks:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='tanh')
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
            blocks.append(Linear(hidden_size[i], hidden_size[i + 1], bias=False))
            blocks.append(nn.Tanh())
        blocks.append(Linear(hidden_size[-1], output_size, bias=False))
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


class AE(nn.Module):
    def __init__(self, num_users, num_items, encoder_hidden_size, decoder_hidden_size):
        super().__init__()
        self.encoder = Encoder(encoder_hidden_size[0], encoder_hidden_size[1:])
        self.decoder = Decoder(decoder_hidden_size[-1], decoder_hidden_size[:-1])
        self.user_weight_encoder = nn.Embedding(num_users, encoder_hidden_size[0])
        self.item_weight_encoder = nn.Embedding(num_items, encoder_hidden_size[0])
        self.user_weight_decoder = nn.Embedding(num_users, decoder_hidden_size[-1])
        self.item_weight_decoder = nn.Embedding(num_items, decoder_hidden_size[-1])
        self.dropout = nn.Dropout(0.5)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.user_weight_encoder.weight, 0.0, 1e-4)
        nn.init.normal_(self.item_weight_encoder.weight, 0.0, 1e-4)
        nn.init.normal_(self.user_weight_decoder.weight, 0.0, 1e-4)
        nn.init.normal_(self.item_weight_decoder.weight, 0.0, 1e-4)
        return

    def user_embedding_encoder(self, user):
        embedding = self.user_weight_encoder(user)
        if hasattr(self, 'num_matched') and self.md_mode == 'user':
            embedding[user < self.num_matched] = self.md_weight_encoder(user[user < self.num_matched])
        return embedding

    def user_embedding_decoder(self, user):
        embedding = self.user_weight_decoder(user)
        if hasattr(self, 'num_matched') and self.md_mode == 'user':
            embedding[user < self.num_matched] = self.md_weight_decoder(user[user < self.num_matched])
        return embedding

    def item_embedding_encoder(self, item):
        embedding = self.item_weight_encoder(item)
        if hasattr(self, 'num_matched') and self.md_mode == 'item':
            embedding[item < self.num_matched] = self.md_weight_encoder(item[item < self.num_matched])
        return embedding

    def item_embedding_decoder(self, item):
        embedding = self.item_weight_decoder(item)
        if hasattr(self, 'num_matched') and self.md_mode == 'item':
            embedding[item < self.num_matched] = self.md_weight_decoder(item[item < self.num_matched])
        return embedding

    def make_md(self, num_matched, md_mode, weight_encoder, weight_decoder):
        self.num_matched = num_matched
        self.md_mode = md_mode
        self.md_weight_encoder = weight_encoder
        self.md_weight_decoder = weight_decoder
        return

    def forward(self, input):
        output = {}
        if self.training:
            user = input['user']
            target_user = input['target_user']
            item = input['item']
            target_item = input['target_item']
            rating = input['rating'].clone().detach()
            rating = normalize(rating, cfg['stats']['min'], cfg['stats']['max'])
            size = input['size']
            target_size = input['target_size']
        else:
            user = input['user']
            target_user = input['target_user']
            item = input['item']
            target_item = input['target_item']
            rating = input['target_rating'].clone().detach()
            rating = normalize(rating, cfg['stats']['min'], cfg['stats']['max'])
            size = input['size']
            target_size = input['target_size']

        if cfg['data_mode'] == 'user':
            user_embedding_encoder = self.user_embedding_encoder(user)[torch.cumsum(size, dim=0) - 1]
            item_embedding_encoder = self.item_embedding_encoder(item)
            item_embedding_encoder_cusum_size = cusum_size(item_embedding_encoder, size)
            item_embedding_encoder_cusum_size = item_embedding_encoder_cusum_size / size.unsqueeze(-1)
            embedding = 0.5 * user_embedding_encoder + 0.5 * item_embedding_encoder_cusum_size
            encoded = self.encoder(embedding)
            code = self.dropout(encoded)
            decoded = self.decoder(code)
            decoded_embedding = torch.repeat_interleave(decoded, target_size, dim=0)
            target_item_embedding_decoder = self.item_embedding_decoder(target_item)
            decoded_embedding = F.normalize(decoded_embedding - decoded_embedding.mean(dim=-1, keepdims=True), dim=-1)
            target_item_embedding_decoder = F.normalize(target_item_embedding_decoder -
                                                        target_item_embedding_decoder.mean(dim=-1,
                                                                                           keepdims=True), dim=-1)
            ae = torch.bmm(decoded_embedding.unsqueeze(1), target_item_embedding_decoder.unsqueeze(-1)).squeeze()
        elif cfg['data_mode'] == 'item':
            item_embedding_encoder = self.item_embedding_encoder(item)[torch.cumsum(size, dim=0) - 1]
            user_embedding_encoder = self.user_embedding_encoder(user)
            user_embedding_encoder_cusum_size = cusum_size(user_embedding_encoder, size) / size.unsqueeze(-1)
            embedding = 0.5 * item_embedding_encoder + 0.5 * user_embedding_encoder_cusum_size
            encoded = self.encoder(embedding)
            code = self.dropout(encoded)
            decoded = self.decoder(code)
            decoded_embedding = torch.repeat_interleave(decoded, target_size, dim=0)
            target_user_embedding_decoder = self.user_embedding_decoder(target_user)
            decoded_embedding = F.normalize(decoded_embedding - decoded_embedding.mean(dim=-1, keepdims=True), dim=-1)
            target_user_embedding_decoder = F.normalize(target_user_embedding_decoder -
                                                        target_user_embedding_decoder.mean(dim=-1,
                                                                                           keepdims=True), dim=-1)
            ae = torch.bmm(decoded_embedding.unsqueeze(1), target_user_embedding_decoder.unsqueeze(-1)).squeeze()
        else:
            raise ValueError('Not valid data mode')
        output['loss'] = loss_fn(ae, rating)
        output['target_rating'] = denormalize(ae, cfg['stats']['min'], cfg['stats']['max'])
        return output


def ae():
    num_users = cfg['num_users']['data']
    num_items = cfg['num_items']['data']
    encoder_hidden_size = cfg['ae']['encoder_hidden_size']
    decoder_hidden_size = cfg['ae']['decoder_hidden_size']
    model = AE(num_users, num_items, encoder_hidden_size, decoder_hidden_size)
    return model
