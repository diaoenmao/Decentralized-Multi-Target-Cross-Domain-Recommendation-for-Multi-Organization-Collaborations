import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


def loss_fn(output, target, reduction='mean'):
    if cfg['data_mode'] == 'implicit':
        loss = F.binary_cross_entropy_with_logits(output, target, reduction=reduction)
    elif cfg['data_mode'] == 'explicit':
        loss = F.mse_loss(output, target, reduction=reduction)
    else:
        raise ValueError('Not valid data type')
    return loss


def parse_implicit_rating_pair(user, item, output, target):
    user, user_idx = torch.unique(user, return_inverse=True)
    item, item_idx = torch.unique(item, return_inverse=True)
    num_user, num_item = len(user), len(item)
    output_rating = torch.full((num_user, num_item), -float('inf'), device=output.device)
    target_rating = torch.full((num_user, num_item), 0., device=target.device)
    output_rating[user_idx, item_idx] = output
    target_rating[user_idx, item_idx] = target
    return output_rating, target_rating


def parse_explicit_rating_flat(output, target):
    target_mask = ~target.isnan()
    output_rating = output[target_mask]
    target_rating = target[target_mask]
    return output_rating, target_rating


def parse_implicit_rating_flat(output, target):
    target_mask = ~target.isnan()
    output_rating = torch.full(output.size(), -float('inf'), device=output.device)
    target_rating = torch.full(target.size(), 0., device=target.device)
    output_rating[target_mask] = output[target_mask]
    target_rating[target_mask] = target[target_mask]
    return output_rating, target_rating
