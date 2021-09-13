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


def parse_implicit_rating_pair(num_items, user, item, output, target):
    user, user_idx = torch.unique(user, return_inverse=True)
    item_idx = item
    num_users = len(user)
    output_rating = torch.full((num_users, num_items), -float('inf'), device=output.device)
    target_rating = torch.full((num_users, num_items), 0., device=target.device)
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
    indices = target_mask.nonzero()
    user, user_idx = torch.unique(indices[:, 0], return_inverse=True)
    item_idx = indices[:, 1]
    num_users = len(user)
    num_items = output.size(1)
    output_rating = torch.full((num_users, num_items), -float('inf'), device=output.device)
    target_rating = torch.full((num_users, num_items), 0., device=target.device)
    output_rating[user_idx, item_idx] = output[target_mask]
    target_rating[user_idx, item_idx] = target[target_mask]
    return output_rating, target_rating
