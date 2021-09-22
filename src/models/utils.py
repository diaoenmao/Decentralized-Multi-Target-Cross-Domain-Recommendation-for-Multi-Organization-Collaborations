import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


def loss_fn(output, target, reduction='mean'):
    if cfg['target_mode'] == 'implicit':
        loss = F.binary_cross_entropy_with_logits(output, target, reduction=reduction)
    elif cfg['target_mode'] == 'explicit':
        loss = F.mse_loss(output, target, reduction=reduction)
    else:
        raise ValueError('Not valid target mode')
    return loss


def parse_implicit_rating_pair(num_users, num_items, user, item, output, target):
    if cfg['data_mode'] == 'user':
        user, user_idx = torch.unique(user, return_inverse=True)
        item_idx = item
        num_users = len(user)
        output_rating = torch.full((num_users, num_items), -float('inf'), device=output.device)
        target_rating = torch.full((num_users, num_items), 0., device=target.device)
        output_rating[user_idx, item_idx] = output
        target_rating[user_idx, item_idx] = target
    elif cfg['data_mode'] == 'item':
        item, item_idx = torch.unique(item, return_inverse=True)
        user_idx = user
        num_items = len(item)
        output_rating = torch.full((num_items, num_users), -float('inf'), device=output.device)
        target_rating = torch.full((num_items, num_users), 0., device=target.device)
        output_rating[item_idx, user_idx] = output
        target_rating[item_idx, user_idx] = target
    else:
        raise ValueError('Not valid data mode')
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
