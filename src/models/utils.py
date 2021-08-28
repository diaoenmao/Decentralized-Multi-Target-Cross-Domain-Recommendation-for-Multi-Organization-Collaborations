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


def parse_implicit_rating(user, item, output, target):
    user, user_idx = torch.unique(user, return_inverse=True)
    item, item_idx = torch.unique(item, return_inverse=True)
    num_user, num_item = len(user), len(item)
    output_rating = torch.full((num_user, num_item), -float('inf'), device=output.device)
    target_rating = torch.full((num_user, num_item), -float('inf'), device=target.device)
    output_rating[user_idx, item_idx] = output
    target_rating[user_idx, item_idx] = target
    return output_rating, target_rating
