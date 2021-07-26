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
