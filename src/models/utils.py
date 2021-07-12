import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


def init_param(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        if m.bias is not None:
            m.bias.data.zero_()
    return m


def normalize(input):
    if cfg['data_name'] in cfg['stats']:
        broadcast_size = [1] * input.dim()
        broadcast_size[1] = input.size(1)
        m, s = cfg['stats'][cfg['data_name']]
        m, s = torch.tensor(m, dtype=input.dtype).view(broadcast_size).to(input.device), \
               torch.tensor(s, dtype=input.dtype).view(broadcast_size).to(input.device)
        input = input.sub(m).div(s)
    return input


def denormalize(input):
    if cfg['data_name'] in cfg['stats']:
        broadcast_size = [1] * input.dim()
        broadcast_size[1] = input.size(1)
        m, s = cfg['stats'][cfg['data_name']]
        m, s = torch.tensor(m, dtype=input.dtype).view(broadcast_size).to(input.device), \
               torch.tensor(s, dtype=input.dtype).view(broadcast_size).to(input.device)
        input = input.mul(s).add(m)
    return input


def make_batchnorm(m, momentum, track_running_stats):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.momentum = momentum
        m.track_running_stats = track_running_stats
        if track_running_stats:
            m.register_buffer('running_mean', torch.zeros(m.num_features, device=cfg['device']))
            m.register_buffer('running_var', torch.ones(m.num_features, device=cfg['device']))
            m.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long, device=cfg['device']))
        else:
            m.running_mean = None
            m.running_var = None
            m.num_batches_tracked = None
    return m


def loss_fn(output, target, reduction='mean'):
    if target.dtype == torch.int64:
        loss = F.cross_entropy(output, target, reduction=reduction)
    else:
        loss = F.mse_loss(output, target, reduction=reduction)
    return loss


def mae_loss(output, target, weight=None):
    mae = F.l1_loss(output, target, reduction='none')
    mae = weight * mae if weight is not None else mae
    mae = torch.sum(mae)
    mae /= output.size(0)
    return mae


def mse_loss(output, target, weight=None):
    mse = F.mse_loss(output, target, reduction='none')
    mse = weight * mse if weight is not None else mse
    mse = torch.sum(mse)
    mse /= output.size(0)
    return mse


def cross_entropy_loss(output, target, weight=None):
    if target.dtype != torch.int64:
        target = (target.topk(1, 1, True, True)[1]).view(-1)
    ce = F.cross_entropy(output, target, weight=weight, reduction='mean')
    return ce


def kld_loss(output, target, weight=None, T=1):
    kld = F.kl_div(F.log_softmax(output, dim=-1), F.softmax(target / T, dim=-1), reduction='none')
    kld = weight * kld if weight is not None else kld
    kld = torch.sum(kld)
    kld /= output.size(0)
    return kld


def margin_loss(output, target, weight=None):
    margin = F.multi_margin_loss(output, target, p=1, margin=1.0, weight=weight, reduction='mean')
    return margin


def make_weight(target):
    cls_indx, cls_counts = torch.unique(target, return_counts=True)
    num_samples_per_cls = torch.zeros(cfg['target_size'], dtype=torch.float32, device=target.device)
    num_samples_per_cls[cls_indx] = cls_counts.float()
    beta = torch.tensor(0.999, dtype=torch.float32, device=target.device)
    effective_num = 1.0 - beta.pow(num_samples_per_cls)
    weight = (1.0 - beta) / effective_num
    weight[torch.isinf(weight)] = 0
    weight = weight / torch.sum(weight) * (weight > 0).float().sum()
    return weight
