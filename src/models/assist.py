import torch
import torch.nn as nn
from .utils import loss_fn
from config import cfg


class Assist(nn.Module):
    def __init__(self, ar, ar_mode, num_outputs, num_organizations, aw_mode):
        super().__init__()
        self.ar_mode = ar_mode
        self.aw_mode = aw_mode
        if self.ar_mode == 'optim':
            self.assist_rate = nn.Parameter(torch.full((num_outputs,), ar))
        elif self.ar_mode == 'constant':
            self.register_buffer('assist_rate', torch.full((num_outputs,), ar))
        else:
            raise ValueError('Not valid ar mode')
        if self.aw_mode == 'optim':
            self.assist_weight = nn.Parameter(torch.ones(num_organizations) / num_organizations)
        elif self.aw_mode == 'constant':
            self.register_buffer('assist_weight', torch.ones(num_organizations) / num_organizations)
        else:
            raise ValueError('Not valid aw mode')

    def forward(self, input):
        assist_rate = self.assist_rate[input['output_idx']]
        # assist_weight = self.assist_weight[input['output_idx']]
        output = {}
        if torch.isnan(input['output']).any():
            nan_mask = torch.isnan(input['output'][:, 0])
            output_target_s = input['history'][~nan_mask] + assist_rate[~nan_mask] * (input['output'][~nan_mask] *
                                                                          self.assist_weight.softmax(-1)).sum(-1)
            output_target_c = input['history'][nan_mask] + assist_rate[nan_mask] * (input['output'][nan_mask, 1:] *
                                                                          self.assist_weight[1:].softmax(-1)).sum(-1)
            output['target'] = torch.cat([output_target_s, output_target_c], dim=0)
        else:
            output['target'] = input['history'] + assist_rate * (input['output'] *
                                                                 self.assist_weight.softmax(-1)).sum(-1)
        # output['target'] = input['history'] + assist_rate * (input['output'] *
        #                                                      self.assist_weight.softmax(-1)).sum(-1)
        if 'target' in input:
            output['loss'] = loss_fn(output['target'], input['target'])
        return output


def assist(num_outputs):
    ar = cfg['assist']['ar']
    ar_mode = cfg['assist']['ar_mode']
    num_organizations = cfg['num_organizations']
    aw_mode = cfg['assist']['aw_mode']
    model = Assist(ar, ar_mode, num_outputs, num_organizations, aw_mode)
    return model
