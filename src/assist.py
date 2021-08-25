import copy
import torch
import models
from config import cfg
from data import make_data_loader
from organization import Organization
from utils import make_optimizer, to_device


class Assist:
    def __init__(self, data_split):
        self.data_split = data_split
        self.num_organizations = len(data_split)
        self.model_name = self.make_model_name()
        self.linesearch_state_dict =[None for _ in range(cfg['global']['num_epochs'] + 1)]
        self.reset()

    def reset(self):
        self.organization_output = [{split: None for split in cfg['data_size']} for _ in
                                    range(cfg['global']['num_epochs'] + 1)]
        self.organization_target = [{split: None for split in cfg['data_size']} for _ in
                                    range(cfg['global']['num_epochs'] + 1)]
        return

    def make_model_name(self):
        model_name_list = cfg['model_name'].split('-')
        if len(model_name_list) == 1:
            model_name = [model_name_list[0] for _ in range(cfg['global']['num_epochs'] + 1)]
            model_name = [model_name for _ in range(self.num_organizations)]
        elif len(model_name_list) == 2:
            model_name = [model_name_list[0]] + [model_name_list[1] for _ in range(cfg['global']['num_epochs'])]
            model_name = [model_name for _ in range(self.num_organizations)]
        else:
            raise ValueError('Not valid model name')
        return model_name

    def make_organization(self):
        model_name = self.model_name
        organization = [None for _ in range(self.num_organizations)]
        for i in range(self.num_organizations):
            model_name_i = model_name[i]
            data_split_i = self.data_split[i]
            organization[i] = Organization(i, data_split_i, model_name_i)
        return organization

    def make_data_loader(self, dataset, iter):
        for split in dataset:
            self.organization_output[iter - 1][split].requires_grad = True
            loss = models.loss_fn(self.organization_output[iter - 1][split],
                                  self.organization_target[0][split], reduction='sum')
            loss.backward()
            self.organization_target[iter][split] = - copy.deepcopy(self.organization_output[iter - 1][split].grad)
            ### Need to fix this
            dataset[split].target = self.organization_target[iter][split].numpy()
            self.organization_output[iter - 1][split].detach_()
        data_loader = [None for _ in range(len(self.feature_split))]
        for i in range(len(self.feature_split)):
            data_loader[i] = make_data_loader(dataset, self.model_name[i][iter])
        return data_loader

    def update(self, organization_outputs, iter):
        _organization_outputs = {split: [] for split in organization_outputs[0]}
        for split in organization_outputs[0]:
            for i in range(len(organization_outputs)):
                _organization_outputs[split].append(organization_outputs[i][split])
            _organization_outputs[split] = torch.stack(_organization_outputs[split], dim=-1)
        if 'train' in organization_outputs[0]:
            input = {'history': self.organization_output[iter - 1]['train'],
                     'output': _organization_outputs['train'],
                     'target': self.organization_target[0]['train']}
            input = to_device(input, cfg['device'])
            model = models.linesearch().to(cfg['device'])
            model.train(True)
            optimizer = make_optimizer(model, 'linesearch')
            for linesearch_epoch in range(1, cfg['linesearch']['num_epochs'] + 1):
                def closure():
                    output = model(input)
                    optimizer.zero_grad()
                    output['loss'].backward()
                    return output['loss']
                optimizer.step(closure)
            self.linesearch_state_dict[iter] = copy.deepcopy(model.to('cpu').state_dict())
        with torch.no_grad():
            model = models.linesearch().to(cfg['device'])
            model.train(False)
            model.load_state_dict(self.linesearch_state_dict[iter])
            for split in organization_outputs[0]:
                input = {'history': self.organization_output[iter - 1][split],
                         'output': _organization_outputs[split]}
                output = model(input)
                self.organization_output[iter][split] = output['target']
        return
