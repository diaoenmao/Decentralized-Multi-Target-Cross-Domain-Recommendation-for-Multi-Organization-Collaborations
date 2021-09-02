import copy
import torch
import models
from scipy.sparse import csr_matrix
from config import cfg
from data import make_data_loader
from organization import Organization
from utils import make_optimizer, to_device


class Assist:
    def __init__(self, data_split):
        self.data_split = data_split
        self.num_organizations = len(data_split)
        self.model_name = self.make_model_name()
        self.linesearch_state_dict = [None for _ in range(cfg['global']['num_epochs'] + 1)]
        self.reset()

    def reset(self):
        self.organization_output = [{split: None for split in cfg['data_size']} for _ in
                                    range(cfg['global']['num_epochs'] + 1)]
        self.organization_target = [{split: None for split in cfg['data_size']} for _ in
                                    range(cfg['global']['num_epochs'] + 1)]
        return

    def make_model_name(self):
        model_name = [cfg['model_name'] for _ in range(cfg['global']['num_epochs'] + 1)]
        model_name = [model_name for _ in range(self.num_organizations)]
        return model_name

    def make_organization(self):
        model_name = self.model_name
        organization = [None for _ in range(self.num_organizations)]
        for i in range(self.num_organizations):
            model_name_i = model_name[i]
            data_split_i = self.data_split[i]
            organization[i] = Organization(i, data_split_i, model_name_i)
        return organization

    def make_dataset(self, dataset, iter):
        for k in dataset[0]:
            output_k = torch.tensor(self.organization_output[iter - 1][k].data, dtype=torch.float32)
            target_k = torch.tensor(self.organization_target[0][k].data, dtype=torch.float32)
            output_k.requires_grad = True
            loss = models.loss_fn(output_k, target_k, reduction='sum')
            loss.backward()
            residual_k = - copy.deepcopy(output_k.grad)
            output_k.detach_()
            for i in range(len(dataset)):
                dataset[i][k].target = csr_matrix((residual_k, self.organization_target[0][k].nonzero()),
                                                  shape=(cfg['num_users'], cfg['num_items']))
                dataset[i][k].target_item_attr_flag = False
                dataset[i][k].transform.transforms[0].target_num_items = cfg['num_items']
        return dataset

    def update(self, organization_outputs, iter):
        organization_outputs_ = {k: [] for k in organization_outputs[0]}
        for split in organization_outputs[0]:
            for i in range(len(organization_outputs)):
                print(organization_outputs[i][split])
                exit()
                organization_outputs_[split].append(organization_outputs[i][split])
            organization_outputs_[split] = torch.stack(organization_outputs_[split], dim=-1)
        if 'train' in organization_outputs[0]:
            input = {'history': self.organization_output[iter - 1]['train'],
                     'output': organization_outputs_['train'],
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
            model.load_state_dict(self.linesearch_state_dict[iter])
            model.train(False)
            for split in organization_outputs[0]:
                input = {'history': self.organization_output[iter - 1][split],
                         'output': organization_outputs_[split]}
                output = model(input)
                self.organization_output[iter][split] = output['target']
        return
