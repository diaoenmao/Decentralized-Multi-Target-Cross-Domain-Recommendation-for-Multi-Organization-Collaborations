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
        self.ar_state_dict = [None for _ in range(cfg['global']['num_epochs'] + 1)]
        self.reset()

    def reset(self):
        self.organization_output = [{k: None for k in cfg['data_size']} for _ in
                                    range(cfg['global']['num_epochs'] + 1)]
        self.organization_target = [{k: None for k in cfg['data_size']} for _ in
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
                coo = self.organization_target[0][k].tocoo()
                row, col = coo.row, coo.col
                dataset[i][k].target = csr_matrix((residual_k, (row, col)),
                                                  shape=(cfg['num_users']['data'], cfg['num_items']['data']))
                if 'target' in dataset[i][k].user_profile:
                    del dataset[i][k].user_profile['target']
                if 'target' in dataset[i][k].item_attr:
                    del dataset[i][k].item_attr['target']
                dataset[i][k].transform.transforms[0].num_items['target'] = cfg['num_items']['target']
        return dataset

    def update(self, organization_outputs, iter):
        organization_outputs_ = {k: [] for k in organization_outputs[0]}
        for k in organization_outputs[0]:
            for i in range(len(organization_outputs)):
                organization_outputs_[k].append(torch.tensor(organization_outputs[i][k].data))
            organization_outputs_[k] = torch.stack(organization_outputs_[k], dim=-1)
        if 'train' in organization_outputs[0]:
            model = models.ar(iter).to(cfg['device'])
            self.ar_state_dict[iter] = {k: v.cpu() for k, v in model.state_dict().items()}
            print(model.assist_rate)
        with torch.no_grad():
            model = models.ar(iter).to(cfg['device'])
            model.load_state_dict(self.ar_state_dict[iter])
            model.train(False)
            for k in organization_outputs[0]:
                input = {'history': torch.tensor(self.organization_output[iter - 1][k].data),
                         'output': organization_outputs_[k],
                         'target': torch.tensor(self.organization_target[0][k].data)}
                input = to_device(input, cfg['device'])
                output = model(input)
                coo = self.organization_output[iter - 1][k].tocoo()
                row, col = coo.row, coo.col
                self.organization_output[iter][k] = csr_matrix((output['target'].cpu().numpy(), (row, col)),
                                                               shape=(cfg['num_users']['target'],
                                                                      cfg['num_items']['target']))
        return
