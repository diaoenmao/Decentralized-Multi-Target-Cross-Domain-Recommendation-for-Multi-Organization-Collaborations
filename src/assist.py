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
        self.linesearch_state_dict = [[None for _ in range(len(self.data_split))] for _ in
                                      range(cfg['global']['num_epochs'] + 1)]
        self.reset()

    def reset(self):
        self.organization_output = [{k: [None for _ in range(len(self.data_split))] for k in cfg['data_size']}
                                    for _ in range(cfg['global']['num_epochs'] + 1)]
        self.organization_target = [{k: [None for _ in range(len(self.data_split))] for k in cfg['data_size']}
                                    for _ in range(cfg['global']['num_epochs'] + 1)]
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
            for i in range(len(dataset)):
                output_k_i = torch.tensor(self.organization_output[iter - 1][k][i].data, dtype=torch.float32)
                target_k_i = torch.tensor(self.organization_target[0][k][i].data, dtype=torch.float32)
                output_k_i.requires_grad = True
                loss = models.loss_fn(output_k_i, target_k_i, reduction='sum')
                print(loss)
                loss.backward()
                residual_k_i = - copy.deepcopy(output_k_i.grad)
                output_k_i.detach_()
                coo = self.organization_target[0][k][i].tocoo()
                row, col = coo.row, coo.col
                dataset[i][k].target = csr_matrix((residual_k_i, (row, col)),
                                                  shape=(dataset[i][k].num_users, dataset[i][k].num_items))
                dataset[i][k].target_item_attr_flag = False
                dataset[i][k].transform.transforms[0].target_num_items = dataset[i][k].num_items
        return dataset

    def update(self, organization_outputs, iter):
        for i in range(len(organization_outputs)):
            for k in organization_outputs[i]:
                organization_outputs_i_k = organization_outputs[i][k].data
                if k == 'train':
                    input = {'history': torch.tensor(self.organization_output[iter - 1]['train'][i].data),
                             'output': torch.tensor(organization_outputs_i_k),
                             'target': torch.tensor(self.organization_target[0]['train'][i].data)}
                    input = to_device(input, cfg['device'])
                    model = models.ar(iter).to(cfg['device'])
                    # model.train(True)
                    # optimizer = make_optimizer(model, 'linesearch')
                    # for linesearch_epoch in range(1, cfg['linesearch']['num_epochs'] + 1):
                    #     global loss
                    #     def closure():
                    #         global loss
                    #         output = model(input)
                    #         optimizer.zero_grad()
                    #         output['loss'].backward()
                    #         loss = output['loss']
                    #         return output['loss']
                    #
                    #     optimizer.step(closure)
                    print(model.assist_rate)
                    self.linesearch_state_dict[iter][i] = {k: v.cpu() for k, v in model.state_dict().items()}
                with torch.no_grad():
                    model = models.ar(iter).to(cfg['device'])
                    model.load_state_dict(self.linesearch_state_dict[iter][i])
                    model.train(False)
                    input = {'history': torch.tensor(self.organization_output[iter - 1][k][i].data),
                             'output': torch.tensor(organization_outputs_i_k)}
                    input = to_device(input, cfg['device'])
                    output = model(input)
                    num_users, num_items = self.organization_output[iter - 1][k][i].shape
                    coo = self.organization_output[iter - 1][k][i].tocoo()
                    row, col = coo.row, coo.col
                    self.organization_output[iter][k][i] = csr_matrix((output['target'].cpu().numpy(), (row, col)),
                                                                   shape=(num_users, num_items))
        return
