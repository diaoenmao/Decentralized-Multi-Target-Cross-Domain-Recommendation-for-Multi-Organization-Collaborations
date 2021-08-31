import copy
import datetime
import numpy as np
import sys
import time
import torch
import models
from config import cfg
from data import make_data_loader, make_pair_transform
from utils import to_device, make_optimizer, make_scheduler, collate


class Organization:
    def __init__(self, organization_id, data_split, model_name):
        self.organization_id = organization_id
        self.data_split = data_split
        self.num_items = len(data_split)
        self.model_name = model_name
        self.model_state_dict = [None for _ in range(cfg['global']['num_epochs'] + 1)]

    def initialize(self, dataset, metric, logger, iter):
        dataset = copy.deepcopy(dataset)
        dataset = make_pair_transform(dataset, cfg['data_mode'])
        model_name = cfg['model_name']
        cfg['model_name'] = 'base'
        data_loader = make_data_loader(dataset, self.model_name[iter], shuffle={'train': False, 'test': False})
        model = models.base(dataset['train'].num_users, dataset['train'].num_items).to(cfg['device'])
        if 'train' in dataset:
            model.train(True)
            for i, input in enumerate(data_loader['train']):
                input = collate(input)
                input = to_device(input, cfg['device'])
                model(input)
            self.model_state_dict[0] = {k: v.cpu() for k, v in model.state_dict().items()}
            with torch.no_grad():
                model.train(False)
                initialization = {'train': [], 'test': []}
                for i, input in enumerate(data_loader['train']):
                    input = collate(input)
                    input_size = len(input['user'])
                    input = to_device(input, cfg['device'])
                    output = model(input)
                    output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
                    evaluation = metric.evaluate(metric.metric_name['train'], input, output)
                    logger.append(evaluation, 'train', input_size)
                    print(output['target_rating'].size())
                    initialization['train'].append(output['target_rating'].cpu())
                initialization['train'] = torch.cat(initialization['train'], dim=0)
            exit()
        with torch.no_grad():
            model.load_state_dict(self.model_state_dict[0])
            model.train(False)
            for i, input in enumerate(data_loader['test']):
                input = collate(input)
                input_size = len(input['user'])
                input = to_device(input, cfg['device'])
                input['no_parse'] = True
                output = model(input)
                output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
                evaluation = metric.evaluate(metric.metric_name['test'], input, output)
                logger.append(evaluation, 'test', input_size)
                initialization['test'].append(output['raw_target_rating'].cpu())
            initialization['test'] = torch.cat(initialization['test'], dim=0)
        cfg['model_name'] = model_name
        return initialization

    def train(self, dataset, metric, logger, iter):
        dataset = make_split_dataset(dataset, self.data_split)
        data_loader = make_data_loader(dataset, self.model_name[iter])
        model = eval('models.{}().to(cfg["device"])'.format(self.model_name[iter]))
        model.train(True)
        optimizer = make_optimizer(model, self.model_name[iter])
        scheduler = make_scheduler(optimizer, self.model_name[iter])
        for local_epoch in range(1, cfg[self.model_name[iter]]['num_epochs'] + 1):
            start_time = time.time()
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                optimizer.zero_grad()
                output = model(input)
                output['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                evaluation = metric.evaluate(metric.metric_name['train'], input, output)
                logger.append(evaluation, 'train', n=input_size)
            scheduler.step()
            local_time = (time.time() - start_time)
            local_finished_time = datetime.timedelta(
                seconds=round((cfg[self.model_name[iter]]['num_epochs'] - local_epoch) * local_time))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Local Epoch: {}({:.0f}%)'.format(local_epoch, 100. * local_epoch /
                                                                     cfg[self.model_name[iter]]['num_epochs']),
                             'ID: {}'.format(self.organization_id),
                             'Local Finished Time: {}'.format(local_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']), end='\r', flush=True)
        sys.stdout.write('\x1b[2K')
        self.model_state_dict[iter] = copy.deepcopy(model.to('cpu').state_dict())
        return

    def predict(self, iter, dataset):
        with torch.no_grad():
            dataset = make_split_dataset(dataset, self.data_split)
            data_loader = make_data_loader(dataset, self.model_name[iter])
            model = eval('models.{}().to(cfg["device"])'.format(self.model_name[iter]))
            model.load_state_dict(self.model_state_dict[iter])
            model.train(False)
            organization_output = {'target_rating': []}
            for i, input in enumerate(data_loader):
                input = collate(input)
                input = to_device(input, cfg['device'])
                output = model(input)
                organization_output['target_rating'].append(output['target_rating'].cpu())
            organization_output['target_rating'] = torch.cat(organization_output['target_rating'], dim=0)
            organization_output['target_rating'] = organization_output['target_rating'][indices]
        return organization_output
