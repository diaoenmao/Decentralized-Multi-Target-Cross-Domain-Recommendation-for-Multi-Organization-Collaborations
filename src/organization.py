import copy
import datetime
import numpy as np
import sys
import time
import torch
import models
from scipy.sparse import csr_matrix
from config import cfg
from data import make_data_loader, make_pair_transform
from utils import to_device, make_optimizer, make_scheduler, collate


def process_output(target_user, target_item, target_rating):
    target_user_i = target_user.cpu()
    if cfg['data_mode'] == 'explicit':
        target_item_i = target_item.cpu()
        target_rating_i = target_rating.cpu()
    elif cfg['data_mode'] == 'implicit':
        mask = target_rating != -float('inf')
        target_item_i = torch.nonzero(mask)[:, 1].cpu()
        target_rating_i = target_rating[mask].cpu()
    else:
        raise ValueError('Not valid data mode')
    return target_user_i, target_item_i, target_rating_i


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
        data_loader = make_data_loader(dataset, self.model_name[iter])
        model = models.base(dataset['train'].num_users, dataset['train'].num_items).to(cfg['device'])
        output = {}
        target = {}
        if 'train' in dataset:
            model.train(True)
            for i, input in enumerate(data_loader['train']):
                input = collate(input)
                input = to_device(input, cfg['device'])
                model(input)
            self.model_state_dict[0] = {k: v.cpu() for k, v in model.state_dict().items()}
            with torch.no_grad():
                model.train(False)
                target_user = []
                target_item = []
                target_rating = []
                for i, input in enumerate(data_loader['train']):
                    input = collate(input)
                    input_size = len(input['user'])
                    if input_size == 0:
                        continue
                    input = to_device(input, cfg['device'])
                    output_ = model(input)
                    output_['loss'] = output_['loss'].mean() if cfg['world_size'] > 1 else output_['loss']
                    evaluation = metric.evaluate(metric.metric_name['train'], input, output_)
                    logger.append(evaluation, 'train', input_size)
                    target_user_i, target_item_i, target_rating_i = process_output(input['target_user'],
                                                                                   input['target_item'],
                                                                                   output_['target_rating'])
                    target_user.append(target_user_i)
                    target_item.append(target_item_i)
                    target_rating.append(target_rating_i)
                target_user = torch.cat(target_user, dim=0).numpy()
                target_item = torch.cat(target_item, dim=0).numpy()
                target_item = np.array(self.data_split)[target_item]
                target_rating = torch.cat(target_rating, dim=0).numpy()
            output['train'] = csr_matrix((target_rating, (target_user, target_item)),
                                         shape=(cfg['num_users'], cfg['num_items']))
            dataset_coo = dataset['train'].target.tocoo()
            row, col = dataset_coo.row, dataset_coo.col
            target['train'] = csr_matrix((dataset['train'].target.data,
                                          (row, np.array(self.data_split)[col])),
                                         shape=(cfg['num_users'], cfg['num_items']))
        with torch.no_grad():
            model.load_state_dict(self.model_state_dict[0])
            model.train(False)
            target_user = []
            target_item = []
            target_rating = []
            for i, input in enumerate(data_loader['test']):
                input = collate(input)
                input_size = len(input['target_user'])
                if input_size == 0:
                    continue
                input = to_device(input, cfg['device'])
                output_ = model(input)
                output_['loss'] = output_['loss'].mean() if cfg['world_size'] > 1 else output_['loss']
                evaluation = metric.evaluate(metric.metric_name['test'], input, output_)
                logger.append(evaluation, 'test', input_size)
                target_user_i, target_item_i, target_rating_i = process_output(input['target_user'],
                                                                               input['target_item'],
                                                                               output_['target_rating'])
                target_user.append(target_user_i)
                target_item.append(target_item_i)
                target_rating.append(target_rating_i)
            target_user = torch.cat(target_user, dim=0).numpy()
            target_item = torch.cat(target_item, dim=0).numpy()
            target_item = np.array(self.data_split)[target_item]
            target_rating = torch.cat(target_rating, dim=0).numpy()
            output['test'] = csr_matrix((target_rating, (target_user, target_item)),
                                        shape=(cfg['num_users'], cfg['num_items']))
            dataset_coo = dataset['test'].target.tocoo()
            row, col = dataset_coo.row, dataset_coo.col
            target['test'] = csr_matrix((dataset['test'].target.data,
                                         (row, np.array(self.data_split)[col])),
                                        shape=(cfg['num_users'], cfg['num_items']))
        cfg['model_name'] = model_name
        return output, target

    def train(self, dataset, metric, logger, iter):
        data_loader = make_data_loader({'train': dataset}, 'local')['train']
        model = eval('models.{}(dataset.num_users, dataset.num_items, dataset.num_users, '
                     'cfg["num_items"]).to(cfg["device"])'.format(self.model_name[iter]))
        model.train(True)
        optimizer = make_optimizer(model, 'local')
        scheduler = make_scheduler(optimizer, 'local')
        for local_epoch in range(1, cfg['local']['num_epochs'] + 1):
            start_time = time.time()
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = input['user'].size(0)
                if input_size == 0:
                    continue
                input = to_device(input, cfg['device'])
                input['local'] = True
                optimizer.zero_grad()
                output = model(input)
                output['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                evaluation = metric.evaluate([metric.metric_name['train'][0]], input, output)
                logger.append(evaluation, 'train', n=input_size)
            scheduler.step()
            local_time = (time.time() - start_time)
            local_finished_time = datetime.timedelta(
                seconds=round((cfg['local']['num_epochs'] - local_epoch) * local_time))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Local Epoch: {}({:.0f}%)'.format(local_epoch, 100. * local_epoch /
                                                                     cfg['local']['num_epochs']),
                             'ID: {}'.format(self.organization_id),
                             'Local Finished Time: {}'.format(local_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', [metric.metric_name['train'][0]]), end='\r', flush=True)
        sys.stdout.write('\x1b[2K')
        self.model_state_dict[iter] = {k: v.cpu() for k, v in model.state_dict().items()}
        return

    def predict(self, dataset, iter):
        with torch.no_grad():
            data_loader = make_data_loader({'train': dataset}, 'local')['train']
            model = eval('models.{}(dataset.num_users, dataset.num_items, dataset.num_users, '
                         'cfg["num_items"]).to(cfg["device"])'.format(self.model_name[iter]))
            model.load_state_dict(self.model_state_dict[iter])
            model.train(False)
            target_user = []
            target_item = []
            target_rating = []
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = len(input['target_user'])
                if input_size == 0:
                    continue
                input = to_device(input, cfg['device'])
                output_ = model(input)
                target_user_i, target_item_i, target_rating_i = process_output(input['target_user'],
                                                                               input['target_item'],
                                                                               output_['target_rating'])
                target_user.append(target_user_i)
                target_item.append(target_item_i)
                target_rating.append(target_rating_i)
            target_user = torch.cat(target_user, dim=0).numpy()
            target_item = torch.cat(target_item, dim=0).numpy()
            target_rating = torch.cat(target_rating, dim=0).numpy()
            output = csr_matrix((target_rating, (target_user, target_item)),
                                shape=(cfg['num_users'], cfg['num_items']))
        return output
