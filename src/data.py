import copy
import torch
import numpy as np
import models
from config import cfg
from scipy.sparse import csr_matrix
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from utils import collate, to_device


def fetch_dataset(data_name, model_name=None, data_split=None, verbose=True):
    import datasets

    model_name = cfg['model_name'] if model_name is None else model_name
    dataset = {}
    if verbose:
        print('fetching data {}...'.format(data_name))
    root = './data/{}'.format(data_name)
    if data_name in ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'NFP']:
        dataset['train'] = eval(
            'datasets.{}(root=root, split=\'train\', data_mode=cfg["data_mode"], data_split=data_split)'.format(
                data_name))
        dataset['test'] = eval(
            'datasets.{}(root=root, split=\'test\', data_mode=cfg["data_mode"], data_split=data_split)'.format(
                data_name))
        if model_name in ['base', 'mf', 'gmf', 'mlp', 'nmf']:
            dataset = make_pair_transform(dataset, cfg['data_mode'])
        elif model_name in ['ae']:
            dataset = make_flat_transform(dataset, cfg['data_mode'])
        else:
            raise ValueError('Not valid model name')
    else:
        raise ValueError('Not valid dataset name')
    if verbose:
        print('data ready')
    return dataset


def make_pair_transform(dataset, data_mode):
    import datasets
    if data_mode == 'explicit':
        dataset['train'].transform = datasets.Compose([PairInput()])
        dataset['test'].transform = datasets.Compose([PairInput()])
    elif data_mode == 'implicit':
        dataset['train'].transform = datasets.Compose(
            [NegativeSample(dataset['train'].item_attr, dataset['train'].num_items, cfg['num_negatives']),
             PairInput()])
        dataset['test'].transform = datasets.Compose(
            [PairInput()])
    else:
        raise ValueError('Not valid data mode')
    return dataset


def make_flat_transform(dataset, data_mode):
    import datasets
    if data_mode == 'explicit':
        dataset['train'].transform = datasets.Compose(
            [FlatInput(dataset['train'].num_items, dataset['train'].num_items)])
        dataset['test'].transform = datasets.Compose(
            [FlatInput(dataset['train'].num_items, dataset['train'].num_items)])
    elif data_mode == 'implicit':
        dataset['train'].transform = datasets.Compose(
            [FlatInput(dataset['train'].num_items, dataset['train'].num_items)])
        dataset['test'].transform = datasets.Compose(
            [FlatInput(dataset['train'].num_items, dataset['train'].num_items)])
    else:
        raise ValueError('Not valid data mode')
    return dataset


def input_collate(batch):
    if isinstance(batch[0], dict):
        output = {key: [] for key in batch[0].keys()}
        for b in batch:
            for key in b:
                output[key].append(b[key])
        return output
    else:
        return default_collate(batch)


def make_data_loader(dataset, tag, batch_size=None, shuffle=None, sampler=None):
    data_loader = {}
    for k in dataset:
        _batch_size = cfg[tag]['batch_size'][k] if batch_size is None else batch_size[k]
        _shuffle = cfg[tag]['shuffle'][k] if shuffle is None else shuffle[k]
        if sampler is None:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, shuffle=_shuffle,
                                        pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))
        else:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, sampler=sampler[k],
                                        pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))
    return data_loader


class NegativeSample(torch.nn.Module):
    def __init__(self, item_attr, num_items, num_negatives):
        super().__init__()
        self.item_attr = item_attr
        self.num_items = num_items
        self.num_negatives = num_negatives

    def forward(self, input):
        positive_item = input['item']
        positive_rating = input['rating']
        negative_item = torch.tensor(list(set(range(self.num_items)) - set(positive_item.tolist())),
                                     dtype=torch.long)
        num_negative_random_item = self.num_negatives * len(positive_item)
        negative_item = negative_item[torch.randperm(len(negative_item))][:num_negative_random_item]
        negative_rating = torch.zeros(negative_item.size(0))
        input['item'] = torch.cat([positive_item, negative_item], dim=0)
        input['rating'] = torch.cat([positive_rating, negative_rating], dim=0)
        if self.item_attr is not None:
            negative_item_attr = torch.tensor(self.item_attr[negative_item]).view(-1, input['item_attr'].size(1))
            input['item_attr'] = torch.cat([input['item_attr'], negative_item_attr], dim=0)
        return input


class PairInput(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        input['user'] = input['user'].repeat(input['item'].size(0))
        input['target_user'] = input['target_user'].repeat(input['target_item'].size(0))
        if cfg['info'] == 1:
            if 'user_profile' in input:
                input['user_profile'] = input['user_profile'].view(1, -1).repeat(input['item'].size(0), 1)
                input['target_user_profile'] = input['target_user_profile'].view(1, -1).repeat(
                    input['target_item'].size(0), 1)
        else:
            if 'user_profile' in input:
                del input['user_profile']
                del input['target_user_profile']
            if 'item_attr' in input:
                del input['item_attr']
                if 'target_item_attr' in input:
                    del input['target_item_attr']
        return input


class FlatInput(torch.nn.Module):
    def __init__(self, data_num_items, target_num_items):
        super().__init__()
        self.data_num_items = data_num_items
        self.target_num_items = target_num_items

    def forward(self, input):
        input['user'] = input['user'].repeat(input['item'].size(0))
        input['target_user'] = input['target_user'].repeat(input['target_item'].size(0))
        rating = torch.zeros(self.data_num_items)
        rating[input['item']] = input['rating']
        input['rating'] = rating
        target_rating = torch.full((self.target_num_items,), float('nan'))
        target_rating[input['target_item']] = input['target_rating']
        input['target_rating'] = target_rating
        if 'target_item_attr' in input:
            del input['target_item_attr']
        if cfg['info'] == 1:
            input['item_attr'] = input['item_attr'].sum(dim=0)
        else:
            if 'user_profile' in input:
                del input['user_profile']
                del input['target_user_profile']
            del input['item_attr']
        return input


def split_dataset(dataset):
    if cfg['data_name'] in ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'NFP']:
        if 'genre' in cfg['data_split_mode']:
            num_organizations = cfg['num_organizations']
            data_split_idx = torch.multinomial(torch.tensor(dataset.item_attr), 1).view(-1).numpy()
            data_split = []
            for i in range(num_organizations):
                data_split_i = data_split_idx[:, 0][data_split_idx[:, 1] == i].tolist()
                data_split.append(data_split_i)
        elif 'random' in cfg['data_split_mode']:
            num_items = dataset['train'].num_items
            num_organizations = cfg['num_organizations']
            data_split = list(torch.randperm(num_items).split(num_items // num_organizations))
            data_split = data_split[:num_organizations - 1] + [torch.cat(data_split[num_organizations - 1:])]
        else:
            raise ValueError('Not valid data split mode')
    else:
        raise ValueError('Not valid data name')
    return data_split


def make_split_dataset(data_split):
    num_organizations = len(data_split)
    dataset = []
    for i in range(num_organizations):
        data_split_i = data_split[i]
        dataset_i = fetch_dataset(cfg['data_name'], model_name=cfg['model_name'], data_split=data_split_i,
                                  verbose=False)
        dataset.append(dataset_i)
    return dataset
