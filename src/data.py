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


def fetch_dataset(data_name, model_name=None, verbose=True):
    import datasets

    model_name = cfg['model_name'] if model_name is None else model_name
    dataset = {}
    if verbose:
        print('fetching data {}...'.format(data_name))
    root = './data/{}'.format(data_name)
    if data_name in ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'NFP']:
        dataset['train'] = eval(
            'datasets.{}(root=root, split=\'train\', data_mode=cfg["data_mode"])'.format(data_name))
        dataset['test'] = eval(
            'datasets.{}(root=root, split=\'test\', data_mode=cfg["data_mode"])'.format(data_name))
        if model_name in ['base', 'mf', 'gmf', 'mlp', 'nmf']:
            dataset = make_pair_transform(dataset)
        elif model_name in ['ae']:
            dataset = make_flat_transform(dataset)
        else:
            raise ValueError('Not valid model name')
    else:
        raise ValueError('Not valid dataset name')
    if verbose:
        print('data ready')
    return dataset


def make_pair_transform(dataset):
    import datasets
    if 'train' in dataset:
        dataset['train'].transform = datasets.Compose([PairInput()])
    if 'test' in dataset:
        dataset['test'].transform = datasets.Compose([PairInput()])
    return dataset


def make_flat_transform(dataset):
    import datasets
    if 'train' in dataset:
        dataset['train'].transform = datasets.Compose([FlatInput(dataset['train'].num_items)])
    if 'test' in dataset:
        dataset['test'].transform = datasets.Compose([FlatInput(dataset['test'].num_items)])
    return dataset


def input_collate(batch):
    if isinstance(batch[0], dict):
        return {key: [b[key] for b in batch] for key in batch[0]}
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


class PairInput(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        input['user'] = input['user'].repeat(input['item'].size(0))
        input['target_user'] = input['target_user'].repeat(input['target_item'].size(0))
        if cfg['info'] == 1:
            if 'user_profile' in input:
                input['user_profile'] = input['user_profile'].view(1, -1).repeat(input['item'].size(0), 1)
            if 'target_user_profile' in input:
                input['target_user_profile'] = input['target_user_profile'].view(1, -1).repeat(
                    input['target_item'].size(0), 1)
        else:
            if 'user_profile' in input:
                del input['user_profile']
            if 'target_user_profile' in input:
                del input['target_user_profile']
            if 'item_attr' in input:
                del input['item_attr']
            if 'target_item_attr' in input:
                del input['target_item_attr']
        return input


class FlatInput(torch.nn.Module):
    def __init__(self, num_items):
        super().__init__()
        self.num_items = num_items

    def forward(self, input):
        input['user'] = input['user'].repeat(input['item'].size(0))
        input['target_user'] = input['target_user'].repeat(input['target_item'].size(0))
        rating = torch.zeros(self.num_items['data'])
        rating[input['item']] = input['rating']
        input['rating'] = rating
        target_rating = torch.full((self.num_items['target'],), float('nan'))
        target_rating[input['target_item']] = input['target_rating']
        input['target_rating'] = target_rating
        if cfg['info'] == 1:
            input['item_attr'] = input['item_attr'].sum(dim=0)
            if 'target_user_profile' in input:
                del input['target_user_profile']
            if 'target_item_attr' in input:
                del input['target_item_attr']
        else:
            if 'user_profile' in input:
                del input['user_profile']
            if 'target_user_profile' in input:
                del input['target_user_profile']
            if 'item_attr' in input:
                del input['item_attr']
            if 'target_item_attr' in input:
                del input['target_item_attr']
        return input


def split_dataset(dataset):
    if cfg['data_name'] in ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'NFP']:
        if 'genre' in cfg['data_split_mode']:
            num_organizations = cfg['num_organizations']
            zero_mask = torch.tensor(dataset['train'].item_attr['data']).sum(dim=-1) == 0
            item_attr = torch.tensor(dataset['train'].item_attr['data'])
            item_attr[zero_mask] = 1
            data_split_idx = torch.nonzero(item_attr)
            data_split = []
            for i in range(num_organizations):
                data_split_i = data_split_idx[:, 0][data_split_idx[:, 1] == i].tolist()
                data_split.append(data_split_i)
        elif 'random' in cfg['data_split_mode']:
            num_items = dataset['train'].num_items['data']
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
        dataset_i = fetch_dataset(cfg['data_name'], model_name=cfg['model_name'], verbose=False)
        for k in dataset_i:
            dataset_i[k].data = dataset_i[k].data[:, data_split_i]
            dataset_i[k].target = dataset_i[k].target[:, data_split_i]
            if hasattr(dataset_i[k], 'item_attr'):
                dataset_i[k].item_attr['data'] = dataset_i[k].item_attr['data'][data_split_i]
                dataset_i[k].item_attr['target'] = dataset_i[k].item_attr['target'][data_split_i]
        if cfg['model_name'] in ['base', 'mf', 'gmf', 'mlp', 'nmf']:
            dataset_i = make_pair_transform(dataset_i)
        elif cfg['model_name'] in ['ae']:
            dataset_i = make_flat_transform(dataset_i)
        dataset.append(dataset_i)
    return dataset
