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


def fetch_dataset(data_name):
    import datasets
    dataset = {}
    print('fetching data {}...'.format(data_name))
    root = './data/{}'.format(data_name)
    if data_name in ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'NFP']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', data_mode=cfg["data_mode"])'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', data_mode=cfg["data_mode"])'.format(data_name))
        if cfg['model_name'] in ['base', 'mf', 'gmf', 'mlp', 'nmf']:
            if cfg['data_mode'] == 'explicit':
                dataset['train'].transform = datasets.Compose([PairInput()])
                dataset['test'].transform = datasets.Compose([PairInput()])
            elif cfg['data_mode'] == 'implicit':
                dataset['train'].transform = datasets.Compose(
                    [NegativeSample(dataset['train'].item_attr, dataset['train'].num_items, cfg['num_negatives']),
                     PairInput()])
                dataset['test'].transform = datasets.Compose(
                    [PairInput()])
            else:
                raise ValueError('Not valid data mode')
        elif cfg['model_name'] in ['ae']:
            if cfg['data_mode'] == 'explicit':
                dataset['train'].transform = datasets.Compose([FlatInput(dataset['train'].num_items, 0)])
                dataset['test'].transform = datasets.Compose([FlatInput(dataset['train'].num_items, 0)])
            elif cfg['data_mode'] == 'implicit':
                dataset['train'].transform = datasets.Compose(
                    [FlatInput(dataset['train'].num_items, cfg['num_negatives'])])
                dataset['test'].transform = datasets.Compose(
                    [FlatInput(dataset['train'].num_items, 0)])
            else:
                raise ValueError('Not valid data mode')
        else:
            raise ValueError('Not valid model name')
    else:
        raise ValueError('Not valid dataset name')
    print('data ready')
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


def split_dataset(dataset, num_users, data_split_mode):
    data_split = {}
    if data_split_mode == 'iid':
        data_split['train'], target_split = iid(dataset['train'], num_users)
        data_split['test'], _ = iid(dataset['test'], num_users)
    elif 'non-iid' in cfg['data_split_mode']:
        data_split['train'], target_split = non_iid(dataset['train'], num_users)
        data_split['test'], _ = non_iid(dataset['test'], num_users)
    else:
        raise ValueError('Not valid data split mode')
    return data_split, target_split


def iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    data_split, idx = {}, list(range(len(dataset)))
    for i in range(num_users):
        num_items_i = min(len(idx), num_items)
        data_split[i] = torch.tensor(idx)[torch.randperm(len(idx))[:num_items_i]].tolist()
        idx = list(set(idx) - set(data_split[i]))
    target_split = [list(range(cfg['target_size'])) for i in range(num_users)]
    return data_split, target_split


def separate_dataset(dataset, idx):
    separated_dataset = copy.deepcopy(dataset)
    separated_dataset.data = [dataset.data[s] for s in idx]
    separated_dataset.target = [dataset.target[s] for s in idx]
    separated_dataset.other['id'] = list(range(len(separated_dataset.data)))
    return separated_dataset


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
        input['item_attr'] = torch.cat([input['item_attr'], torch.tensor(self.item_attr[negative_item])],
                                       dim=0)
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
            del input['item_attr']
            del input['target_item_attr']
        return input


class FlatInput(torch.nn.Module):
    def __init__(self, num_items, num_negatives):
        super().__init__()
        self.num_items = num_items
        self.num_negatives = num_negatives

    def forward(self, input):
        rating = torch.zeros(self.num_items)
        rating[input['item']] = input['rating']
        input['rating'] = rating
        if self.num_negatives == 0:
            target_rating = torch.full((self.num_items,), float('nan'))
            target_rating[input['target_item']] = input['target_rating']
            input['target_rating'] = target_rating
        else:
            target_rating = torch.full((self.num_items,), float('nan'))
            positive_item = input['item']
            negative_item = torch.tensor(list(set(range(self.num_items)) - set(positive_item.tolist())),
                                         dtype=torch.long)
            num_negative_random_item = self.num_negatives * len(positive_item)
            negative_item = negative_item[torch.randperm(len(negative_item))][:num_negative_random_item]
            negative_rating = torch.zeros(negative_item.size(0))
            target_rating[negative_item] = negative_rating
            target_rating[input['target_item']] = input['target_rating']
            input['target_rating'] = target_rating
        del input['item']
        del input['target_item']
        if cfg['info'] == 1:
            input['item_attr'] = input['item_attr'].sum(dim=0)
            input['target_item_attr'] = input['target_item_attr'].sum(dim=0)
        else:
            if 'user_profile' in input:
                del input['user_profile']
                del input['target_user_profile']
            del input['item_attr']
            del input['target_item_attr']
        return input
