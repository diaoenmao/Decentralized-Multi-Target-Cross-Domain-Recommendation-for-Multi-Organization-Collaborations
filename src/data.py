import copy
import torch
import numpy as np
import models
from config import cfg
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
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', mode=cfg["data_mode"])'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', mode=cfg["data_mode"])'.format(data_name))
        if cfg['data_mode'] == 'implicit':
            dataset['train'].transform = datasets.Compose([NegativeSample(dataset['train'].num_items, 2)])
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
    def __init__(self, num_items, num_negatives=1):
        super().__init__()
        self.num_items = num_items
        self.num_negatives = num_negatives

    def forward(self, input):
        positive_item = input['item']
        positive_target = input['target']
        if 'semi_user' in input:
            postive_semi_item = input['semi_item']
            positive_full_item = torch.cat([positive_item, postive_semi_item], dim=0)
            negative_item = torch.tensor(list(set(range(self.num_items)) - set(positive_full_item.tolist())),
                                         dtype=torch.long)
            num_negative_random_item = self.num_negatives * len(positive_full_item)
            negative_random_item = negative_item[torch.randperm(len(negative_item))[:num_negative_random_item]]
            negative_random_target = torch.zeros(len(negative_random_item))
            input['user'] = torch.full((len(positive_item) + len(negative_random_item),), input['user'][0])
            input['item'] = torch.cat([positive_item, negative_random_item])
            input['target'] = torch.cat([positive_target, negative_random_target])
        else:
            negative_item = torch.tensor(list(set(range(self.num_items)) - set(positive_item.tolist())),
                                         dtype=torch.long)
            num_negative_random_item = self.num_negatives * len(positive_item)
            negative_item = negative_item[torch.randperm(len(negative_item))[:num_negative_random_item]]
            negative_target = torch.zeros(len(negative_item))
            input['user'] = torch.full((len(positive_item) + len(negative_item),), input['user'][0])
            input['item'] = torch.cat([positive_item, negative_item])
            input['target'] = torch.cat([positive_target, negative_target])
        return input

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class FullDataset(Dataset):
    def __init__(self, dataset, semi_dataset):
        self.data = dataset.data
        self.semi_data = semi_dataset.data
        self.transform = dataset.transform

    def __getitem__(self, index):
        data = self.data[index].tocoo()
        semi_data = self.semi_data[index].tocoo()
        user = np.array(index).reshape(-1)[data.row]
        semi_user = np.array(index).reshape(-1)[semi_data.row]
        input = {'user': torch.tensor(user, dtype=torch.long),
                 'item': torch.tensor(data.col, dtype=torch.long),
                 'target': torch.tensor(data.data),
                 'semi_user': torch.tensor(semi_user, dtype=torch.long),
                 'semi_item': torch.tensor(semi_data.col, dtype=torch.long),
                 'semi_target': torch.tensor(semi_data.data, dtype=torch.long)}
        if self.transform is not None:
            input = self.transform(input)
        return input

    def __len__(self):
        return self.data.shape[0]
