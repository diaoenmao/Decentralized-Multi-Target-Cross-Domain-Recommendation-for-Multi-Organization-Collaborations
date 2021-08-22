import numpy as np
import scipy
import os
import torch
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file
from scipy.sparse import csr_matrix


class NFP(Dataset):
    data_name = 'NFP'

    def __init__(self, root, split, data_mode, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.data_mode = data_mode
        self.transform = transform
        if not check_exists(self.processed_folder):
            self.process()
        self.train_data = load(os.path.join(self.processed_folder, self.data_mode, 'train.pt'), mode='pickle')
        self.test_data = load(os.path.join(self.processed_folder, self.data_mode, 'test.pt'), mode='pickle')
        self.num_users, self.num_items = self.train_data.shape

    def __getitem__(self, index):
        if self.split == 'train':
            train_data = self.train_data[index].tocoo()
            input = {'user': torch.tensor(np.array([index]), dtype=torch.long),
                     'item': torch.tensor(train_data.col, dtype=torch.long),
                     'rating': torch.tensor(train_data.data),
                     'target_user': torch.tensor(np.array([index]), dtype=torch.long),
                     'target_item': torch.tensor(train_data.col, dtype=torch.long),
                     'target_rating': torch.tensor(train_data.data)}
        elif self.split == 'test':
            train_data = self.train_data[index].tocoo()
            test_data = self.test_data[index].tocoo()
            input = {'user': torch.tensor(np.array([index]), dtype=torch.long),
                     'item': torch.tensor(train_data.col, dtype=torch.long),
                     'rating': torch.tensor(train_data.data),
                     'target_user': torch.tensor(np.array([index]), dtype=torch.long),
                     'target_item': torch.tensor(test_data.col, dtype=torch.long),
                     'target_rating': torch.tensor(test_data.data)}

        else:
            raise ValueError('Not valid load mode')
        if self.transform is not None:
            input = self.transform(input)
        return input

    def __len__(self):
        return self.num_users

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        train_set, test_set = self.make_explicit_data()
        save(train_set, os.path.join(self.processed_folder, 'explicit', 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'explicit', 'test.pt'), mode='pickle')
        train_set, test_set = self.make_implicit_data()
        save(train_set, os.path.join(self.processed_folder, 'implicit', 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'implicit', 'test.pt'), mode='pickle')
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split)
        return fmt_str

    def make_explicit_data(self):
        extract_file(os.path.join(self.raw_folder, 'nf_prize_dataset.tar.gz'))
        extract_file(os.path.join(self.raw_folder, 'download', 'training_set.tar'))
        filenames = os.listdir(os.path.join(self.raw_folder, 'download', 'training_set'))
        user, item, rating = [], [], []
        for i in range(len(filenames)):
            data = np.genfromtxt(os.path.join(self.raw_folder, 'download', 'training_set', filenames[i]), delimiter=',',
                                 skip_header=1)
            user.append(data[:, 0])
            item.append(np.repeat(i, data.shape[0]))
            rating.append(data[:, 1])
        user = np.concatenate(user, axis=0).astype(np.int64)
        item = np.concatenate(item, axis=0).astype(np.int64)
        rating = np.concatenate(rating, axis=0).astype(np.float32)
        user_id, user_inv = np.unique(user, return_inverse=True)
        item_id, item_inv = np.unique(item, return_inverse=True)
        M, N = len(user_id), len(item_id)
        user_id_map = {user_id[i]: i for i in range(len(user_id))}
        item_id_map = {item_id[i]: i for i in range(len(item_id))}
        user = np.array([user_id_map[i] for i in user_id], dtype=np.int64)[user_inv].reshape(user.shape)
        item = np.array([item_id_map[i] for i in item_id], dtype=np.int64)[item_inv].reshape(item.shape)
        idx = np.random.permutation(user.shape[0])
        num_train = int(user.shape[0] * 0.9)
        train_idx, test_idx = idx[:num_train], idx[num_train:]
        train_user, train_item, train_rating = user[train_idx], item[train_idx], rating[train_idx]
        test_user, test_item, test_rating = user[test_idx], item[test_idx], rating[test_idx]
        train_data = csr_matrix((train_rating, (train_user, train_item)), shape=(M, N))
        test_data = csr_matrix((test_rating, (test_user, test_item)), shape=(M, N))
        return train_data, test_data

    def make_implicit_data(self):
        import datetime
        import pandas as pd
        extract_file(os.path.join(self.raw_folder, 'nf_prize_dataset.tar.gz'))
        extract_file(os.path.join(self.raw_folder, 'download', 'training_set.tar'))
        filenames = os.listdir(os.path.join(self.raw_folder, 'download', 'training_set'))
        user, item, rating, ts = [], [], [], []
        for i in range(len(filenames)):
            data = pd.read_csv(os.path.join(self.raw_folder, 'download', 'training_set', filenames[i]), skiprows=[0],
                               header=None)
            user_i = data[0].to_numpy()
            item_i = np.repeat(i, data.shape[0])
            rating_i = data[1].to_numpy()
            ts_i = (pd.to_datetime(data[2], format='%Y-%m-%d').values.astype(np.int64))
            user.append(user_i)
            item.append(item_i)
            rating.append(rating_i)
            ts.append(ts_i)
        user = np.concatenate(user, axis=0).astype(np.int64)
        item = np.concatenate(item, axis=0).astype(np.int64)
        rating = np.concatenate(rating, axis=0).astype(np.float32)
        ts = np.concatenate(ts, axis=0).astype(np.float32)
        user_id, user_inv = np.unique(user, return_inverse=True)
        item_id, item_inv = np.unique(item, return_inverse=True)
        M, N = len(user_id), len(item_id)
        user_id_map = {user_id[i]: i for i in range(len(user_id))}
        item_id_map = {item_id[i]: i for i in range(len(item_id))}
        user = np.array([user_id_map[i] for i in user_id], dtype=np.int64)[user_inv].reshape(user.shape)
        item = np.array([item_id_map[i] for i in item_id], dtype=np.int64)[item_inv].reshape(item.shape)
        rating.fill(1)
        train_user = user
        train_item = item
        train_rating = rating
        train_ts = ts
        train_data = csr_matrix((train_rating, (train_user, train_item)), shape=(M, N))
        random_user = np.arange(M).repeat(100)
        random_item = []
        step_size = 10000
        for i in range(0, M, step_size):
            valid_step_size = min(i + step_size, M) - i
            nonzero_user, nonzero_item = train_data[i:i + valid_step_size].nonzero()
            random_item_i = np.random.rand(valid_step_size, N)
            random_item_i[nonzero_user, nonzero_item] = np.inf
            random_item_i = random_item_i.argsort(axis=1)[:, :100].reshape(-1)
            random_item.append(random_item_i)
        random_item = np.concatenate(random_item, axis=0)
        random_rating = np.zeros(random_user.shape[0], dtype=np.float32)
        train_ts = csr_matrix((train_ts, (train_user, train_item)), shape=(M, N))
        withheld_user = np.arange(M)
        withheld_item = np.asarray(train_ts.argmax(axis=1)).reshape(-1)
        withheld_rating = np.ones(M, dtype=np.float32)
        train_data[withheld_user, withheld_item] = 0
        train_data.eliminate_zeros()
        test_user = np.concatenate([withheld_user, random_user], axis=0)
        test_item = np.concatenate([withheld_item, random_item], axis=0)
        test_rating = np.concatenate([withheld_rating, random_rating], axis=0)
        test_data = csr_matrix((test_rating, (test_user, test_item)), shape=(M, N))
        return train_data, test_data
