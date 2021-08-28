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

    def __init__(self, root, split, data_mode, data_split=None, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.data_mode = data_mode
        self.data_split = data_split
        self.transform = transform
        if not check_exists(self.processed_folder):
            self.process()
        self.train_data = load(os.path.join(self.processed_folder, self.data_mode, 'train.pt'), mode='pickle')
        self.test_data = load(os.path.join(self.processed_folder, self.data_mode, 'test.pt'), mode='pickle')
        if data_split is not None:
            self.train_data = self.train_data[:, data_split]
            self.test_data = self.test_data[:, data_split]
        self.item_attr = None
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
        extract_file(os.path.join(self.raw_folder, 'nf_prize_dataset.tar.gz'))
        extract_file(os.path.join(self.raw_folder, 'download', 'training_set.tar'))
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
        train_rating[train_rating < 3.5] = 0
        train_rating[train_rating >= 3.5] = 1
        test_user, test_item, test_rating = user[test_idx], item[test_idx], rating[test_idx]
        test_rating[test_rating < 3.5] = 0
        test_rating[test_rating >= 3.5] = 1
        train_data = csr_matrix((train_rating, (train_user, train_item)), shape=(M, N))
        test_data = csr_matrix((test_rating, (test_user, test_item)), shape=(M, N))
        return train_data, test_data
