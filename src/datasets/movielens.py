import numpy as np
import os
import torch
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file


class ML100K(Dataset):
    data_name = 'ML100K'
    file = [('https://files.grouplens.org/datasets/movielens/ml-100k.zip', '0e33842e24a9c977be4e0107933c0723')]

    def __init__(self, root, split):
        self.root = os.path.expanduser(root)
        self.split = split
        if not check_exists(self.processed_folder):
            self.process()
        self.data = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)), mode='pickle')

    def __getitem__(self, index):
        data = torch.tensor(self.data[index])
        input = {'data': data}
        return input

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        train_set, test_set = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'test.pt'), mode='pickle')
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            filename = os.path.basename(url)
            download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split)
        return fmt_str

    def make_data(self):
        data = np.genfromtxt(os.path.join(self.raw_folder, 'ml-100k', 'u.data'), delimiter='\t')
        user, item, rating = data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2].astype(np.float32)
        idx = np.random.permutation(data.shape[0])
        num_train = int(data.shape[0] * 0.9)
        train_idx, test_idx = idx[:num_train], idx[num_train:]
        user_id, item_id = np.unique(user), np.unique(item)
        m, n = len(user_id), len(item_id)
        user_id_map = {user_id[i]: i for i in range(len(user_id))}
        item_id_map = {item_id[i]: i for i in range(len(item_id))}
        train_data, test_data = np.zeros((m, n), dtype=np.float32), np.zeros((m, n), dtype=np.float32)
        for i in range(len(train_idx)):
            train_data[user_id_map[user[train_idx[i]]], item_id_map[item[train_idx[i]]]] = rating[train_idx[i]]
        for i in range(len(test_idx)):
            test_data[user_id_map[user[test_idx[i]]], item_id_map[item[test_idx[i]]]] = rating[test_idx[i]]
        return train_data, test_data


class ML1M(Dataset):
    data_name = 'ML1M'
    file = [('https://files.grouplens.org/datasets/movielens/ml-1m.zip', 'c4d9eecfca2ab87c1945afe126590906')]

    def __init__(self, root, split):
        self.root = os.path.expanduser(root)
        self.split = split
        if not check_exists(self.processed_folder):
            self.process()
        self.data = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)), mode='pickle')

    def __getitem__(self, index):
        data = torch.tensor(self.data[index])
        input = {'data': data}
        return input

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        train_set, test_set = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'test.pt'), mode='pickle')
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            filename = os.path.basename(url)
            download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split)
        return fmt_str

    def make_data(self):
        data = np.genfromtxt(os.path.join(self.raw_folder, 'ml-1m', 'ratings.dat'), delimiter='::')
        user, item, rating = data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2].astype(np.float32)
        idx = np.random.permutation(data.shape[0])
        num_train = int(data.shape[0] * 0.9)
        train_idx, test_idx = idx[:num_train], idx[num_train:]
        user_id, item_id = np.unique(user), np.unique(item)
        m, n = len(user_id), len(item_id)
        user_id_map = {user_id[i]: i for i in range(len(user_id))}
        item_id_map = {item_id[i]: i for i in range(len(item_id))}
        train_data, test_data = -np.ones((m, n), dtype=np.float32), -np.ones((m, n), dtype=np.float32)
        for i in range(len(train_idx)):
            train_data[user_id_map[user[train_idx[i]]], item_id_map[item[train_idx[i]]]] = rating[train_idx[i]]
        for i in range(len(test_idx)):
            test_data[user_id_map[user[test_idx[i]]], item_id_map[item[test_idx[i]]]] = rating[test_idx[i]]
        return train_data, test_data


class ML10M(Dataset):
    data_name = 'ML10M'
    file = [('https://files.grouplens.org/datasets/movielens/ml-10m.zip', 'ce571fd55effeba0271552578f2648bd')]

    def __init__(self, root, split):
        self.root = os.path.expanduser(root)
        self.split = split
        if not check_exists(self.processed_folder):
            self.process()
        self.data = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)), mode='pickle')

    def __getitem__(self, index):
        data = torch.tensor(self.data[index])
        input = {'data': data}
        return input

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        train_set, test_set = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'test.pt'), mode='pickle')
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            filename = os.path.basename(url)
            download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split)
        return fmt_str

    def make_data(self):
        data = np.genfromtxt(os.path.join(self.raw_folder, 'ml-10M100K', 'ratings.dat'), delimiter='::')
        user, item, rating = data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2].astype(np.float32)
        idx = np.random.permutation(data.shape[0])
        num_train = int(data.shape[0] * 0.9)
        train_idx, test_idx = idx[:num_train], idx[num_train:]
        user_id, item_id = np.unique(user), np.unique(item)
        m, n = len(user_id), len(item_id)
        user_id_map = {user_id[i]: i for i in range(len(user_id))}
        item_id_map = {item_id[i]: i for i in range(len(item_id))}
        train_data, test_data = -np.ones((m, n), dtype=np.float32), -np.ones((m, n), dtype=np.float32)
        for i in range(len(train_idx)):
            train_data[user_id_map[user[train_idx[i]]], item_id_map[item[train_idx[i]]]] = rating[train_idx[i]]
        for i in range(len(test_idx)):
            test_data[user_id_map[user[test_idx[i]]], item_id_map[item[test_idx[i]]]] = rating[test_idx[i]]
        return train_data, test_data


class ML20M(Dataset):
    data_name = 'ML20M'
    file = [('https://files.grouplens.org/datasets/movielens/ml-20m.zip', 'cd245b17a1ae2cc31bb14903e1204af3')]

    def __init__(self, root, split):
        self.root = os.path.expanduser(root)
        self.split = split
        if not check_exists(self.processed_folder):
            self.process()
        self.data = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)), mode='pickle')

    def __getitem__(self, index):
        data = torch.tensor(self.data[index])
        input = {'data': data}
        return input

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        train_set, test_set = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'test.pt'), mode='pickle')
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            filename = os.path.basename(url)
            download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split)
        return fmt_str

    def make_data(self):
        data = np.genfromtxt(os.path.join(self.raw_folder, 'ml-20m', 'ratings.csv'), delimiter=',', skip_header=1)
        user, item, rating = data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2].astype(np.float32)
        idx = np.random.permutation(data.shape[0])
        num_train = int(data.shape[0] * 0.9)
        train_idx, test_idx = idx[:num_train], idx[num_train:]
        user_id, item_id = np.unique(user), np.unique(item)
        m, n = len(user_id), len(item_id)
        user_id_map = {user_id[i]: i for i in range(len(user_id))}
        item_id_map = {item_id[i]: i for i in range(len(item_id))}
        train_data, test_data = -np.ones((m, n), dtype=np.float32), -np.ones((m, n), dtype=np.float32)
        for i in range(len(train_idx)):
            train_data[user_id_map[user[train_idx[i]]], item_id_map[item[train_idx[i]]]] = rating[train_idx[i]]
        for i in range(len(test_idx)):
            test_data[user_id_map[user[test_idx[i]]], item_id_map[item[test_idx[i]]]] = rating[test_idx[i]]
        return train_data, test_data
