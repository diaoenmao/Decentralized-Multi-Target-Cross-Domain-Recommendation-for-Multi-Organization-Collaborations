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

    def __init__(self, root, split, mode):
        self.root = os.path.expanduser(root)
        self.split = split
        self.mode = mode
        if not check_exists(self.processed_folder):
            self.process()
        self.data = load(os.path.join(self.processed_folder, self.mode, '{}.pt'.format(self.split)), mode='pickle')

    def __getitem__(self, index):
        data = torch.tensor(self.data[index].todense())[0]
        input = {'data': data}
        return input

    def __len__(self):
        return self.data.shape[0]

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
        rating[:] = 1
        test_item = []
        withheld_idxs = []
        for i in range(len(user_id)):
            withheld_idx = np.argmax(ts[user == i])
            withheld_item = np.array([item[withheld_idx]])
            random_item = list(set(range(N)) - set(item[user == i].tolist()))
            random_item = np.random.choice(np.array(list(random_item)), 100, replace=False)
            test_item_i = np.concatenate([withheld_item, random_item], axis=0)
            test_item.append(test_item_i)
            withheld_idxs.append(withheld_idx)
        user = np.delete(user, withheld_idxs, axis=0)
        item = np.delete(item, withheld_idxs, axis=0)
        rating = np.delete(rating, withheld_idxs, axis=0)
        train_user = user
        train_item = item
        train_rating = rating
        test_user = np.arange(M).repeat(101)
        test_item = np.concatenate(test_item, axis=0)
        test_rating = np.zeros(101, dtype=np.float32)
        test_rating[0] = 1
        test_rating = np.tile(test_rating, M)
        train_data = csr_matrix((train_rating, (train_user, train_item)), shape=(M, N))
        test_data = csr_matrix((test_rating, (test_user, test_item)), shape=(M, N))
        return train_data, test_data
