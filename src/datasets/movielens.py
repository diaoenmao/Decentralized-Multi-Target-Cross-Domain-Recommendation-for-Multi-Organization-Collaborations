import numpy as np
import scipy
import os
import torch
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file
from scipy.sparse import csr_matrix


class ML100K(Dataset):
    data_name = 'ML100K'
    file = [('https://files.grouplens.org/datasets/movielens/ml-100k.zip', '0e33842e24a9c977be4e0107933c0723')]

    def __init__(self, root, split, mode, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.mode = mode
        self.transform = transform
        # if not check_exists(self.processed_folder):
        self.process()
        self.data = load(os.path.join(self.processed_folder, self.mode, '{}.pt'.format(self.split)), mode='pickle')
        self.num_users, self.num_items = self.data.shape

    def __getitem__(self, index):
        data = self.data[index].tocoo()
        user = np.array(index).reshape(-1)[data.row]
        input = {'user': torch.tensor(user, dtype=torch.long), 'item': torch.tensor(data.col, dtype=torch.long),
                 'target': torch.tensor(data.data)}
        if self.transform is not None:
            input = self.transform(input)
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
        train_set, test_set, user_profile, item_attr = self.make_explicit_data()
        save(train_set, os.path.join(self.processed_folder, 'explicit', 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'explicit', 'test.pt'), mode='pickle')
        save(user_profile, os.path.join(self.processed_folder, 'explicit', 'user_profile.pt'), mode='pickle')
        save(item_attr, os.path.join(self.processed_folder, 'explicit', 'item_attr.pt'), mode='pickle')
        train_set, test_set = self.make_implicit_data()
        save(train_set, os.path.join(self.processed_folder, 'implicit', 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'implicit', 'test.pt'), mode='pickle')
        save(user_profile, os.path.join(self.processed_folder, 'implicit', 'user_profile.pt'), mode='pickle')
        save(item_attr, os.path.join(self.processed_folder, 'implicit', 'item_attr.pt'), mode='pickle')
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

    def make_explicit_data(self):
        import pandas as pd
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        user_profile = pd.read_csv(os.path.join(self.raw_folder, 'ml-100k', 'u.user'), delimiter='|',
                                   names=['id', 'age', 'gender', 'occupation', 'zipcode'])
        age = user_profile['age'].to_numpy().astype(np.int64)
        age[age <= 17] = 0
        age[(age >= 18) & (age <= 24)] = 1
        age[(age >= 25) & (age <= 34)] = 2
        age[(age >= 35) & (age <= 44)] = 3
        age[(age >= 45) & (age <= 49)] = 4
        age[(age >= 50) & (age <= 55)] = 5
        age[age >= 56] = 6
        age = np.eye(7)[age]
        gender = le.fit_transform(user_profile['gender'].to_numpy()).astype(np.int64)
        gender = np.eye(len(le.classes_))[gender]
        occupation = le.fit_transform(user_profile['occupation'].to_numpy()).astype(np.int64)
        occupation = np.eye(len(le.classes_))[occupation]
        user_profile = np.hstack([age, gender, occupation])
        item_attr = pd.read_csv(os.path.join(self.raw_folder, 'ml-100k', 'u.item'), delimiter='|', header=None)
        genre = item_attr.iloc[:, 5:].to_numpy().astype(np.int64)
        time = le.fit_transform(pd.to_datetime(item_attr.iloc[:, 2]).dt.year.to_numpy()).astype(np.int64)
        time = np.eye(len(le.classes_))[time]
        item_attr = np.hstack([genre, time])
        data = np.genfromtxt(os.path.join(self.raw_folder, 'ml-100k', 'u.data'), delimiter='\t')
        user, item, rating = data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2].astype(np.float32)
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
        return train_data, test_data, user_profile, item_attr

    def make_implicit_data(self):
        import pandas as pd
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        user_profile = pd.read_csv(os.path.join(self.raw_folder, 'ml-100k', 'u.user'), delimiter='|',
                                   names=['id', 'age', 'gender', 'occupation', 'zipcode'])
        age = user_profile['age'].to_numpy().astype(np.int64)
        age[age <= 17] = 0
        age[(age >= 18) & (age <= 24)] = 1
        age[(age >= 25) & (age <= 34)] = 2
        age[(age >= 35) & (age <= 44)] = 3
        age[(age >= 45) & (age <= 49)] = 4
        age[(age >= 50) & (age <= 55)] = 5
        age[age >= 56] = 6
        age = np.eye(7)[age]
        gender = le.fit_transform(user_profile['gender'].to_numpy()).astype(np.int64)
        gender = np.eye(len(le.classes_))[gender]
        occupation = le.fit_transform(user_profile['occupation'].to_numpy()).astype(np.int64)
        occupation = np.eye(len(le.classes_))[occupation]
        user_profile = np.hstack([age, gender, occupation])
        item_attr = pd.read_csv(os.path.join(self.raw_folder, 'ml-100k', 'u.item'), delimiter='|', header=None)
        genre = item_attr.iloc[:, 5:].to_numpy().astype(np.int64)
        time = le.fit_transform(pd.to_datetime(item_attr.iloc[:, 2]).dt.year.to_numpy()).astype(np.int64)
        time = np.eye(len(le.classes_))[time]
        item_attr = np.hstack([genre, time])
        data = np.genfromtxt(os.path.join(self.raw_folder, 'ml-100k', 'u.data'), delimiter='\t')
        user, item, rating, ts = data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2].astype(
            np.float32), data[:, 3].astype(np.float32)
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
        random_rating = np.zeros(random_user.shape[0])
        train_ts = csr_matrix((train_ts, (train_user, train_item)), shape=(M, N))
        withheld_user = np.arange(M)
        withheld_item = np.asarray(train_ts.argmax(axis=1)).reshape(-1)
        withheld_rating = np.ones(M)
        train_data[withheld_user, withheld_item] = 0
        train_data.eliminate_zeros()
        test_user = np.concatenate([withheld_user, random_user], axis=0)
        test_item = np.concatenate([withheld_item, random_item], axis=0)
        test_rating = np.concatenate([withheld_rating, random_rating], axis=0).astype(np.float32)
        test_data = csr_matrix((test_rating, (test_user, test_item)), shape=(M, N))
        return train_data, test_data, user_profile, item_attr


class ML1M(Dataset):
    data_name = 'ML1M'
    file = [('https://files.grouplens.org/datasets/movielens/ml-1m.zip', 'c4d9eecfca2ab87c1945afe126590906')]

    def __init__(self, root, split, mode, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.mode = mode
        self.transform = transform
        if not check_exists(self.processed_folder):
            self.process()
        self.data = load(os.path.join(self.processed_folder, self.mode, '{}.pt'.format(self.split)), mode='pickle')
        self.num_users, self.num_items = self.data.shape

    def __getitem__(self, index):
        data = self.data[index].tocoo()
        user = np.array(index).reshape(-1)[data.row]
        input = {'user': torch.tensor(user, dtype=torch.long), 'item': torch.tensor(data.col, dtype=torch.long),
                 'target': torch.tensor(data.data)}
        if self.transform is not None:
            input = self.transform(input)
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

    def make_explicit_data(self):
        data = np.genfromtxt(os.path.join(self.raw_folder, 'ml-1m', 'ratings.dat'), delimiter='::')
        user, item, rating = data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2].astype(np.float32)
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
        data = np.genfromtxt(os.path.join(self.raw_folder, 'ml-1m', 'ratings.dat'), delimiter='::')
        user, item, rating, ts = data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2].astype(
            np.float32), data[:, 3].astype(np.float32)
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
        random_rating = np.zeros(random_user.shape[0])
        train_ts = csr_matrix((train_ts, (train_user, train_item)), shape=(M, N))
        withheld_user = np.arange(M)
        withheld_item = np.asarray(train_ts.argmax(axis=1)).reshape(-1)
        withheld_rating = np.ones(M)
        train_data[withheld_user, withheld_item] = 0
        train_data.eliminate_zeros()
        test_user = np.concatenate([withheld_user, random_user], axis=0)
        test_item = np.concatenate([withheld_item, random_item], axis=0)
        test_rating = np.concatenate([withheld_rating, random_rating], axis=0).astype(np.float32)
        test_data = csr_matrix((test_rating, (test_user, test_item)), shape=(M, N))
        return train_data, test_data


class ML10M(Dataset):
    data_name = 'ML10M'
    file = [('https://files.grouplens.org/datasets/movielens/ml-10m.zip', 'ce571fd55effeba0271552578f2648bd')]

    def __init__(self, root, split, mode, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.mode = mode
        self.transform = transform
        if not check_exists(self.processed_folder):
            self.process()
        self.data = load(os.path.join(self.processed_folder, self.mode, '{}.pt'.format(self.split)), mode='pickle')
        self.num_users, self.num_items = self.data.shape

    def __getitem__(self, index):
        data = self.data[index].tocoo()
        user = np.array(index).reshape(-1)[data.row]
        input = {'user': torch.tensor(user, dtype=torch.long), 'item': torch.tensor(data.col, dtype=torch.long),
                 'target': torch.tensor(data.data)}
        if self.transform is not None:
            input = self.transform(input)
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

    def make_explicit_data(self):
        data = np.genfromtxt(os.path.join(self.raw_folder, 'ml-10M100K', 'ratings.dat'), delimiter='::')
        user, item, rating = data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2].astype(np.float32)
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
        data = np.genfromtxt(os.path.join(self.raw_folder, 'ml-10M100K', 'ratings.dat'), delimiter='::')
        user, item, rating, ts = data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2].astype(
            np.float32), data[:, 3].astype(np.float32)
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
        random_rating = np.zeros(random_user.shape[0])
        train_ts = csr_matrix((train_ts, (train_user, train_item)), shape=(M, N))
        withheld_user = np.arange(M)
        withheld_item = np.asarray(train_ts.argmax(axis=1)).reshape(-1)
        withheld_rating = np.ones(M)
        train_data[withheld_user, withheld_item] = 0
        train_data.eliminate_zeros()
        test_user = np.concatenate([withheld_user, random_user], axis=0)
        test_item = np.concatenate([withheld_item, random_item], axis=0)
        test_rating = np.concatenate([withheld_rating, random_rating], axis=0).astype(np.float32)
        test_data = csr_matrix((test_rating, (test_user, test_item)), shape=(M, N))
        return train_data, test_data


class ML20M(Dataset):
    data_name = 'ML20M'
    file = [('https://files.grouplens.org/datasets/movielens/ml-20m.zip', 'cd245b17a1ae2cc31bb14903e1204af3')]

    def __init__(self, root, split, mode, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.mode = mode
        self.transform = transform
        if not check_exists(self.processed_folder):
            self.process()
        self.data = load(os.path.join(self.processed_folder, self.mode, '{}.pt'.format(self.split)), mode='pickle')
        self.num_users, self.num_items = self.data.shape

    def __getitem__(self, index):
        data = self.data[index].tocoo()
        user = np.array(index).reshape(-1)[data.row]
        input = {'user': torch.tensor(user, dtype=torch.long), 'item': torch.tensor(data.col, dtype=torch.long),
                 'target': torch.tensor(data.data)}
        if self.transform is not None:
            input = self.transform(input)
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

    def make_explicit_data(self):
        data = np.genfromtxt(os.path.join(self.raw_folder, 'ml-20m', 'ratings.csv'), delimiter=',', skip_header=1)
        user, item, rating = data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2].astype(np.float32)
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
        data = np.genfromtxt(os.path.join(self.raw_folder, 'ml-20m', 'ratings.csv'), delimiter=',', skip_header=1)
        user, item, rating, ts = data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2].astype(
            np.float32), data[:, 3].astype(np.float32)
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
        random_rating = np.zeros(random_user.shape[0])
        train_ts = csr_matrix((train_ts, (train_user, train_item)), shape=(M, N))
        withheld_user = np.arange(M)
        withheld_item = np.asarray(train_ts.argmax(axis=1)).reshape(-1)
        withheld_rating = np.ones(M)
        train_data[withheld_user, withheld_item] = 0
        train_data.eliminate_zeros()
        test_user = np.concatenate([withheld_user, random_user], axis=0)
        test_item = np.concatenate([withheld_item, random_item], axis=0)
        test_rating = np.concatenate([withheld_rating, random_rating], axis=0).astype(np.float32)
        test_data = csr_matrix((test_rating, (test_user, test_item)), shape=(M, N))
        return train_data, test_data
