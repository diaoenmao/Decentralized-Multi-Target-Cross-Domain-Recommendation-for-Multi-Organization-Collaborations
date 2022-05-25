import numpy as np
import scipy
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file
from scipy.sparse import csr_matrix
from config import cfg


class Douban(Dataset):
    data_name = 'Douban'
    filename = 'archive.zip'  # https://www.kaggle.com/datasets/fengzhujoey/douban-datasetratingreviewside-information
    genre = ['book', 'movie', 'music']

    def __init__(self, root, split, data_mode, target_mode, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.data_mode = data_mode
        self.target_mode = target_mode
        self.transform = transform
        if not check_exists(self.processed_folder):
            self.process()
        self.data, self.target = load(os.path.join(self.processed_folder, self.target_mode, '{}.pt'.format(self.split)),
                                      mode='pickle')
        if self.data_mode == 'user':
            pass
        elif self.data_mode == 'item':
            data_coo = self.data.tocoo()
            target_coo = self.target.tocoo()
            self.data = csr_matrix((data_coo.data, (data_coo.col, data_coo.row)),
                                   shape=(self.data.shape[1], self.data.shape[0]))
            self.target = csr_matrix((target_coo.data, (target_coo.col, target_coo.row)),
                                     shape=(self.target.shape[1], self.target.shape[0]))
        else:
            raise ValueError('Not valid data mode')
        user_profile = load(os.path.join(self.processed_folder, 'user_profile.pt'), mode='pickle')
        self.user_profile = {'data': user_profile, 'target': user_profile}
        item_attr = load(os.path.join(self.processed_folder, 'item_attr.pt'), mode='pickle')
        self.item_attr = {'data': item_attr, 'target': item_attr}

    def __getitem__(self, index):
        data = self.data[index].tocoo()
        target = self.target[index].tocoo()
        if self.data_mode == 'user':
            input = {'user': torch.tensor(np.array([index]), dtype=torch.long),
                     'item': torch.tensor(data.col, dtype=torch.long),
                     'rating': torch.tensor(data.data),
                     'target_user': torch.tensor(np.array([index]), dtype=torch.long),
                     'target_item': torch.tensor(target.col, dtype=torch.long),
                     'target_rating': torch.tensor(target.data)}
            if 'data' in self.user_profile:
                input['user_profile'] = torch.tensor(self.user_profile['data'][index])
            if 'target' in self.user_profile:
                input['target_user_profile'] = torch.tensor(self.user_profile['target'][index])
            if 'data' in self.item_attr:
                input['item_attr'] = torch.tensor(self.item_attr['data'][data.col])
            if 'target' in self.item_attr:
                input['target_item_attr'] = torch.tensor(self.item_attr['target'][target.col])
        elif self.data_mode == 'item':
            input = {'user': torch.tensor(data.col, dtype=torch.long),
                     'item': torch.tensor(np.array([index]), dtype=torch.long),
                     'rating': torch.tensor(data.data),
                     'target_user': torch.tensor(target.col, dtype=torch.long),
                     'target_item': torch.tensor(np.array([index]), dtype=torch.long),
                     'target_rating': torch.tensor(target.data)}
            if 'data' in self.user_profile:
                input['user_profile'] = torch.tensor(self.user_profile['data'][data.col])
            if 'target' in self.user_profile:
                input['target_user_profile'] = torch.tensor(self.user_profile['target'][target.col])
            if 'data' in self.item_attr:
                input['item_attr'] = torch.tensor(self.item_attr['data'][index])
            if 'target' in self.item_attr:
                input['target_item_attr'] = torch.tensor(self.item_attr['target'][index])
        else:
            raise ValueError('Not valid data mode')
        if self.transform is not None:
            input = self.transform(input)
        return input

    def __len__(self):
        if self.data_mode == 'user':
            len_ = self.num_users['data']
        elif self.data_mode == 'item':
            len_ = self.num_items['data']
        else:
            raise ValueError('Not valid data mode')
        return len_

    @property
    def num_users(self):
        if self.data_mode == 'user':
            num_users_ = {'data': self.data.shape[0], 'target': self.target.shape[0]}
        elif self.data_mode == 'item':
            num_users_ = {'data': self.data.shape[1], 'target': self.target.shape[1]}
        else:
            raise ValueError('Not valid data mode')
        return num_users_

    @property
    def num_items(self):
        if self.data_mode == 'user':
            num_items_ = {'data': self.data.shape[1], 'target': self.target.shape[1]}
        elif self.data_mode == 'item':
            num_items_ = {'data': self.data.shape[0], 'target': self.target.shape[0]}
        else:
            raise ValueError('Not valid data mode')
        return num_items_

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
        user_profile, item_attr = self.make_info()
        save(user_profile, os.path.join(self.processed_folder, 'user_profile.pt'), mode='pickle')
        save(item_attr, os.path.join(self.processed_folder, 'item_attr.pt'), mode='pickle')
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        extract_file(os.path.join(self.raw_folder, self.filename))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split)
        return fmt_str

    def make_explicit_data(self):
        user = []
        item = []
        rating = []
        for i in range(len(self.genre)):
            data_i = pd.read_csv(os.path.join(self.raw_folder, 'douban_dataset(text information)',
                                              '{}reviews_cleaned.txt'.format(self.genre[i])), delimiter='\t')
            user_i = data_i.iloc[:, 0].to_numpy()
            item_i = data_i.iloc[:, 1].to_numpy()
            rating_i = data_i.iloc[:, 2].astype(np.float32)
            user.append(user_i)
            if i > 0:
                item_i = item_i + len(item[i - 1])
            item.append(item_i)
            rating.append(rating_i)
        # matched_user_id = []
        # for i in range(len(self.genre)):
        #     for j in range(i + 1, len(self.genre)):
        #         matched_userid_i_j = set(user[i].tolist()).intersection(set(user[j].tolist()))
        #         matched_user_id.append(matched_userid_i_j)
        # matched_user_id = np.array(list(set.intersection(*matched_user_id)))
        user = np.concatenate(user, axis=0)
        item = np.concatenate(item, axis=0)
        rating = np.concatenate(rating, axis=0)
        # user_id, user_inv = np.unique(user, return_inverse=True)
        # _, _, unique_matched_user_inv = np.intersect1d(matched_user_id, user_id, return_indices=True)
        # mask = np.isin(user_inv, unique_matched_user_inv)
        # user = user[mask]
        # item = item[mask]
        # rating = rating[mask]
        user_id, user_inv = np.unique(user, return_inverse=True)
        item_id, item_inv = np.unique(item, return_inverse=True)
        M, N = len(user_id), len(item_id)
        user_id_map = {user_id[i]: i for i in range(len(user_id))}
        item_id_map = {item_id[i]: i for i in range(len(item_id))}
        user = np.array([user_id_map[i] for i in user_id], dtype=np.int64)[user_inv].reshape(user.shape)
        item = np.array([item_id_map[i] for i in item_id], dtype=np.int64)[item_inv].reshape(user.shape)
        idx = np.random.permutation(user.shape[0])
        num_train = int(user.shape[0] * 0.9)
        train_idx, test_idx = idx[:num_train], idx[num_train:]
        train_user, train_item, train_rating = user[train_idx], item[train_idx], rating[train_idx]
        test_user, test_item, test_rating = user[test_idx], item[test_idx], rating[test_idx]
        train_data = csr_matrix((train_rating, (train_user, train_item)), shape=(M, N))
        train_target = train_data
        test_data = train_data
        test_target = csr_matrix((test_rating, (test_user, test_item)), shape=(M, N))
        return (train_data, train_target), (test_data, test_target)

    def make_implicit_data(self):
        user = []
        item = []
        rating = []
        for i in range(len(self.genre)):
            data_i = pd.read_csv(os.path.join(self.raw_folder, 'douban_dataset(text information)',
                                              '{}reviews_cleaned.txt'.format(self.genre[i])), delimiter='\t')
            user_i = data_i.iloc[:, 0].to_numpy()
            item_i = data_i.iloc[:, 1].to_numpy()
            rating_i = data_i.iloc[:, 2].astype(np.float32)
            user.append(user_i)
            if i > 0:
                item_i = item_i + len(item[i - 1])
            item.append(item_i)
            rating.append(rating_i)
        # matched_user_id = []
        # for i in range(len(self.genre)):
        #     for j in range(i + 1, len(self.genre)):
        #         matched_userid_i_j = set(user[i].tolist()).intersection(set(user[j].tolist()))
        #         matched_user_id.append(matched_userid_i_j)
        # matched_user_id = np.array(list(set.intersection(*matched_user_id)))
        user = np.concatenate(user, axis=0)
        item = np.concatenate(item, axis=0)
        rating = np.concatenate(rating, axis=0)
        # user_id, user_inv = np.unique(user, return_inverse=True)
        # _, _, unique_matched_user_inv = np.intersect1d(matched_user_id, user_id, return_indices=True)
        # mask = np.isin(user_inv, unique_matched_user_inv)
        # user = user[mask]
        # item = item[mask]
        # rating = rating[mask]
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
        train_target = train_data
        test_data = train_data
        test_target = csr_matrix((test_rating, (test_user, test_item)), shape=(M, N))
        return (train_data, train_target), (test_data, test_target)

    def make_info(self):
        user = []
        item = []
        left_pivot = []
        right_pivot = []
        for i in range(len(self.genre)):
            if len(left_pivot) > 0:
                left_pivot.append(right_pivot[i - 1])
            else:
                left_pivot.append(0)
            data_i = pd.read_csv(os.path.join(self.raw_folder, 'douban_dataset(text information)',
                                              '{}reviews_cleaned.txt'.format(self.genre[i])), delimiter='\t')
            user_i = data_i.iloc[:, 0].to_numpy()
            item_i = data_i.iloc[:, 1].to_numpy()
            user.append(user_i)
            if i > 0:
                item_i = item_i + len(item[i - 1])
            item.append(item_i)
            right_pivot.append(left_pivot[i] + len(item_i))
        # matched_user_id = []
        # for i in range(len(self.genre)):
        #     for j in range(i + 1, len(self.genre)):
        #         matched_userid_i_j = set(user[i].tolist()).intersection(set(user[j].tolist()))
        #         matched_user_id.append(matched_userid_i_j)
        # matched_user_id = np.array(list(set.intersection(*matched_user_id)))
        user = np.concatenate(user, axis=0)
        item = np.concatenate(item, axis=0)
        user_id, user_inv = np.unique(user, return_inverse=True)
        # _, _, unique_matched_user_inv = np.intersect1d(matched_user_id, user_id, return_indices=True)
        # mask = np.in1d(user_inv, unique_matched_user_inv)
        # item_id, item_inv = np.unique(item[mask], return_inverse=True)
        item_id, item_inv = np.unique(item, return_inverse=True)
        item_attr = np.zeros((len(item_id), len(self.genre)), dtype=np.float32)
        for i in range(len(self.genre)):
            item_inv_i = np.intersect1d(item_id, item[left_pivot[i]:right_pivot[i]], return_indices=True)[1]
            item_attr[item_inv_i, i] += 1
        user_info = pd.read_csv(os.path.join(self.raw_folder, 'douban_dataset(text information)', 'users_cleaned.txt'),
                                delimiter='\t')
        user_id_info = user_info.iloc[:, -1].to_numpy()
        living_place = user_info.iloc[:, 1].to_numpy()
        _, user_id_info_idx, _ = np.intersect1d(user_id_info, user_id, return_indices=True)
        # user_id_info = user_id_info[user_id_info_idx]
        living_place = living_place[user_id_info_idx].tolist()
        provinces = ['北京', '天津', '河北', '山西', '内蒙古', '辽宁', '吉林', '黑龙江', '上海', '江苏', '浙江', '安徽', '福建', '江西', '山东', '河南',
                     '湖北', '湖南', '广东', '广西', '海南', '重庆', '四川', '贵州', '云南', '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆', '台湾',
                     '香港', '澳门']
        code_idx = -1
        visited_provinces = {}
        for i in range(len(living_place)):
            check_in = [p in str(living_place[i]) for p in provinces]
            if any(check_in):
                visiting_p = provinces[np.where(check_in)[0].item()]
                if visiting_p not in visited_provinces:
                    code_idx += 1
                    visited_provinces[visiting_p] = code_idx
                code_idx_i = visited_provinces[visiting_p]
                living_place[i] = code_idx_i
            else:
                living_place[i] = -1
        living_place = np.array(living_place)
        living_place[living_place == -1] = code_idx + 1
        living_place_id, living_place_inv = np.unique(living_place, return_inverse=True)
        user_profile = np.zeros((len(user_id), len(living_place_id)), dtype=np.float32)
        for i in range(len(living_place_id)):
            mask_i = living_place == living_place_id[i]
            user_profile[mask_i, i] += 1
        return user_profile, item_attr