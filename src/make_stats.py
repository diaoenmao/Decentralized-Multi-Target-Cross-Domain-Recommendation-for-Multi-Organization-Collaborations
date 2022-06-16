import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import datasets
from config import cfg, process_args
from data import fetch_dataset, make_data_loader
from utils import save, process_control, process_dataset, collate, makedir_exist_ok

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)

if __name__ == "__main__":
    process_control()
    cfg['seed'] = 0
    data_names = ['ML100K', 'ML1M', 'ML10M', 'Douban', 'Amazon']
    with torch.no_grad():
        for data_name in data_names:
            cfg['data_name'] = data_name
            root = os.path.join('data', cfg['data_name'])
            dataset = fetch_dataset(cfg['data_name'], verbose=False)['train']
            stats = {'m': dataset.num_users, 'n': dataset.num_items}
            if hasattr(dataset, 'user_profile'):
                stats['user_profile'] = {k: dataset.user_profile[k].shape[-1] for k in dataset.user_profile}
            if hasattr(dataset, 'item_attr'):
                stats['item_attr'] = {k: dataset.item_attr[k].shape[-1] for k in dataset.item_attr}
            print(cfg['data_name'])
            print(stats)
