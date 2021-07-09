from config import cfg
from data import fetch_dataset, make_data_loader
from utils import collate, process_dataset, save_img, process_control, resume, to_device
import torch
import models

if __name__ == "__main__":
    data_name = 'MovieLens100K'
    cfg['batch_size'] = {'train': 10, 'test': 10}
    dataset = fetch_dataset(data_name)
    data_loader = make_data_loader(dataset)
    for i, input in enumerate(data_loader['train']):
        input = collate(input)
        print(input['data'].size(), input['target'].size())
        exit()