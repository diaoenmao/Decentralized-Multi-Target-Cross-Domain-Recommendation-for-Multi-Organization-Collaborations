from config import cfg
from data import fetch_dataset, make_data_loader
from utils import collate, process_dataset, save_img, process_control, resume, to_device
import torch
import models

if __name__ == "__main__":
    process_control()
    cfg['seed'] = 0
    data_name = 'ML10M'
    batch_size = {'train': 10, 'test': 10}
    dataset = fetch_dataset(data_name)
    data_loader = make_data_loader(dataset, cfg['model_name'], batch_size=batch_size)
    for i, input in enumerate(data_loader['train']):
        input = collate(input)
        print(input['data'].shape)
        print(torch.unique(input['data']))
        exit()

# import numpy as np
# from scipy.sparse import csr_matrix
# import torch
#
# if __name__ == "__main__":
#     csr = csr_matrix([[1, 2.5, 0], [0, 0, 3.5], [4.5, 0, 5]])
#     print('csr', csr)
#     coo = csr.tocoo()
#     print('coo', coo)
#     t = torch.sparse_coo_tensor([coo.row.tolist(), coo.col.tolist()],
#                                 coo.data)
#     print('t', t)
#     z = csr[0].todense()
#     print('z', z)
