from config import cfg
from data import fetch_dataset, make_data_loader
from utils import collate, process_dataset, save_img, process_control, resume, to_device
import torch
import models

# if __name__ == "__main__":
#     process_control()
#     cfg['seed'] = 0
#     data_name = 'NFP'
#     batch_size = {'train': 10, 'test': 10}
#     dataset = fetch_dataset(data_name)
#     data_loader = make_data_loader(dataset, cfg['model_name'], batch_size=batch_size)
#     for i, input in enumerate(data_loader['train']):
#         input = collate(input)
#         print(input['data'].shape)
#         print(torch.unique(input['data']))
#         exit()

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

# if __name__ == "__main__":
#     process_control()
#     cfg['seed'] = 0
#     data_name = 'ML100K'
#     batch_size = {'train': 10, 'test': 10}
#     dataset = fetch_dataset(data_name)
#     data_loader = make_data_loader(dataset, cfg['model_name'], batch_size=batch_size)
#     for i, input in enumerate(data_loader['train']):
#         a = input['data'][0].tocoo()
#         t1 = torch.sparse_coo_tensor(torch.tensor([a.row.tolist(), a.col.tolist()]), torch.tensor(a.data))
#         b = input['data'][1].tocoo()
#         t2 = torch.sparse_coo_tensor(torch.tensor([a.row.tolist(), a.col.tolist()]), torch.tensor(a.data))
#         c = torch.cat([t1, t2], dim=0)[0]
#         print(c)
#         exit()

if __name__ == "__main__":
    process_control()
    cfg['seed'] = 0
    data_name = 'ML100K'
    batch_size = {'train': 10, 'test': 10}
    dataset = fetch_dataset(data_name)
    data_loader = make_data_loader(dataset, cfg['model_name'], batch_size=batch_size)
    for i, input in enumerate(data_loader['train']):
        input = collate(input)
        print(torch.unique(input['target']))
        print(input['user'].size(), input['item'].size(), input['target'].size())
        exit()


# if __name__ == "__main__":
#     process_control()
#     cfg['seed'] = 0
#     data_names = ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'NFP']
#     batch_size = {'train': 10, 'test': 10}
#     for data_name in data_names:
#         print(data_name)
#         dataset = fetch_dataset(data_name)
#         data_loader = make_data_loader(dataset, cfg['model_name'], batch_size=batch_size)
#         for i, input in enumerate(data_loader['train']):
#             input = collate(input)
#             print(input['user'][0].dtype, input['item'][0].dtype, input['target'][0].dtype)
#             break
#         for i, input in enumerate(data_loader['test']):
#             input = collate(input)
#             print(input['user'][0].dtype, input['item'][0].dtype, input['target'][0].dtype)
#             break


# import numpy as np
# from scipy.sparse import csr_matrix
# import torch
#
# if __name__ == "__main__":
#     data = [[1, 2.5, 0], [0, 0, 3.5], [4.5, 0, 5]]
#     csr = csr_matrix(data)
#     print('csr', csr)
#     print('csr', csr[2])
#     print('coo', csr[2].tocoo().col)
#     print('coo', csr[2].tocoo().data)

# import numpy as np
# from scipy.sparse import csr_matrix
# import torch
#
# if __name__ == "__main__":
#     data = [[0, 1, 0], [0, 0, 0], [4.5, 0, 5]]
#     csr = csr_matrix(data)
#     print('csr', csr)
#     print(csr[:3].tocoo().row)
#     print(csr[:3].tocoo().col)
