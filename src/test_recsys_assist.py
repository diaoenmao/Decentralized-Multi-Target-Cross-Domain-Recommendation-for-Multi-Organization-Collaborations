import argparse
import copy
import datetime
import models
import os
import shutil
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from data import fetch_dataset, make_data_loader, split_dataset, make_split_dataset
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import make_logger
from assist import Assist
from scipy.sparse import csr_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)
    if cfg['target_mode'] == 'explicit':
        metric = Metric({'train': ['Loss', 'RMSE'], 'test': ['Loss', 'RMSE']})
    elif cfg['target_mode'] == 'implicit':
        metric = Metric({'train': ['Loss', 'MAP'], 'test': ['Loss', 'MAP']})
    else:
        raise ValueError('Not valid target mode')
    result = resume(cfg['model_tag'])
    last_epoch = result['epoch']
    data_split = result['data_split']
    dataset = make_split_dataset(data_split)
    dataset = [{'test': dataset[i]['test']} for i in range(len(dataset))]
    assist = result['assist']
    assist.reset()
    organization = result['organization']
    test_logger = make_logger('output/runs/test_{}'.format(cfg['model_tag']))
    test_each_logger = make_logger('output/runs/test_{}'.format(cfg['model_tag']))
    initialize(dataset, assist, organization, metric, test_logger, 0, test_each_logger)
    for epoch in range(1, last_epoch):
        test_logger.safe(True)
        test_each_logger.safe(True)
        dataset = assist.make_dataset(dataset, epoch)
        organization_outputs = gather(dataset, organization, epoch)
        assist.update(organization_outputs, epoch)
        test(assist, metric, test_logger, epoch, test_each_logger)
        test_logger.reset()
    assist.reset()
    result = resume(cfg['model_tag'], load_tag='checkpoint')
    train_logger = result['logger'] if 'logger' in result else None
    save_result = {'cfg': cfg, 'epoch': last_epoch, 'assist': assist,
                   'logger': {'train': train_logger, 'test': test_logger, 'test_each': test_each_logger}}
    save(save_result, './output/result/{}.pt'.format(cfg['model_tag']))
    return


def initialize(dataset, assist, organization, metric, logger, epoch, each_logger):
    logger.safe(True)
    output_data = {'test': []}
    output_row = {'test': []}
    output_col = {'test': []}
    target_data = {'test': []}
    target_row = {'test': []}
    target_col = {'test': []}
    for i in range(len(dataset)):
        each_logger.safe(True)
        output_i, target_i = organization[i].initialize(dataset[i], metric, logger, epoch, each_logger)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.),
                         'ID: {}/{}'.format(i + 1, len(dataset))]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
        for k in dataset[0]:
            output_coo_i_k = output_i[k].tocoo()
            output_data[k].append(output_coo_i_k.data)
            output_row[k].append(output_coo_i_k.row)
            output_col[k].append(output_coo_i_k.col)
            target_coo_i_k = target_i[k].tocoo()
            target_data[k].append(target_coo_i_k.data)
            target_row[k].append(target_coo_i_k.row)
            target_col[k].append(target_coo_i_k.col)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.),
                         'ID: {}/{}'.format(i + 1, len(dataset))]}
        each_logger.append(info, 'test', mean=False)
        print(each_logger.write('test', metric.metric_name['test']))
        each_logger.safe(False)
        each_logger.reset()
    if cfg['data_mode'] == 'user':
        for k in dataset[0]:
            assist.organization_output[0][k] = csr_matrix(
                (np.concatenate(output_data[k]), (np.concatenate(output_row[k]), np.concatenate(output_col[k]))),
                shape=(cfg['num_users']['target'], cfg['num_items']['target']))
            assist.organization_target[0][k] = csr_matrix(
                (np.concatenate(target_data[k]), (np.concatenate(target_row[k]), np.concatenate(target_col[k]))),
                shape=(cfg['num_users']['target'], cfg['num_items']['target']))
    elif cfg['data_mode'] == 'item':
        for k in dataset[0]:
            assist.organization_output[0][k] = csr_matrix(
                (np.concatenate(output_data[k]), (np.concatenate(output_row[k]), np.concatenate(output_col[k]))),
                shape=(cfg['num_items']['target'], cfg['num_users']['target']))
            assist.organization_target[0][k] = csr_matrix(
                (np.concatenate(target_data[k]), (np.concatenate(target_row[k]), np.concatenate(target_col[k]))),
                shape=(cfg['num_items']['target'], cfg['num_users']['target']))
    else:
        raise ValueError('Not valid data mode')
    logger.safe(False)
    logger.reset()
    return


def gather(dataset, organization, epoch):
    with torch.no_grad():
        organization_outputs = [{split: None for split in dataset[i]} for i in range(len(dataset))]
        for i in range(len(dataset)):
            for split in organization_outputs[i]:
                organization_outputs[i][split] = organization[i].predict(dataset[i][split], epoch)
    return organization_outputs


def test(assist, metric, logger, epoch, each_logger):
    logger.safe(True)
    with torch.no_grad():
        organization_output = assist.organization_output[epoch]['test']
        organization_target = assist.organization_target[0]['test']
        batch_size = cfg[cfg['model_name']]['batch_size']['test']
        for i in range(len(assist.data_split)):
            each_logger.safe(True)
            output_i = organization_output[:, assist.data_split[i]]
            target_i = organization_target[:, assist.data_split[i]]
            for j in range(0, output_i.shape[0], batch_size):
                output_i_j = output_i[j:j + batch_size]
                target_i_j = target_i[j:j + batch_size]
                output_i_j_coo = output_i_j.tocoo()
                output_i_j_rating = torch.tensor(output_i_j_coo.data)
                target_i_j_coo = target_i_j.tocoo()
                target_i_j_rating = torch.tensor(target_i_j_coo.data)
                target_mask = ~target_i_j_rating.isnan()
                input_size = target_i_j_rating.size(0)
                if input_size == 0:
                    continue
                if cfg['target_mode'] == 'explicit':
                    output = {'target_rating': torch.tensor(output_i_j.data)}
                    input = {'target_rating': torch.tensor(target_i_j.data)}
                elif cfg['target_mode'] == 'implicit':
                    if cfg['data_mode'] == 'user':
                        target_i_j_user = torch.tensor(target_i_j_coo.row, dtype=torch.long)
                        target_i_j_item = torch.tensor(target_i_j_coo.col, dtype=torch.long)
                        user, user_idx = torch.unique(target_i_j_user, return_inverse=True)
                        item_idx = target_i_j_item
                        num_users = len(user)
                        num_items = len(assist.data_split[i])
                        output_rating = torch.full((num_users, num_items), -float('inf'))
                        target_rating = torch.full((num_users, num_items), 0.)
                        output_rating[user_idx, item_idx] = output_i_j_rating
                        target_rating[user_idx, item_idx] = target_i_j_rating
                        output = {'target_rating': output_rating}
                        input = {'target_rating': target_rating}
                    elif cfg['data_mode'] == 'item':
                        target_i_j_item = torch.tensor(target_i_j_coo.row, dtype=torch.long)
                        target_i_j_user = torch.tensor(target_i_j_coo.col, dtype=torch.long)
                        item, item_idx = torch.unique(target_i_j_item, return_inverse=True)
                        user_idx = target_i_j_user
                        num_items = len(item)
                        num_users = len(assist.data_split[i])
                        output_rating = torch.full((num_items, num_users), -float('inf'))
                        target_rating = torch.full((num_items, num_users), 0.)
                        output_rating[item_idx, user_idx] = output_i_j_rating
                        target_rating[item_idx, user_idx] = target_i_j_rating
                        output = {'target_rating': output_rating}
                        input = {'target_rating': target_rating}
                    else:
                        raise ValueError('Not valid data mode')
                else:
                    raise ValueError('Not valid target mode')
                output['loss'] = models.loss_fn(output_i_j_rating[target_mask], target_i_j_rating[target_mask])
                output = to_device(output, cfg['device'])
                input = to_device(input, cfg['device'])
                input_size = len(input['target_rating'])
                evaluation = metric.evaluate(metric.metric_name['test'], input, output)
                logger.append(evaluation, 'test', n=input_size)
                each_logger.append(evaluation, 'test', n=input_size)
            info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.),
                             'ID: {}/{}'.format(i + 1, len(assist.data_split))]}
            each_logger.append(info, 'test', mean=False)
            print(each_logger.write('test', metric.metric_name['test']))
            each_logger.safe(False)
            each_logger.reset()
        lr = assist.ar_state_dict[epoch]['assist_rate'].item()
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.),
                         'Assist rate: {:.6f}'.format(lr), ]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
        logger.safe(False)
    return


if __name__ == "__main__":
    main()
