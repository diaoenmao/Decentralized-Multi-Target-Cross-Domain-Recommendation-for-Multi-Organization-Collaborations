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
    data_split = split_dataset(dataset)
    dataset = make_split_dataset(data_split)
    assist = Assist(data_split)
    organization = assist.make_organization()
    if cfg['data_mode'] == 'explicit':
        metric = Metric({'train': ['Loss', 'RMSE'], 'test': ['Loss', 'RMSE']})
    elif cfg['data_mode'] == 'implicit':
        metric = Metric({'train': ['Loss', 'MAP'], 'test': ['Loss', 'MAP']})
    else:
        raise ValueError('Not valid data mode')
    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'])
        last_epoch = result['epoch']
        if last_epoch > 1:
            assist = result['assist']
            organization = result['organization']
            logger = result['logger']
        else:
            logger = make_logger('output/runs/train_{}'.format(cfg['model_tag']))
    else:
        last_epoch = 1
        logger = make_logger('output/runs/train_{}'.format(cfg['model_tag']))
    if last_epoch == 1:
        initialize(dataset, assist, organization, metric, logger, 0)
    for epoch in range(last_epoch, cfg['global']['num_epochs'] + 1):
        dataset = assist.make_dataset(dataset, epoch)
        train(dataset, organization, metric, logger, epoch)
        organization_outputs = gather(dataset, organization, epoch)
        assist.update(organization_outputs, epoch)
        exit()
        test(assist, metric, logger, epoch)
        result = {'cfg': cfg, 'epoch': epoch + 1, 'assist': assist, 'organization': organization, 'logger': logger}
        save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    return


# def initialize(dataset, assist, organization, metric, logger, epoch):
#     logger.safe(True)
#     output = {'train': 0, 'test': 0}
#     target = {'train': 0, 'test': 0}
#     count = {'train': 0, 'test': 0}
#     for i in range(len(dataset)):
#         output_i, target_i, count_i = organization[i].initialize(dataset[i], metric, logger, epoch)
#         info = {'info': ['Model: {}'.format(cfg['model_tag']),
#                          'Train Epoch: {}'.format(epoch), 'ID: {}'.format(i + 1)]}
#         logger.append(info, 'train', mean=False)
#         print(logger.write('train', metric.metric_name['train']))
#         info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
#         logger.append(info, 'test', mean=False)
#         print(logger.write('test', metric.metric_name['test']))
#         for k in output:
#             output[k] = output[k] + output_i[k]
#             target[k] = target[k] + target_i[k]
#             count[k] = count[k] + count_i[k]
#     for k in output:
#         coo = count[k].tocoo()
#         row, col = coo.row, coo.col
#         assist.organization_output[0][k] = csr_matrix((output[k].data / count[k].data, (row, col)),
#                                                       shape=(cfg['num_users'], cfg['num_items']))
#         assist.organization_target[0][k] = csr_matrix((target[k].data / count[k].data, (row, col)),
#                                                       shape=(cfg['num_users'], cfg['num_items']))
#     logger.safe(False)
#     logger.reset()
#     return

def initialize(dataset, assist, organization, metric, logger, epoch):
    logger.safe(True)
    output_data = {'train': [], 'test': []}
    output_row = {'train': [], 'test': []}
    output_col = {'train': [], 'test': []}
    target_data = {'train': [], 'test': []}
    target_row = {'train': [], 'test': []}
    target_col = {'train': [], 'test': []}
    for i in range(len(dataset)):
        output_i, target_i = organization[i].initialize(dataset[i], metric, logger, epoch)
        info = {'info': ['Model: {}'.format(cfg['model_tag']),
                         'Train Epoch: {}'.format(epoch), 'ID: {}'.format(i + 1)]}
        logger.append(info, 'train', mean=False)
        print(logger.write('train', metric.metric_name['train']))
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
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
    for k in dataset[0]:
        assist.organization_output[0][k] = csr_matrix(
            (np.concatenate(output_data[k]), (np.concatenate(output_row[k]), np.concatenate(output_col[k]))),
            shape=(cfg['num_users'], cfg['num_items']))
        assist.organization_target[0][k] = csr_matrix(
            (np.concatenate(target_data[k]), (np.concatenate(target_row[k]), np.concatenate(target_col[k]))),
            shape=(cfg['num_users'], cfg['num_items']))
    logger.safe(False)
    logger.reset()
    return


def train(dataset, organization, metric, logger, epoch):
    logger.safe(True)
    start_time = time.time()
    num_organizations = len(organization)
    for i in range(num_organizations):
        organization[i].train(dataset[i]['train'], metric, logger, epoch)
        if i % int((num_organizations * cfg['log_interval']) + 1) == 0:
            local_time = (time.time() - start_time) / (i + 1)
            epoch_finished_time = datetime.timedelta(seconds=local_time * (num_organizations - i - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['global']['num_epochs'] - epoch) * local_time * num_organizations))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / num_organizations),
                             'ID: {}/{}'.format(i + 1, num_organizations),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', [metric.metric_name['train'][0]]))
    logger.safe(False)
    return


def gather(dataset, organization, epoch):
    with torch.no_grad():
        organization_outputs = [{split: None for split in dataset[i]} for i in range(len(dataset))]
        for i in range(len(dataset)):
            for split in organization_outputs[i]:
                organization_outputs[i][split] = organization[i].predict(dataset[i][split], epoch)
    return organization_outputs


def test(assist, metric, logger, epoch):
    logger.safe(True)
    with torch.no_grad():
        input_size = assist.organization_target[0]['test'].size(0)
        input = {'target_rating': assist.organization_target[0]['test']}
        output = {'target_rating': assist.organization_output[epoch]['test']}
        output['loss'] = models.loss_fn(output['target_rating'], input['target_rating'])
        evaluation = metric.evaluate(metric.metric_name['test'], input, output)
        logger.append(evaluation, 'test', n=input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    logger.safe(False)
    return


if __name__ == "__main__":
    main()
