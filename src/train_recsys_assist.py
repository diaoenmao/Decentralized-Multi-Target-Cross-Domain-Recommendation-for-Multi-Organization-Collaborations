import argparse
import copy
import datetime
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from data import fetch_dataset, make_data_loader, split_dataset
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import make_logger

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
    assist = Assist(data_split)
    organization = assist.make_organization()
    if cfg['data_mode'] == 'explicit':
        metric = Metric({'train': ['Loss', 'RMSE'], 'test': ['Loss', 'RMSE']})
    elif cfg['data_mode'] == 'implicit':
        metric = Metric({'train': ['Loss'], 'test': ['Loss', 'HR', 'NDCG']})
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
    for epoch in range(last_epoch, cfg[cfg['global']]['num_epochs'] + 1):
        dataset = assist.make_dataset(dataset, epoch)
        train(dataset, organization, metric, logger, epoch)
        organization_outputs = gather(organization, epoch)
        assist.update(organization_outputs, epoch)
        test(assist, metric, logger, epoch)
        result = {'cfg': cfg, 'epoch': epoch + 1, 'assist': assist, 'organization': organization, 'logger': logger}
        save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    return


def initialize(dataset, assist, organization, metric, logger, epoch):
    logger.safe(True)
    initialization = organization[0].initialize(dataset, metric, logger)
    info = {'info': ['Model: {}'.format(cfg['model_tag']),
                     'Train Epoch: {}'.format(epoch), 'ID: 1']}
    logger.append(info, 'train', mean=False)
    print(logger.write('train', metric.metric_name['train']))
    info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
    logger.append(info, 'test', mean=False)
    print(logger.write('test', metric.metric_name['test']))
    assist.organization_output[0]['train'] = initialization['train']
    assist.organization_target[0]['train'] = torch.tensor(dataset['train'].train_data.data)
    assist.organization_output[0]['test'] = initialization['test']
    assist.organization_target[0]['test'] = torch.tensor(dataset['test'].test_data.data)
    logger.safe(False)
    logger.reset()
    return


def train(dataset, organization, metric, logger, epoch):
    logger.safe(True)
    start_time = time.time()
    num_organizations = len(organization)
    for i in range(num_organizations):
        data_loader = make_data_loader(dataset, organization[i].model_name[iter])
        organization[i].train(epoch, data_loader['train'], metric, logger)
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
            print(logger.write('train', metric.metric_name['train']))
    logger.safe(False)
    return


def gather(organization, epoch):
    with torch.no_grad():
        num_organizations = len(organization)
        organization_outputs = [{'train': None, 'test': None} for _ in range(num_organizations)]
        for i in range(num_organizations):
            for split in organization_outputs[i]:
                organization_outputs[i][split] = organization[i].predict(epoch, data_loader[i][split])['target_rating']
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
