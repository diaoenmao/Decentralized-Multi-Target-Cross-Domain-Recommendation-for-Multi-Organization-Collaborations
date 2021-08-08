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
from scipy.sparse import csr_matrix
from config import cfg, process_args
from data import fetch_dataset, make_data_loader, input_collate, make_labeled_dataset, FullDataset
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
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['model_name'], cfg['control_name']]
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
    dataset['train'] = make_labeled_dataset(dataset['train'])
    data_loader = make_data_loader(dataset, cfg['model_name'])
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    optimizer = make_optimizer(model, cfg['model_name'])
    scheduler = make_scheduler(optimizer, cfg['model_name'])
    if cfg['data_mode'] == 'explicit':
        metric = Metric({'train': ['Loss', 'RMSE'], 'test': ['Loss', 'RMSE']})
    elif cfg['data_mode'] == 'implicit':
        metric = Metric({'train': ['Loss'], 'test': ['Loss', 'HR', 'NDCG'], 'make': ['Confidence', 'Confidence Rate']})
    else:
        raise ValueError('Not valid data mode')
    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'])
        last_epoch = result['epoch']
        if last_epoch > 1:
            model.load_state_dict(result['model_state_dict'])
            optimizer.load_state_dict(result['optimizer_state_dict'])
            scheduler.load_state_dict(result['scheduler_state_dict'])
            logger = result['logger']
        else:
            logger = make_logger('output/runs/train_{}'.format(cfg['model_tag']))
    else:
        last_epoch = 1
        logger = make_logger('output/runs/train_{}'.format(cfg['model_tag']))
    if cfg['world_size'] > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(cfg['world_size'])))
    for epoch in range(last_epoch, cfg[cfg['model_name']]['num_epochs'] + 1):
        full_dataset = make_dataset(dataset['train'], model, metric,  logger, epoch)
        data_loader['train'] = make_data_loader({'train': full_dataset}, cfg['model_name'])['train']
        train(data_loader['train'], model, optimizer, metric, logger, epoch)
        test(data_loader['test'], model, metric, logger, epoch)
        scheduler.step()
        model_state_dict = model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict()
        result = {'cfg': cfg, 'epoch': epoch + 1, 'model_state_dict': model_state_dict,
                  'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                  'logger': logger}
        save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    return


def train(data_loader, model, optimizer, metric, logger, epoch):
    logger.safe(True)
    model.train(True)
    start_time = time.time()
    for i, input in enumerate(data_loader):
        input = collate(input)
        input_size = len(input['target'])
        input = to_device(input, cfg['device'])
        optimizer.zero_grad()
        input['aug'] = True
        output = model(input)
        loss = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        evaluation = metric.evaluate(metric.metric_name['train'], input, output)
        logger.append(evaluation, 'train', n=input_size)
        if i % int((len(data_loader) * cfg['log_interval']) + 1) == 0:
            _time = (time.time() - start_time) / (i + 1)
            lr = optimizer.param_groups[0]['lr']
            epoch_finished_time = datetime.timedelta(seconds=round(_time * (len(data_loader) - i - 1)))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg[cfg['model_name']]['num_epochs'] - epoch) * _time * len(data_loader)))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / len(data_loader)),
                             'Learning rate: {:.6f}'.format(lr), 'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
    logger.safe(False)
    return


def test(data_loader, model, metric, logger, epoch):
    logger.safe(True)
    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = len(input['target'])
            input = to_device(input, cfg['device'])
            output = model(input)
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    logger.safe(False)
    return


def make_dataset(dataset, model, metric, logger, epoch):
    logger.safe(True)
    with torch.no_grad():
        semi_dataset = copy.deepcopy(dataset)
        semi_dataset.transform = None
        data_loader = make_data_loader({'train': semi_dataset}, cfg['model_name'], shuffle={'train': False})['train']
        model.train(False)
        user = []
        item = []
        target = []
        chunk_size = 500
        num_chunks = semi_dataset.num_items // chunk_size
        num_unknown = 0
        num_confident = 0
        for i, input in enumerate(data_loader):
            user_i = []
            item_i = []
            target_i = []
            for j in range(len(input['user'])):
                positive_item = input['item'][j]
                positive_target = input['target'][j]
                negative_item = torch.tensor(list(set(range(semi_dataset.num_items)) - set(positive_item.tolist())))
                item_j = negative_item
                item_j_chunks = torch.chunk(item_j, num_chunks)
                output_j = []
                for k in range(num_chunks):
                    item_j_k = item_j_chunks[k]
                    _input = {'user': input['user'][j][0].expand_as(item_j_k), 'item': item_j_k}
                    _input = to_device(_input, cfg['device'])
                    _input['aug'] = True
                    _output = model(_input)
                    output_j_k = _output['target']
                    output_j.append(output_j_k.cpu())
                output_j = torch.cat(output_j, dim=0)
                p_1 = torch.sigmoid(output_j)
                p_0 = 1 - p_1
                soft_pseudo_label = torch.stack([p_0, p_1], dim=-1)
                max_p, output_j = torch.max(soft_pseudo_label, dim=-1)
                # output_j = output_j.float()
                output_j = output_j.float().fill_(cfg['threshold'])
                # mask = max_p.ge(cfg['threshold'])
                mask = p_1.ge(cfg['threshold'])
                # max_p_1, mask = torch.max(p_1, dim=0)
                # print(max_p_1, mask)
                num_unknown += mask.size(0)
                num_confident += mask.float().sum()
                if not torch.all(~mask):
                    item_j = torch.cat([positive_item, item_j[mask]], dim=0)
                    output_j = torch.cat([positive_target, output_j[mask]], dim=0)
                    user_j = input['user'][j][0].expand_as(item_j)
                else:
                    item_j = positive_item
                    output_j = positive_target
                    user_j = input['user'][j][0].expand_as(item_j)
                user_i.append(user_j)
                item_i.append(item_j)
                target_i.append(output_j)
            user_i = torch.cat(user_i, dim=0)
            item_i = torch.cat(item_i, dim=0)
            target_i = torch.cat(target_i, dim=0)
            user.append(user_i.numpy())
            item.append(item_i.numpy())
            target.append(target_i.numpy())
        input = {'num_confident': num_confident, 'num_unknown': num_unknown}
        evaluation = metric.evaluate(metric.metric_name['make'], input, None)
        logger.append(evaluation, 'make', n=1)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Make Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'make', mean=False)
        print(logger.write('make', metric.metric_name['make']))
        user = np.concatenate(user, axis=0)
        item = np.concatenate(item, axis=0)
        target = np.concatenate(target, axis=0)
        semi_dataset.data = csr_matrix((target, (user, item)),
                                       shape=(semi_dataset.num_users, semi_dataset.num_items))
        full_dataset = FullDataset(dataset, semi_dataset)
    logger.safe(False)
    return full_dataset

if __name__ == "__main__":
    main()
