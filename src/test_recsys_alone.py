import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import models
from config import cfg, process_args
from data import fetch_dataset, make_data_loader, split_dataset, make_split_dataset
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, resume, collate
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
    if cfg['data_mode'] == 'explicit':
        metric = Metric({'train': ['Loss', 'RMSE'], 'test': ['Loss', 'RMSE']})
    elif cfg['data_mode'] == 'implicit':
        metric = Metric({'train': ['Loss', 'MAP'], 'test': ['Loss', 'MAP']})
    else:
        raise ValueError('Not valid data mode')
    result = resume(cfg['model_tag'], load_tag='best')
    last_epoch = result['epoch']
    data_split = result['data_split']
    dataset = make_split_dataset(data_split)
    data_loader = {'train': [], 'test': []}
    model = []
    for i in range(len(dataset)):
        data_loader_i = make_data_loader(dataset[i], cfg['model_name'])
        num_users = dataset[i]["train"].num_users['data']
        num_items = dataset[i]["train"].num_items['data']
        if cfg['model_name'] == 'ae':
            model_i = eval(
                'models.{}(num_items, num_items).to(cfg["device"])'.format(cfg['model_name']))
        else:
            model_i = eval(
                'models.{}(num_users, num_items).to(cfg["device"])'.format(cfg['model_name']))
        data_loader['train'].append(data_loader_i['train'])
        data_loader['test'].append(data_loader_i['test'])
        model.append(model_i)
    for i in range(len(dataset)):
        model[i].load_state_dict(result['model_state_dict'][i])
    test_logger = make_logger('output/runs/test_{}'.format(cfg['model_tag']))
    test_each_logger = make_logger('output/runs/test_{}'.format(cfg['model_tag']))
    test(data_loader['test'], model, metric, test_logger, last_epoch, test_each_logger)
    result = resume(cfg['model_tag'], load_tag='checkpoint')
    train_logger = result['logger'] if 'logger' in result else None
    result = {'cfg': cfg, 'epoch': last_epoch,
              'logger': {'train': train_logger, 'test': test_logger, 'test_each': test_each_logger}}
    save(result, './output/result/{}.pt'.format(cfg['model_tag']))
    return


def test(data_loader, model, metric, logger, epoch, each_logger):
    logger.safe(True)
    with torch.no_grad():
        for m in range(len(data_loader)):
            each_logger.safe(True)
            model[m].train(False)
            for i, input in enumerate(data_loader[m]):
                input = collate(input)
                input_size = len(input['target_user'])
                if input_size == 0:
                    continue
                input = to_device(input, cfg['device'])
                output = model[m](input)
                output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
                evaluation = metric.evaluate(metric.metric_name['test'], input, output)
                logger.append(evaluation, 'test', input_size)
                each_logger.append(evaluation, 'test', input_size)
            info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.),
                             'ID: {}/{}'.format(m + 1, len(data_loader))]}
            each_logger.append(info, 'test', mean=False)
            print(each_logger.write('test', metric.metric_name['test']))
            each_logger.safe(False)
            each_logger.reset()
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    logger.safe(False)
    return

if __name__ == "__main__":
    main()
