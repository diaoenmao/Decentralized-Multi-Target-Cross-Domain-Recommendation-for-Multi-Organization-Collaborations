import argparse
import itertools

parser = argparse.ArgumentParser(description='config')
parser.add_argument('--run', default='train', type=str)
parser.add_argument('--num_gpus', default=4, type=int)
parser.add_argument('--world_size', default=1, type=int)
parser.add_argument('--init_seed', default=0, type=int)
parser.add_argument('--round', default=4, type=int)
parser.add_argument('--experiment_step', default=1, type=int)
parser.add_argument('--num_experiments', default=1, type=int)
parser.add_argument('--resume_mode', default=0, type=int)
parser.add_argument('--file', default=None, type=str)
parser.add_argument('--data', default=None, type=str)
parser.add_argument('--split_round', default=65535, type=int)
args = vars(parser.parse_args())


def make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    control_names = [control_names]
    controls = script_name + init_seeds + world_size + num_experiments + resume_mode + control_names
    controls = list(itertools.product(*controls))
    return controls


def main():
    run = args['run']
    num_gpus = args['num_gpus']
    world_size = args['world_size']
    round = args['round']
    experiment_step = args['experiment_step']
    init_seed = args['init_seed']
    num_experiments = args['num_experiments']
    resume_mode = args['resume_mode']
    file = args['file']
    split_round = args['split_round']
    data = args['data'].split('_')
    gpu_ids = [','.join(str(i) for i in list(range(x, x + world_size))) for x in list(range(0, num_gpus, world_size))]
    init_seeds = [list(range(init_seed, init_seed + num_experiments, experiment_step))]
    world_size = [[world_size]]
    num_experiments = [[experiment_step]]
    resume_mode = [[resume_mode]]
    filename = '{}_{}_{}'.format(run, file, args['data'])
    if file == 'joint':
        controls = []
        script_name = [['{}_recsys_joint.py'.format(run)]]
        control_name = [[data, ['user', 'item'], ['explicit', 'implicit'], ['base'], ['0']]]
        base_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
        controls.extend(base_controls)
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user', 'item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0']]]
            ml100k_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                            control_name)
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user', 'item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0']]]
            ml1m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                          control_name)
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user', 'item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0']]]
            ml10m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                           control_name)
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user', 'item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0']]]
            ml20m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                           control_name)
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['user', 'item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0']]]
            nfp_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                         control_name)
            controls.extend(nfp_controls)
        if 'Amazon' in data:
            control_name = [[['Amazon'], ['user'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0']]]
            amazon_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                            control_name)
            controls.extend(amazon_controls)
    elif file == 'alone':
        controls = []
        script_name = [['{}_recsys_alone.py'.format(run)]]
        control_name = [[data, ['user'], ['explicit', 'implicit'], ['base'], ['0'], ['genre']]]
        base_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                           control_name)
        control_name = [[data, ['item'], ['explicit', 'implicit'], ['base'], ['0'], ['random-8']]]
        base_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                           control_name)
        base_controls = base_user_controls + base_item_controls
        controls.extend(base_controls)
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0'], ['genre']]]
            ml100k_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                 control_name)
            control_name = [[['ML100K'], ['item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0'], ['random-8']]]
            ml100k_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                 control_name)
            ml100k_controls = ml100k_user_controls + ml100k_item_controls
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0'], ['genre']]]
            ml1m_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                               control_name)
            control_name = [[['ML1M'], ['item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0'], ['random-8']]]
            ml1m_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                               control_name)
            ml1m_controls = ml1m_user_controls + ml1m_item_controls
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0'], ['genre']]]
            ml10m_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            ml10m_controls = ml10m_user_controls
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0'], ['genre']]]
            ml20m_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            control_name = [[['ML20M'], ['item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0'], ['random-8']]]
            ml20m_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            ml20m_controls = ml20m_user_controls + ml20m_item_controls
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['user', 'item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0'], ['random-8']]]
            nfp_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                         control_name)
            controls.extend(nfp_controls)
        if 'Amazon' in data:
            control_name = [[['Amazon'], ['user'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0'], ['genre']]]
            amazon_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                 control_name)
            controls.extend(amazon_controls)
    elif file == 'assist':
        controls = []
        script_name = [['{}_recsys_assist.py'.format(run)]]
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['constant']]]
            ml100k_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                 control_name)
            control_name = [[['ML100K'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant']]]
            ml100k_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                 control_name)
            ml100k_controls = ml100k_user_controls + ml100k_item_controls
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['constant']]]
            ml1m_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                               control_name)
            control_name = [[['ML1M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant']]]
            ml1m_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                               control_name)
            ml1m_controls = ml1m_user_controls + ml1m_item_controls
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['constant']]]
            ml10m_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            control_name = [[['ML10M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant']]]
            ml10m_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            ml10m_controls = ml10m_user_controls + ml10m_item_controls
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['constant']]]
            ml20m_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            control_name = [[['ML20M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant']]]
            ml20m_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            ml20m_controls = ml20m_user_controls + ml20m_item_controls
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['user', 'item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant']]]
            nfp_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                         control_name)
            controls.extend(nfp_controls)
        if 'Amazon' in data:
            control_name = [[['Amazon'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'], ['genre'], ['constant-0.1'],
                             ['constant']]]
            amazon_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                 control_name)
            controls.extend(amazon_controls)
    elif file == 'info':
        controls = []
        script_name = [['{}_recsys_assist.py'.format(run)]]
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['ae'], ['1'],
                             ['genre'], ['constant-0.1'], ['constant']]]
            ml100k_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                 control_name)
            ml100k_controls = ml100k_user_controls
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['ae'], ['1'],
                             ['genre'], ['constant-0.1'], ['constant']]]
            ml1m_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                               control_name)
            ml1m_controls = ml1m_user_controls
            controls.extend(ml1m_controls)
    elif file == 'ar':
        controls = []
        script_name = [['{}_recsys_assist.py'.format(run)]]
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.3'], ['constant']]]
            ml100k_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                 control_name)
            control_name = [[['ML100K'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.3'], ['constant']]]
            ml100k_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                 control_name)
            ml100k_controls = ml100k_user_controls + ml100k_item_controls
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.3'], ['constant']]]
            ml1m_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                               control_name)
            control_name = [[['ML1M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.3'], ['constant']]]
            ml1m_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                               control_name)
            ml1m_controls = ml1m_user_controls + ml1m_item_controls
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.3'], ['constant']]]
            ml10m_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            control_name = [[['ML10M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.3'], ['constant']]]
            ml10m_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            ml10m_controls = ml10m_user_controls + ml10m_item_controls
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.3'], ['constant']]]
            ml20m_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            control_name = [[['ML20M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.3'], ['constant']]]
            ml20m_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            ml20m_controls = ml20m_user_controls + ml20m_item_controls
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['user', 'item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.3'], ['constant']]]
            nfp_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                         control_name)
            controls.extend(nfp_controls)
        if 'Amazon' in data:
            control_name = [[['Amazon'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'], ['genre'], ['constant-0.3'],
                             ['constant']]]
            amazon_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                 control_name)
            controls.extend(amazon_controls)
    elif file == 'aw':
        controls = []
        script_name = [['{}_recsys_assist.py'.format(run)]]
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['optim']]]
            ml100k_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                 control_name)
            control_name = [[['ML100K'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['optim']]]
            ml100k_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                 control_name)
            ml100k_controls = ml100k_user_controls + ml100k_item_controls
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['optim']]]
            ml1m_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                               control_name)
            control_name = [[['ML1M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['optim']]]
            ml1m_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                               control_name)
            ml1m_controls = ml1m_user_controls + ml1m_item_controls
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['optim']]]
            ml10m_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            control_name = [[['ML10M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['optim']]]
            ml10m_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            ml10m_controls = ml10m_user_controls + ml10m_item_controls
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['optim']]]
            ml20m_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            control_name = [[['ML20M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['optim']]]
            ml20m_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            ml20m_controls = ml20m_user_controls + ml20m_item_controls
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['user', 'item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['optim']]]
            nfp_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                         control_name)
            controls.extend(nfp_controls)
        if 'Amazon' in data:
            control_name = [[['Amazon'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'], ['genre'], ['constant-0.1'],
                             ['optim']]]
            amazon_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                 control_name)
            controls.extend(amazon_controls)
    elif file == 'ar-optim':
        controls = []
        script_name = [['{}_recsys_assist.py'.format(run)]]
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['optim-0.1'], ['constant']]]
            ml100k_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                 control_name)
            control_name = [[['ML100K'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['optim-0.1'], ['constant']]]
            ml100k_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                 control_name)
            ml100k_controls = ml100k_user_controls + ml100k_item_controls
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['optim-0.1'], ['constant']]]
            ml1m_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                               control_name)
            control_name = [[['ML1M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['optim-0.1'], ['constant']]]
            ml1m_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                               control_name)
            ml1m_controls = ml1m_user_controls + ml1m_item_controls
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['optim-0.1'], ['constant']]]
            ml10m_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            control_name = [[['ML10M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['optim-0.1'], ['constant']]]
            ml10m_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            ml10m_controls = ml10m_user_controls + ml10m_item_controls
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['optim-0.1'], ['constant']]]
            ml20m_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            control_name = [[['ML20M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['optim-0.1'], ['constant']]]
            ml20m_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            ml20m_controls = ml20m_user_controls + ml20m_item_controls
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['user', 'item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['optim-0.1'], ['constant']]]
            nfp_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                         control_name)
            controls.extend(nfp_controls)
        if 'Amazon' in data:
            control_name = [[['Amazon'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'], ['genre'], ['optim-0.1'],
                             ['constant']]]
            amazon_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                 control_name)
            controls.extend(amazon_controls)
    elif file == 'optim-optim':
        controls = []
        script_name = [['{}_recsys_assist.py'.format(run)]]
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['optim-0.1'], ['optim']]]
            ml100k_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                 control_name)
            control_name = [[['ML100K'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['optim-0.1'], ['optim']]]
            ml100k_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                 control_name)
            ml100k_controls = ml100k_user_controls + ml100k_item_controls
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['optim-0.1'], ['optim']]]
            ml1m_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                               control_name)
            control_name = [[['ML1M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['optim-0.1'], ['optim']]]
            ml1m_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                               control_name)
            ml1m_controls = ml1m_user_controls + ml1m_item_controls
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['optim-0.1'], ['optim']]]
            ml10m_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            control_name = [[['ML10M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['optim-0.1'], ['optim']]]
            ml10m_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            ml10m_controls = ml10m_user_controls + ml10m_item_controls
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['optim-0.1'], ['optim']]]
            ml20m_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            control_name = [[['ML20M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['optim-0.1'], ['optim']]]
            ml20m_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            ml20m_controls = ml20m_user_controls + ml20m_item_controls
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['user', 'item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['optim-0.1'], ['optim']]]
            nfp_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                         control_name)
            controls.extend(nfp_controls)
        if 'Amazon' in data:
            control_name = [[['Amazon'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'], ['genre'], ['optim-0.1'],
                             ['optim']]]
            amazon_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                 control_name)
            controls.extend(amazon_controls)
    elif file == 'match':
        controls = []
        script_name = [['{}_recsys_assist.py'.format(run)]]
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['constant'], ['0.5']]]
            ml100k_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                 control_name)
            control_name = [[['ML100K'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant'], ['0.5']]]
            ml100k_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                 control_name)
            ml100k_controls = ml100k_user_controls + ml100k_item_controls
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['constant'], ['0.5']]]
            ml1m_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                               control_name)
            control_name = [[['ML1M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant'], ['0.5']]]
            ml1m_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                               control_name)
            ml1m_controls = ml1m_user_controls + ml1m_item_controls
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['constant'], ['0.5']]]
            ml10m_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            control_name = [[['ML10M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant'], ['0.5']]]
            ml10m_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            ml10m_controls = ml10m_user_controls + ml10m_item_controls
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['constant'], ['0.5']]]
            ml20m_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            control_name = [[['ML20M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant'], ['0.5']]]
            ml20m_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            ml20m_controls = ml20m_user_controls + ml20m_item_controls
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['user', 'item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant'], ['0.5']]]
            nfp_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                         control_name)
            controls.extend(nfp_controls)
        if 'Amazon' in data:
            control_name = [[['Amazon'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'], ['genre'], ['constant-0.1'],
                             ['constant'], ['0.5']]]
            amazon_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                 control_name)
            controls.extend(amazon_controls)
    elif file == 'pl':
        controls = []
        script_name = [['{}_recsys_assist.py'.format(run)]]
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['constant'], ['1'], ['dp-10', 'ip-10']]]
            ml100k_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                 control_name)
            control_name = [[['ML100K'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant'], ['1'], ['dp-10', 'ip-10']]]
            ml100k_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                 control_name)
            ml100k_controls = ml100k_user_controls + ml100k_item_controls
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['constant'], ['1'], ['dp-10', 'ip-10']]]
            ml1m_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                               control_name)
            control_name = [[['ML1M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant'], ['1'], ['dp-10', 'ip-10']]]
            ml1m_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                               control_name)
            ml1m_controls = ml1m_user_controls + ml1m_item_controls
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['constant'], ['1'], ['dp-10', 'ip-10']]]
            ml10m_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            control_name = [[['ML10M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant'], ['1'], ['dp-10', 'ip-10']]]
            ml10m_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            ml10m_controls = ml10m_user_controls + ml10m_item_controls
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['constant'], ['1'], ['dp-10', 'ip-10']]]
            ml20m_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            control_name = [[['ML20M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant'], ['1'], ['dp-10', 'ip-10']]]
            ml20m_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                control_name)
            ml20m_controls = ml20m_user_controls + ml20m_item_controls
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['user', 'item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant'], ['1'], ['dp-10', 'ip-10']]]
            nfp_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                         control_name)
            controls.extend(nfp_controls)
        if 'Amazon' in data:
            control_name = [[['Amazon'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'], ['genre'], ['constant-0.1'],
                             ['constant'], ['1'], ['dp-10', 'ip-10']]]
            amazon_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                                 control_name)
            controls.extend(amazon_controls)
    else:
        raise ValueError('Not valid file')
    s = '#!/bin/bash\n'
    j = 1
    k = 1
    for i in range(len(controls)):
        controls[i] = list(controls[i])
        s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --init_seed {} --world_size {} ' \
                '--num_experiments {} --resume_mode {} --control_name {}&\n'.format(
            gpu_ids[i % len(gpu_ids)], *controls[i])
        if i % round == round - 1:
            s = s[:-2] + '\nwait\n'
            if j % split_round == 0:
                print(s)
                run_file = open('./{}_{}.sh'.format(filename, k), 'w')
                run_file.write(s)
                run_file.close()
                s = '#!/bin/bash\n'
                k = k + 1
            j = j + 1
    if s != '#!/bin/bash\n':
        if s[-5:-1] != 'wait':
            s = s + 'wait\n'
        print(s)
        run_file = open('./{}_{}.sh'.format(filename, k), 'w')
        run_file.write(s)
        run_file.close()
    return


if __name__ == '__main__':
    main()
