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
    data = args['data'].split('_')
    gpu_ids = [','.join(str(i) for i in list(range(x, x + world_size))) for x in list(range(0, num_gpus, world_size))]
    init_seeds = [list(range(init_seed, init_seed + num_experiments, experiment_step))]
    world_size = [[world_size]]
    num_experiments = [[experiment_step]]
    resume_mode = [[resume_mode]]
    filename = '{}_{}'.format(run, file)
    if file == 'joint':
        controls = []
        script_name = [['{}_recsys_joint.py'.format(run)]]
        control_name = [[data, ['explicit', 'implicit'], ['base'], ['0']]]
        base_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
        controls.extend(base_controls)
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['explicit', 'implicit'], ['mf', 'gmf', 'mlp', 'nmf', 'ae'],
                             ['0', '1']]]
            ml100k_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                            control_name)
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['explicit', 'implicit'], ['mf', 'gmf', 'mlp', 'nmf', 'ae'],
                             ['0', '1']]]
            ml1m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                          control_name)
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['explicit', 'implicit'], ['mf', 'gmf', 'mlp', 'nmf', 'ae'],
                             ['0', '1']]]
            ml10m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                           control_name)
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['explicit', 'implicit'], ['mf', 'gmf', 'mlp', 'nmf', 'ae'],
                             ['0', '1']]]
            ml20m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                           control_name)
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['ML20M'], ['explicit', 'implicit'], ['mf', 'gmf', 'mlp', 'nmf', 'ae'],
                             ['0', '1']]]
            nfp_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                         control_name)
            controls.extend(nfp_controls)
    elif file == 'alone':
        controls = []
        script_name = [['{}_recsys_alone.py'.format(run)]]
        control_name = [[data, ['explicit', 'implicit'], ['base'], ['0'], ['genre', 'random-8'], ['alone']]]
        base_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
        controls.extend(base_controls)
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['explicit', 'implicit'], ['mf', 'gmf', 'mlp', 'nmf', 'ae'],
                             ['0', '1'], ['genre', 'random-8'], ['alone']]]
            ml100k_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                            control_name)
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['explicit', 'implicit'], ['mf', 'gmf', 'mlp', 'nmf', 'ae'],
                             ['0', '1'], ['genre', 'random-8'], ['alone']]]
            ml1m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                          control_name)
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['explicit', 'implicit'], ['mf', 'gmf', 'mlp', 'nmf', 'ae'],
                             ['0', '1'], ['genre', 'random-8'], ['alone']]]
            ml10m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                           control_name)
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['explicit', 'implicit'], ['mf', 'gmf', 'mlp', 'nmf', 'ae'],
                             ['0', '1'], ['genre', 'random-8'], ['alone']]]
            ml20m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                           control_name)
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['explicit', 'implicit'], ['mf', 'gmf', 'mlp', 'nmf', 'ae'],
                             ['0', '1'], ['genre', 'random-8'], ['alone']]]
            nfp_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                         control_name)
            controls.extend(nfp_controls)
    elif file == 'assist':
        controls = []
        script_name = [['{}_recsys_assist.py'.format(run)]]
        if 'ML100K' in data:
            control_name = [
                [['ML100K'], ['explicit', 'implicit'], ['ae'], ['0', '1'], ['genre', 'random-8'], ['assist']]]
            ml100k_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                            control_name)
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['explicit', 'implicit'], ['ae'], ['0', '1'], ['genre', 'random-8'], ['assist']]]
            ml1m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                          control_name)
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [
                [['ML10M'], ['explicit', 'implicit'], ['ae'], ['0', '1'], ['genre', 'random-8'], ['assist']]]
            ml10m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                           control_name)
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [
                [['ML20M'], ['explicit', 'implicit'], ['ae'], ['0', '1'], ['genre', 'random-8'], ['assist']]]
            ml20m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                           control_name)
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['explicit', 'implicit'], ['ae'], ['0', '1'], ['genre', 'random-8'], ['assist']]]
            nfp_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                         control_name)
            controls.extend(nfp_controls)
    else:
        raise ValueError('Not valid file')
    s = '#!/bin/bash\n'
    k = 0
    for i in range(len(controls)):
        controls[i] = list(controls[i])
        s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --init_seed {} --world_size {} --num_experiments {} ' \
                '--resume_mode {} --control_name {}&\n'.format(gpu_ids[k % len(gpu_ids)], *controls[i])
        if k % round == round - 1:
            s = s[:-2] + '\nwait\n'
        k = k + 1
    if s[-5:-1] != 'wait':
        s = s + 'wait\n'
    print(s)
    run_file = open('./{}.sh'.format(filename), 'w')
    run_file.write(s)
    run_file.close()
    return


if __name__ == '__main__':
    main()
