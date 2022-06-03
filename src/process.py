import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import itertools
import json
import numpy as np
import pandas as pd
from utils import save, load, makedir_exist_ok
import matplotlib.pyplot as plt
from collections import defaultdict

result_path = './output/result'
save_format = 'png'
vis_path = './output/vis/{}'.format(save_format)
num_experiments = 4
exp = [str(x) for x in list(range(num_experiments))]
dpi = 300

def make_controls(control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    controls = [exp] + [control_names]
    controls = list(itertools.product(*controls))
    return controls


def make_control_list(data, file):
    if file == 'joint':
        controls = []
        control_name = [[data, ['user', 'item'], ['explicit', 'implicit'], ['base'], ['0']]]
        base_controls = make_controls(control_name)
        controls.extend(base_controls)
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user', 'item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0']]]
            ml100k_controls = make_controls(control_name)
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user', 'item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0']]]
            ml1m_controls = make_controls(control_name)
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user', 'item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0']]]
            ml10m_controls = make_controls(control_name)
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user', 'item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0']]]
            ml20m_controls = make_controls(control_name)
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['user', 'item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0']]]
            nfp_controls = make_controls(control_name)
            controls.extend(nfp_controls)
        if 'Douban' in data:
            control_name = [[['Douban'], ['user'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0']]]
            douban_controls = make_controls(control_name)
            controls.extend(douban_controls)
        if 'Amazon' in data:
            control_name = [[['Amazon'], ['user'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0']]]
            amazon_controls = make_controls(control_name)
            controls.extend(amazon_controls)
    elif file == 'alone':
        controls = []
        control_name = [[data, ['user'], ['explicit', 'implicit'], ['base'], ['0'], ['genre']]]
        base_user_controls = make_controls(control_name)
        control_name = [[data, ['item'], ['explicit', 'implicit'], ['base'], ['0'], ['random-8']]]
        base_item_controls = make_controls(control_name)
        base_controls = base_user_controls + base_item_controls
        controls.extend(base_controls)
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0'], ['genre']]]
            ml100k_user_controls = make_controls(control_name)
            control_name = [[['ML100K'], ['item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0'], ['random-8']]]
            ml100k_item_controls = make_controls(control_name)
            ml100k_controls = ml100k_user_controls + ml100k_item_controls
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0'], ['genre']]]
            ml1m_user_controls = make_controls(control_name)
            control_name = [[['ML1M'], ['item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0'], ['random-8']]]
            ml1m_item_controls = make_controls(control_name)
            ml1m_controls = ml1m_user_controls + ml1m_item_controls
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0'], ['genre']]]
            ml10m_user_controls = make_controls(control_name)
            control_name = [[['ML10M'], ['item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0'], ['random-8']]]
            ml10m_item_controls = make_controls(control_name)
            ml10m_controls = ml10m_user_controls + ml10m_item_controls
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0'], ['genre']]]
            ml20m_user_controls = make_controls(control_name)
            control_name = [[['ML20M'], ['item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0'], ['random-8']]]
            ml20m_item_controls = make_controls(control_name)
            ml20m_controls = ml20m_user_controls + ml20m_item_controls
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['user', 'item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0'], ['random-8']]]
            nfp_controls = make_controls(control_name)
            controls.extend(nfp_controls)
        if 'Douban' in data:
            control_name = [[['Douban'], ['user'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0'], ['genre']]]
            douban_controls = make_controls(control_name)
            controls.extend(douban_controls)
        if 'Amazon' in data:
            control_name = [[['Amazon'], ['user'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0'], ['genre']]]
            amazon_controls = make_controls(control_name)
            controls.extend(amazon_controls)
    elif file == 'assist':
        controls = []
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['constant']]]
            ml100k_user_controls = make_controls(control_name)
            control_name = [[['ML100K'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant']]]
            ml100k_item_controls = make_controls(control_name)
            ml100k_controls = ml100k_user_controls + ml100k_item_controls
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['constant']]]
            ml1m_user_controls = make_controls(control_name)
            control_name = [[['ML1M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant']]]
            ml1m_item_controls = make_controls(control_name)
            ml1m_controls = ml1m_user_controls + ml1m_item_controls
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['constant']]]
            ml10m_user_controls = make_controls(control_name)
            control_name = [[['ML10M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant']]]
            ml10m_item_controls = make_controls(control_name)
            ml10m_controls = ml10m_user_controls + ml10m_item_controls
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['constant']]]
            ml20m_user_controls = make_controls(control_name)
            control_name = [[['ML20M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant']]]
            ml20m_item_controls = make_controls(control_name)
            ml20m_controls = ml20m_user_controls + ml20m_item_controls
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['user', 'item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant']]]
            nfp_controls = make_controls(control_name)
            controls.extend(nfp_controls)
        if 'Douban' in data:
            control_name = [[['Douban'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'], ['genre'], ['constant-0.1'],
                             ['constant']]]
            douban_controls = make_controls(control_name)
            controls.extend(douban_controls)
        if 'Amazon' in data:
            control_name = [[['Amazon'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'], ['genre'], ['constant-0.1'],
                             ['constant']]]
            amazon_controls = make_controls(control_name)
            controls.extend(amazon_controls)
    elif file == 'info':
        controls = []
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['ae'], ['1'],
                             ['genre'], ['constant-0.1'], ['constant']]]
            ml100k_user_controls = make_controls(control_name)
            ml100k_controls = ml100k_user_controls
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['ae'], ['1'],
                             ['genre'], ['constant-0.1'], ['constant']]]
            ml1m_user_controls = make_controls(control_name)
            ml1m_controls = ml1m_user_controls
            controls.extend(ml1m_controls)
    elif file == 'ar':
        controls = []
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.3'], ['constant']]]
            ml100k_user_controls = make_controls(control_name)
            control_name = [[['ML100K'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.3'], ['constant']]]
            ml100k_item_controls = make_controls(control_name)
            ml100k_controls = ml100k_user_controls + ml100k_item_controls
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.3'], ['constant']]]
            ml1m_user_controls = make_controls(control_name)
            control_name = [[['ML1M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.3'], ['constant']]]
            ml1m_item_controls = make_controls(control_name)
            ml1m_controls = ml1m_user_controls + ml1m_item_controls
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.3'], ['constant']]]
            ml10m_user_controls = make_controls(control_name)
            control_name = [[['ML10M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.3'], ['constant']]]
            ml10m_item_controls = make_controls(control_name)
            ml10m_controls = ml10m_user_controls + ml10m_item_controls
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.3'], ['constant']]]
            ml20m_user_controls = make_controls(control_name)
            control_name = [[['ML20M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.3'], ['constant']]]
            ml20m_item_controls = make_controls(control_name)
            ml20m_controls = ml20m_user_controls + ml20m_item_controls
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['user', 'item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.3'], ['constant']]]
            nfp_controls = make_controls(control_name)
            controls.extend(nfp_controls)
        if 'Douban' in data:
            control_name = [[['Douban'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'], ['genre'], ['constant-0.3'],
                             ['constant']]]
            douban_controls = make_controls(control_name)
            controls.extend(douban_controls)
        if 'Amazon' in data:
            control_name = [[['Amazon'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'], ['genre'], ['constant-0.3'],
                             ['constant']]]
            amazon_controls = make_controls(control_name)
            controls.extend(amazon_controls)
    elif file == 'aw':
        controls = []
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['optim']]]
            ml100k_user_controls = make_controls(control_name)
            control_name = [[['ML100K'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['optim']]]
            ml100k_item_controls = make_controls(control_name)
            ml100k_controls = ml100k_user_controls + ml100k_item_controls
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['optim']]]
            ml1m_user_controls = make_controls(control_name)
            control_name = [[['ML1M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['optim']]]
            ml1m_item_controls = make_controls(control_name)
            ml1m_controls = ml1m_user_controls + ml1m_item_controls
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['optim']]]
            ml10m_user_controls = make_controls(control_name)
            control_name = [[['ML10M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['optim']]]
            ml10m_item_controls = make_controls(control_name)
            ml10m_controls = ml10m_user_controls + ml10m_item_controls
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['optim']]]
            ml20m_user_controls = make_controls(control_name)
            control_name = [[['ML20M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['optim']]]
            ml20m_item_controls = make_controls(control_name)
            ml20m_controls = ml20m_user_controls + ml20m_item_controls
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['user', 'item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['optim']]]
            nfp_controls = make_controls(control_name)
            controls.extend(nfp_controls)
        if 'Douban' in data:
            control_name = [[['Douban'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'], ['genre'], ['constant-0.1'],
                             ['optim']]]
            douban_controls = make_controls(control_name)
            controls.extend(douban_controls)
        if 'Amazon' in data:
            control_name = [[['Amazon'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'], ['genre'], ['constant-0.1'],
                             ['optim']]]
            amazon_controls = make_controls(control_name)
            controls.extend(amazon_controls)
    elif file == 'ar-optim':
        controls = []
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['optim-0.1'], ['constant']]]
            ml100k_user_controls = make_controls(control_name)
            control_name = [[['ML100K'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['optim-0.1'], ['constant']]]
            ml100k_item_controls = make_controls(control_name)
            ml100k_controls = ml100k_user_controls + ml100k_item_controls
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['optim-0.1'], ['constant']]]
            ml1m_user_controls = make_controls(control_name)
            control_name = [[['ML1M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['optim-0.1'], ['constant']]]
            ml1m_item_controls = make_controls(control_name)
            ml1m_controls = ml1m_user_controls + ml1m_item_controls
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['optim-0.1'], ['constant']]]
            ml10m_user_controls = make_controls(control_name)
            control_name = [[['ML10M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['optim-0.1'], ['constant']]]
            ml10m_item_controls = make_controls(control_name)
            ml10m_controls = ml10m_user_controls + ml10m_item_controls
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['optim-0.1'], ['constant']]]
            ml20m_user_controls = make_controls(control_name)
            control_name = [[['ML20M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['optim-0.1'], ['constant']]]
            ml20m_item_controls = make_controls(control_name)
            ml20m_controls = ml20m_user_controls + ml20m_item_controls
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['user', 'item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['optim-0.1'], ['constant']]]
            nfp_controls = make_controls(control_name)
            controls.extend(nfp_controls)
        if 'Douban' in data:
            control_name = [[['Douban'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'], ['genre'], ['optim-0.1'],
                             ['constant']]]
            douban_controls = make_controls(control_name)
            controls.extend(douban_controls)
        if 'Amazon' in data:
            control_name = [[['Amazon'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'], ['genre'], ['optim-0.1'],
                             ['constant']]]
            amazon_controls = make_controls(control_name)
            controls.extend(amazon_controls)
    elif file == 'optim-optim':
        controls = []
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['optim-0.1'], ['optim']]]
            ml100k_user_controls = make_controls(control_name)
            control_name = [[['ML100K'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['optim-0.1'], ['optim']]]
            ml100k_item_controls = make_controls(control_name)
            ml100k_controls = ml100k_user_controls + ml100k_item_controls
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['optim-0.1'], ['optim']]]
            ml1m_user_controls = make_controls(control_name)
            control_name = [[['ML1M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['optim-0.1'], ['optim']]]
            ml1m_item_controls = make_controls(control_name)
            ml1m_controls = ml1m_user_controls + ml1m_item_controls
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['optim-0.1'], ['optim']]]
            ml10m_user_controls = make_controls(control_name)
            control_name = [[['ML10M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['optim-0.1'], ['optim']]]
            ml10m_item_controls = make_controls(control_name)
            ml10m_controls = ml10m_user_controls + ml10m_item_controls
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['optim-0.1'], ['optim']]]
            ml20m_user_controls = make_controls(control_name)
            control_name = [[['ML20M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['optim-0.1'], ['optim']]]
            ml20m_item_controls = make_controls(control_name)
            ml20m_controls = ml20m_user_controls + ml20m_item_controls
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['user', 'item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['optim-0.1'], ['constant']]]
            nfp_controls = make_controls(control_name)
            controls.extend(nfp_controls)
        if 'Douban' in data:
            control_name = [[['Douban'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'], ['genre'], ['optim-0.1'],
                             ['optim']]]
            douban_controls = make_controls(control_name)
            controls.extend(douban_controls)
        if 'Amazon' in data:
            control_name = [[['Amazon'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'], ['genre'], ['optim-0.1'],
                             ['constant']]]
            amazon_controls = make_controls(control_name)
            controls.extend(amazon_controls)
    elif file == 'match':
        controls = []
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['constant'], ['0.5']]]
            ml100k_user_controls = make_controls(control_name)
            control_name = [[['ML100K'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant'], ['0.5']]]
            ml100k_item_controls = make_controls(control_name)
            ml100k_controls = ml100k_user_controls + ml100k_item_controls
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['constant'], ['0.5']]]
            ml1m_user_controls = make_controls(control_name)
            control_name = [[['ML1M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant'], ['0.5']]]
            ml1m_item_controls = make_controls(control_name)
            ml1m_controls = ml1m_user_controls + ml1m_item_controls
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['constant'], ['0.5']]]
            ml10m_user_controls = make_controls(control_name)
            control_name = [[['ML10M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant'], ['0.5']]]
            ml10m_item_controls = make_controls(control_name)
            ml10m_controls = ml10m_user_controls + ml10m_item_controls
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['constant'], ['0.5']]]
            ml20m_user_controls = make_controls(control_name)
            control_name = [[['ML20M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant'], ['0.5']]]
            ml20m_item_controls = make_controls(control_name)
            ml20m_controls = ml20m_user_controls + ml20m_item_controls
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['user', 'item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant'], ['0.5']]]
            nfp_controls = make_controls(control_name)
            controls.extend(nfp_controls)
        if 'Douban' in data:
            control_name = [[['Douban'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'], ['genre'], ['constant-0.1'],
                             ['constant'], ['0.5']]]
            douban_controls = make_controls(control_name)
            controls.extend(douban_controls)
        if 'Amazon' in data:
            control_name = [[['Amazon'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'], ['genre'], ['constant-0.1'],
                             ['constant'], ['0.5']]]
            amazon_controls = make_controls(control_name)
            controls.extend(amazon_controls)
    elif file == 'pl':
        controls = []
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['constant'], ['1'], ['dp-10', 'ip-10']]]
            ml100k_user_controls = make_controls(control_name)
            control_name = [[['ML100K'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant'], ['1'], ['dp-10', 'ip-10']]]
            ml100k_item_controls = make_controls(control_name)
            ml100k_controls = ml100k_user_controls + ml100k_item_controls
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['constant'], ['1'], ['dp-10', 'ip-10']]]
            ml1m_user_controls = make_controls(control_name)
            control_name = [[['ML1M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant'], ['1'], ['dp-10', 'ip-10']]]
            ml1m_item_controls = make_controls(control_name)
            ml1m_controls = ml1m_user_controls + ml1m_item_controls
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['constant'], ['1'], ['dp-10', 'ip-10']]]
            ml10m_user_controls = make_controls(control_name)
            control_name = [[['ML10M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant'], ['1'], ['dp-10', 'ip-10']]]
            ml10m_item_controls = make_controls(control_name)
            ml10m_controls = ml10m_user_controls + ml10m_item_controls
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['genre'], ['constant-0.1'], ['constant'], ['1'], ['dp-10', 'ip-10']]]
            ml20m_user_controls = make_controls(control_name)
            control_name = [[['ML20M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant'], ['1'], ['dp-10', 'ip-10']]]
            ml20m_item_controls = make_controls(control_name)
            ml20m_controls = ml20m_user_controls + ml20m_item_controls
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['user', 'item'], ['explicit', 'implicit'], ['ae'], ['0'],
                             ['random-8'], ['constant-0.1'], ['constant'], ['1'], ['dp-10', 'ip-10']]]
            nfp_controls = make_controls(control_name)
            controls.extend(nfp_controls)
        if 'Douban' in data:
            control_name = [[['Douban'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'], ['genre'], ['constant-0.1'],
                             ['constant'], ['1'], ['dp-10', 'ip-10']]]
            douban_controls = make_controls(control_name)
            controls.extend(douban_controls)
        if 'Amazon' in data:
            control_name = [[['Amazon'], ['user'], ['explicit', 'implicit'], ['ae'], ['0'], ['genre'], ['constant-0.1'],
                             ['constant'], ['1'], ['dp-10', 'ip-10']]]
            amazon_controls = make_controls(control_name)
            controls.extend(amazon_controls)
    else:
        raise ValueError('Not valid file')
    return controls


def main():
    write = False
    data = ['ML100K', 'ML1M', 'ML10M', 'Douban', 'Amazon']
    # files = ['joint', 'alone', 'assist', 'info', 'ar', 'aw', 'ar-optim', 'match', 'optim-optim', 'pl']
    files = ['joint', 'alone', 'assist', 'ar', 'ar-optim']
    controls = []
    for file in files:
        controls += make_control_list(data, file)
    processed_result = process_result(controls)
    save(processed_result, os.path.join(result_path, 'processed_result.pt'))
    extracted_processed_result = {}
    extract_processed_result(extracted_processed_result, processed_result, [])
    df = make_df(extracted_processed_result, write)
    make_vis(df, 'RMSE')
    make_vis(df, 'MAP')
    return


def process_result(controls):
    processed_result = {}
    for control in controls:
        model_tag = '_'.join(control)
        extract_result(list(control), model_tag, processed_result)
    summarize_result(processed_result)
    return processed_result


def extract_result(control, model_tag, processed_result):
    metric_name = ['MAP', 'RMSE']
    if len(control) == 1:
        exp_idx = exp.index(control[0])
        base_result_path_i = os.path.join(result_path, '{}.pt'.format(model_tag))
        if os.path.exists(base_result_path_i):
            base_result = load(base_result_path_i)
            model_tag_list = model_tag.split('_')
            if len(model_tag_list) == 6:
                for k in base_result['logger']['test'].mean:
                    metric_name = k.split('/')[1]
                    if metric_name not in processed_result:
                        processed_result[metric_name] = {'train': [None for _ in range(num_experiments)],
                                                         'test': [None for _ in range(num_experiments)]}
                    processed_result[metric_name]['train'][exp_idx] = base_result['logger']['train'].history[k]
                    processed_result[metric_name]['test'][exp_idx] = base_result['logger']['test'].mean[k]
            elif len(model_tag_list) == 7:
                for k in base_result['logger']['test'].mean:
                    metric_name = k.split('/')[1]
                    if metric_name not in processed_result:
                        processed_result[metric_name] = {'train': [None for _ in range(num_experiments)],
                                                         'test': [None for _ in range(num_experiments)],
                                                         'test_each': [None for _ in range(num_experiments)]}
                    processed_result[metric_name]['train'][exp_idx] = base_result['logger']['train'].history[k]
                    processed_result[metric_name]['test'][exp_idx] = base_result['logger']['test'].mean[k]
                    processed_result[metric_name]['test_each'][exp_idx] = base_result['logger']['test_each'].history[k]
            elif len(model_tag_list) in [9, 10, 11]:
                for k in base_result['logger']['test'].history:
                    metric_name = k.split('/')[1]
                    if metric_name not in processed_result:
                        processed_result[metric_name] = {'train': [None for _ in range(num_experiments)],
                                                         'test': [None for _ in range(num_experiments)],
                                                         'test_each': [None for _ in range(num_experiments)],
                                                         'test_history': [None for _ in range(num_experiments)]}
                    processed_result[metric_name]['train'][exp_idx] = base_result['logger']['train'].history[k]
                    if metric_name in ['Loss', 'RMSE']:
                        processed_result[metric_name]['test'][exp_idx] = min(base_result['logger']['test'].history[k])
                        processed_result[metric_name]['test_each'][exp_idx] = \
                            np.array(base_result['logger']['test_each'].history[k]).reshape(-1, 11).min(axis=-1)
                    else:
                        processed_result[metric_name]['test'][exp_idx] = max(base_result['logger']['test'].history[k])
                        processed_result[metric_name]['test_each'][exp_idx] = \
                            np.array(base_result['logger']['test_each'].history[k]).reshape(-1, 11).max(axis=-1)
                    processed_result[metric_name]['test_history'][exp_idx] = base_result['logger']['test'].history[k]
            else:
                raise ValueError('Not valid model tag')
        else:
            print('Missing {}'.format(base_result_path_i))
    else:
        if control[1] not in processed_result:
            processed_result[control[1]] = {}
            processed_result[control[1]] = {}
        extract_result([control[0]] + control[2:], model_tag, processed_result[control[1]])
    return


def summarize_result(processed_result):
    pivot = ['train', 'test', 'test_each', 'test_history']
    leaf = False
    for k, v in processed_result.items():
        if k in pivot:
            leaf = True
            for x in processed_result[k]:
                if x is not None:
                    tmp = x
            for i in range(len(processed_result[k])):
                if processed_result[k][i] is None:
                    processed_result[k][i] = tmp
            e1 = [len(x) for x in processed_result[k] if isinstance(x, list)]
            for i in range(len(e1)):
                if e1[i] in [201, 12]:
                    if isinstance(processed_result[k][i], list):
                        tmp_processed_result = None
                        for j in range(1, len(processed_result[k][i])):
                            if processed_result[k][i][j - 1] == processed_result[k][i][j]:
                                tmp_processed_result = processed_result[k][i][:j] + processed_result[k][i][j + 1:]
                        if tmp_processed_result is not None:
                            processed_result[k][i] = tmp_processed_result
                if e1[i] > 18 and e1[i] < 200:
                    if isinstance(processed_result[k][i], list):
                        tmp_processed_result = processed_result[k][i] + [processed_result[k][i][-1]] * (
                                200 - len(processed_result[k][i]))
                        processed_result[k][i] = tmp_processed_result
                if e1[i] > 200:
                    if isinstance(processed_result[k][i], list):
                        tmp_processed_result = processed_result[k][i][:200]
                        processed_result[k][i] = tmp_processed_result
            stacked_result = np.stack(processed_result[k], axis=0)
            processed_result[k] = {}
            processed_result[k]['mean'] = np.mean(stacked_result, axis=0)
            processed_result[k]['std'] = np.std(stacked_result, axis=0)
            processed_result[k]['max'] = np.max(stacked_result, axis=0)
            processed_result[k]['min'] = np.min(stacked_result, axis=0)
            processed_result[k]['argmax'] = np.argmax(stacked_result, axis=0)
            processed_result[k]['argmin'] = np.argmin(stacked_result, axis=0)
            processed_result[k]['val'] = stacked_result.tolist()
    if not leaf:
        for k, v in processed_result.items():
            summarize_result(v)
        return
    return


def extract_processed_result(extracted_processed_result, processed_result, control):
    pivot = ['train', 'test', 'test_each', 'test_history']
    leaf = False
    for k, v in processed_result.items():
        if k in pivot:
            leaf = True
            exp_name = '_'.join(control[:-1])
            metric_name = control[-1]
            if exp_name not in extracted_processed_result:
                extracted_processed_result[exp_name] = {p: defaultdict() for p in processed_result.keys()}
            extracted_processed_result[exp_name][k]['{}_mean'.format(metric_name)] = processed_result[k]['mean']
            extracted_processed_result[exp_name][k]['{}_std'.format(metric_name)] = processed_result[k]['std']
    if not leaf:
        for k, v in processed_result.items():
            extract_processed_result(extracted_processed_result, v, control + [k])
    return


def make_df(extracted_processed_result, write):
    pivot = ['train', 'test', 'test_each', 'test_history']
    df = {p: defaultdict(list) for p in pivot}
    for exp_name in extracted_processed_result:
        control = exp_name.split('_')
        for p in extracted_processed_result[exp_name]:
            if len(control) == 5:
                data_name, data_mode, target_mode, model_name, info = control
                index_name = [model_name]
                df_name = '_'.join([data_name, data_mode, target_mode, info])
            elif len(control) == 6:
                data_name, data_mode, target_mode, model_name, info, data_split_mode = control
                index_name = [model_name]
                df_name = '_'.join([data_name, data_mode, target_mode, info, data_split_mode])
            elif len(control) == 8:
                data_name, data_mode, target_mode, model_name, info, data_split_mode, ar, aw = control
                index_name = ['_'.join([model_name, ar, aw])]
                df_name = '_'.join([data_name, data_mode, target_mode, info, data_split_mode])
            elif len(control) == 9:
                data_name, data_mode, target_mode, model_name, info, data_split_mode, ar, aw, match = control
                index_name = ['_'.join([model_name, ar, aw, match])]
                df_name = '_'.join([data_name, data_mode, target_mode, info, data_split_mode])
            elif len(control) == 10:
                data_name, data_mode, target_mode, model_name, info, data_split_mode, ar, aw, match, pl = control
                index_name = ['_'.join([model_name, ar, aw, match, pl])]
                df_name = '_'.join([data_name, data_mode, target_mode, info, data_split_mode])
            else:
                raise ValueError('Not valid control')
            metric = list(extracted_processed_result[exp_name][p].keys())
            if len(metric) > 0:
                if len(extracted_processed_result[exp_name][p][metric[0]].shape) == 0:
                    df[p][df_name].append(pd.DataFrame(data=extracted_processed_result[exp_name][p], index=index_name))
                else:
                    for m in metric:
                        df_name_ = '{}_{}'.format(df_name, m)
                        df[p][df_name_].append(
                            pd.DataFrame(data=extracted_processed_result[exp_name][p][m].reshape(1, -1),
                                         index=index_name))
    if write:
        for p in pivot:
            startrow = 0
            writer = pd.ExcelWriter('{}/{}.xlsx'.format(result_path, p), engine='xlsxwriter')
            for df_name in df[p]:
                df[p][df_name] = pd.concat(df[p][df_name])
                df[p][df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1)
                writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
                startrow = startrow + len(df[p][df_name].index) + 3
            writer.save()
    else:
        for p in pivot:
            for df_name in df[p]:
                df[p][df_name] = pd.concat(df[p][df_name])
    return df


def make_vis(df, vis_mode):
    control_dict = {'Joint': 'Joint', 'ae': 'MTAL', 'Alone': 'Alone'}
    color_dict = {'Joint': 'black', 'ae': 'red', 'Alone': 'blue'}
    linestyle_dict = {'Joint': '-', 'ae': '--', 'Alone': ':'}
    label_loc_dict = {'RMSE': 'upper right', 'MAP': 'lower right'}
    marker_dict = {'Joint': 'X', 'ae': 'D', 'Alone': 's'}
    fontsize = {'legend': 16, 'label': 16, 'ticks': 16}
    ae_ablation = ['constant-0.1_constant', 'constant-0.3_constant', 'optim-0.1_constant']
    alone_ablation = ['base', 'mf', 'mlp', 'nmf', 'ae']
    figsize = (5, 4)
    fig = {}
    ax_dict_1 = {}
    for df_name in df['test_history']:
        data_name, data_mode, target_mode, info, data_split_mode, metric_name, stat = df_name.split('_')
        valid_mask = stat == 'mean' and metric_name == vis_mode
        if valid_mask:
            df_name_std = '_'.join([data_name, data_mode, target_mode, info, data_split_mode, metric_name, 'std'])
            df_name_joint = '_'.join([data_name, data_mode, target_mode, info])
            df_name_alone = '_'.join([data_name, data_mode, target_mode, info, data_split_mode])
            fig_name = '_'.join([data_name, data_mode, target_mode, info, data_split_mode, metric_name])
            fig[fig_name] = plt.figure(fig_name, figsize=figsize)
            if fig_name not in ax_dict_1:
                ax_dict_1[fig_name] = fig[fig_name].add_subplot(111)
            ax_1 = ax_dict_1[fig_name]
            print(df_name, df_name_std, df_name_joint, df_name_alone)
            x = np.arange(11)
            pivot = float('inf') if vis_mode in ['RMSE'] else -float('inf')
            pivot_index = None
            for (index, row) in df['test'][df_name_joint].iterrows():
                y_index = row['{}_mean'.format(metric_name)]
                if vis_mode in ['RMSE']:
                    if y_index < pivot:
                        pivot_index = index
                        pivot = y_index
                else:
                    if y_index > pivot:
                        pivot_index = index
                        pivot = y_index
            control = 'Joint'
            y_joint = df['test'][df_name_joint]['{}_mean'.format(metric_name)].loc[pivot_index]
            y_joint = np.full(x.shape, y_joint)
            y_err_joint = df['test'][df_name_joint]['{}_std'.format(metric_name)].loc[pivot_index]
            y_err_joint = np.full(x.shape, y_err_joint)
            ax_1.errorbar(x, y_joint, yerr=y_err_joint, color=color_dict[control], linestyle=linestyle_dict[control],
                          label=control_dict[control])
            pivot = float('inf') if vis_mode in ['RMSE'] else -float('inf')
            pivot_index = None
            for (index, row) in df['test'][df_name_alone].iterrows():
                control = index
                if control in alone_ablation:
                    y_index = row['{}_mean'.format(metric_name)]
                    if vis_mode in ['RMSE']:
                        if y_index < pivot:
                            pivot_index = index
                            pivot = y_index
                    else:
                        if y_index > pivot:
                            pivot_index = index
                            pivot = y_index
            control = 'Alone'
            y_alone = df['test'][df_name_alone]['{}_mean'.format(metric_name)].loc[pivot_index]
            y_alone = np.full(x.shape, y_alone)
            y_err_alone = df['test'][df_name_alone]['{}_std'.format(metric_name)].loc[pivot_index]
            y_err_alone = np.full(x.shape, y_err_alone)
            ax_1.errorbar(x, y_alone, yerr=y_err_alone, color=color_dict[control], linestyle=linestyle_dict[control],
                          label=control_dict[control])
            pivot = float('inf') if vis_mode in ['RMSE'] else -float('inf')
            pivot_index = None
            for ((index, row), (_, row_std)) in zip(df['test_history'][df_name].iterrows(),
                                                    df['test_history'][df_name_std].iterrows()):
                control = '_'.join(index.split('_')[1:])
                if control in ae_ablation:
                    if vis_mode in ['RMSE']:
                        y_index = min(row.to_numpy()).item()
                        if y_index < pivot:
                            pivot_index = index
                            pivot = y_index
                    else:
                        y_index = max(row.to_numpy()).item()
                        if y_index > pivot:
                            pivot_index = index
                            pivot = y_index
            control = 'ae'
            y = df['test_history'][df_name].loc[pivot_index].to_numpy()
            y_err = df['test_history'][df_name_std].loc[pivot_index].to_numpy()
            ax_1.errorbar(x, y, yerr=y_err, color=color_dict[control], linestyle=linestyle_dict[control],
                          label=control_dict[control], marker=marker_dict[control])
            ax_1.set_xticks(x)
            ax_1.set_xlabel('Assistance Rounds', fontsize=fontsize['label'])
            ax_1.set_ylabel(vis_mode, fontsize=fontsize['label'])
            ax_1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.legend(loc=label_loc_dict[vis_mode], fontsize=fontsize['legend'])
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        fig[fig_name].tight_layout()
        control = fig_name.split('_')
        dir_path = os.path.join(vis_path, vis_mode, *control[:-1])
        fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


if __name__ == '__main__':
    main()
