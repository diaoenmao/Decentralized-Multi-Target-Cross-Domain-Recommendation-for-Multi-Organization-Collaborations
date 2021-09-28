import os
import itertools
import json
import numpy as np
import pandas as pd
from utils import save, load, makedir_exist_ok
import matplotlib.pyplot as plt
from collections import defaultdict

result_path = './output/result'
save_format = 'pdf'
vis_path = './output/vis/{}'.format(save_format)
num_experiments = 4
exp = [str(x) for x in list(range(num_experiments))]


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
            control_name = [[['ML100K'], ['user', 'item'], ['explicit', 'implicit'], ['mf', 'gmf', 'mlp', 'nmf', 'ae'],
                             ['0', '1']]]
            ml100k_controls = make_controls(control_name)
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user', 'item'], ['explicit', 'implicit'], ['mf', 'gmf', 'mlp', 'nmf', 'ae'],
                             ['0', '1']]]
            ml1m_controls = make_controls(control_name)
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user', 'item'], ['explicit', 'implicit'], ['ae'],
                             ['0', '1']]]
            ml10m_controls = make_controls(control_name)
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user', 'item'], ['explicit', 'implicit'], ['ae'],
                             ['0', '1']]]
            ml20m_controls = make_controls(control_name)
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['ML20M'], ['user', 'item'], ['explicit', 'implicit'], ['ae'],
                             ['0', '1']]]
            nfp_controls = make_controls(control_name)
            controls.extend(nfp_controls)
    elif file == 'alone':
        controls = []
        control_name = [[data, ['user'], ['explicit', 'implicit'], ['base'], ['0'], ['genre', 'random-8']]]
        base_user_controls = make_controls(control_name)
        control_name = [[data, ['item'], ['explicit', 'implicit'], ['base'], ['0'], ['random-8']]]
        base_item_controls = make_controls(control_name)
        base_controls = base_user_controls + base_item_controls
        controls.extend(base_controls)
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['mf', 'gmf', 'mlp', 'nmf', 'ae'],
                             ['0', '1'], ['genre', 'random-8']]]
            ml100k_user_controls = make_controls(control_name)
            control_name = [[['ML100K'], ['item'], ['explicit', 'implicit'], ['mf', 'gmf', 'mlp', 'nmf', 'ae'],
                             ['0', '1'], ['random-8']]]
            ml100k_item_controls = make_controls(control_name)
            ml100k_controls = ml100k_user_controls + ml100k_item_controls
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['mf', 'gmf', 'mlp', 'nmf', 'ae'],
                             ['0', '1'], ['genre', 'random-8']]]
            ml1m_user_controls = make_controls(control_name)
            control_name = [[['ML1M'], ['item'], ['explicit', 'implicit'], ['mf', 'gmf', 'mlp', 'nmf', 'ae'],
                             ['0', '1'], ['random-8']]]
            ml1m_item_controls = make_controls(control_name)
            ml1m_controls = ml1m_user_controls + ml1m_item_controls
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['ae'],
                             ['0', '1'], ['genre', 'random-8']]]
            ml10m_user_controls = make_controls(control_name)
            control_name = [[['ML10M'], ['item'], ['explicit', 'implicit'], ['ae'],
                             ['0', '1'], ['random-8']]]
            ml10m_item_controls = make_controls(control_name)
            ml10m_controls = ml10m_user_controls + ml10m_item_controls
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['ae'],
                             ['0', '1'], ['genre', 'random-8']]]
            ml20m_user_controls = make_controls(control_name)
            control_name = [[['ML20M'], ['item'], ['explicit', 'implicit'], ['ae'],
                             ['0', '1'], ['random-8']]]
            ml20m_item_controls = make_controls(control_name)
            ml20m_controls = ml20m_user_controls + ml20m_item_controls
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['user', 'item'], ['explicit', 'implicit'], ['ae'],
                             ['0', '1'], ['random-8']]]
            nfp_controls = make_controls(control_name)
            controls.extend(nfp_controls)
    elif file == 'assist':
        controls = []
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['genre', 'random-8'], ['constant-0.1'], ['constant']]]
            ml100k_user_controls = make_controls(control_name)
            control_name = [[['ML100K'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.1'], ['constant']]]
            ml100k_item_controls = make_controls(control_name)
            ml100k_controls = ml100k_user_controls + ml100k_item_controls
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['genre', 'random-8'], ['constant-0.1'], ['constant']]]
            ml1m_user_controls = make_controls(control_name)
            control_name = [[['ML1M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.1'], ['constant']]]
            ml1m_item_controls = make_controls(control_name)
            ml1m_controls = ml1m_user_controls + ml1m_item_controls
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['genre', 'random-8'], ['constant-0.1'], ['constant']]]
            ml10m_user_controls = make_controls(control_name)
            control_name = [[['ML10M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.1'], ['constant']]]
            ml10m_item_controls = make_controls(control_name)
            ml10m_controls = ml10m_user_controls + ml10m_item_controls
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['genre', 'random-8'], ['constant-0.1'], ['constant']]]
            ml20m_user_controls = make_controls(control_name)
            control_name = [[['ML20M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.1'], ['constant']]]
            ml20m_item_controls = make_controls(control_name)
            ml20m_controls = ml20m_user_controls + ml20m_item_controls
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['user', 'item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.1'], ['constant']]]
            nfp_controls = make_controls(control_name)
            controls.extend(nfp_controls)
    elif file == 'ar':
        controls = []
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.3'], ['constant']]]
            ml100k_user_controls = make_controls(control_name)
            control_name = [[['ML100K'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.3'], ['constant']]]
            ml100k_item_controls = make_controls(control_name)
            ml100k_controls = ml100k_user_controls + ml100k_item_controls
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.3'], ['constant']]]
            ml1m_user_controls = make_controls(control_name)
            control_name = [[['ML1M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.3'], ['constant']]]
            ml1m_item_controls = make_controls(control_name)
            ml1m_controls = ml1m_user_controls + ml1m_item_controls
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.3'], ['constant']]]
            ml10m_user_controls = make_controls(control_name)
            control_name = [[['ML10M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.3'], ['constant']]]
            ml10m_item_controls = make_controls(control_name)
            ml10m_controls = ml10m_user_controls + ml10m_item_controls
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.3'], ['constant']]]
            ml20m_user_controls = make_controls(control_name)
            control_name = [[['ML20M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.3'], ['constant']]]
            ml20m_item_controls = make_controls(control_name)
            ml20m_controls = ml20m_user_controls + ml20m_item_controls
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['user', 'item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.3'], ['constant']]]
            nfp_controls = make_controls(control_name)
            controls.extend(nfp_controls)
    elif file == 'aw':
        controls = []
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['genre'], ['constant-0.1'], ['optim']]]
            ml100k_user_controls = make_controls(control_name)
            ml100k_controls = ml100k_user_controls
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['genre'], ['constant-0.1'], ['optim']]]
            ml1m_user_controls = make_controls(control_name)
            ml1m_controls = ml1m_user_controls
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['genre'], ['constant-0.1'], ['optim']]]
            ml10m_user_controls = make_controls(control_name)
            ml10m_controls = ml10m_user_controls
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['genre'], ['constant-0.1'], ['optim']]]
            ml20m_user_controls = make_controls(control_name)
            ml20m_controls = ml20m_user_controls
            controls.extend(ml20m_controls)
    elif file == 'ar-optim':
        controls = []
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['optim-0.1'], ['constant']]]
            ml100k_user_controls = make_controls(control_name)
            control_name = [[['ML100K'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['optim-0.1'], ['constant']]]
            ml100k_item_controls = make_controls(control_name)
            ml100k_controls = ml100k_user_controls + ml100k_item_controls
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['optim-0.1'], ['constant']]]
            ml1m_user_controls = make_controls(control_name)
            control_name = [[['ML1M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['optim-0.1'], ['constant']]]
            ml1m_item_controls = make_controls(control_name)
            ml1m_controls = ml1m_user_controls + ml1m_item_controls
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['optim-0.1'], ['constant']]]
            ml10m_user_controls = make_controls(control_name)
            control_name = [[['ML10M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['optim-0.1'], ['constant']]]
            ml10m_item_controls = make_controls(control_name)
            ml10m_controls = ml10m_user_controls + ml10m_item_controls
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['optim-0.1'], ['constant']]]
            ml20m_user_controls = make_controls(control_name)
            control_name = [[['ML20M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['optim-0.1'], ['constant']]]
            ml20m_item_controls = make_controls(control_name)
            ml20m_controls = ml20m_user_controls + ml20m_item_controls
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['user', 'item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['optim-0.1'], ['constant']]]
            nfp_controls = make_controls(control_name)
            controls.extend(nfp_controls)
    else:
        raise ValueError('Not valid file')
    return controls


def main():
    data = ['ML100K', 'ML1M']
    joint_control_list = make_control_list(data, 'joint')
    alone_control_list = make_control_list(data, 'alone')
    assist_control_list = make_control_list(data, 'assist')
    if 'ML100K' in data or 'ML1M' in data:
        ar_control_list = make_control_list(data, 'ar')
        ar_optim_control_list = make_control_list(data, 'ar-optim')
        aw_epoch_control_list = make_control_list(data, 'aw')
        controls = joint_control_list + alone_control_list + assist_control_list + ar_control_list + \
                   ar_optim_control_list + aw_epoch_control_list
    else:
        controls = joint_control_list + alone_control_list + assist_control_list
    processed_result_exp, processed_result_history = process_result(controls)
    print(processed_result_exp)
    exit()
    with open('{}/processed_result_exp.json'.format(result_path), 'w') as fp:
        json.dump(processed_result_exp, fp, indent=2)
    save(processed_result_exp, os.path.join(result_path, 'processed_result_exp.pt'))
    save(processed_result_history, os.path.join(result_path, 'processed_result_history.pt'))
    extracted_processed_result_exp = {}
    extracted_processed_result_history = {}
    extract_processed_result(extracted_processed_result_exp, processed_result_exp, [])
    extract_processed_result(extracted_processed_result_history, processed_result_history, [])
    df_exp = make_df_exp(extracted_processed_result_exp)
    df_history = make_df_history(extracted_processed_result_history)
    make_vis(df_history)
    return


def process_result(controls):
    processed_result_exp, processed_result_history = {}, {}
    for control in controls:
        model_tag = '_'.join(control)
        extract_result(list(control), model_tag, processed_result_exp, processed_result_history)
    summarize_result(processed_result_exp)
    summarize_result(processed_result_history)
    return processed_result_exp, processed_result_history


def extract_result(control, model_tag, processed_result_exp, processed_result_history):
    if len(control) == 1:
        exp_idx = exp.index(control[0])
        base_result_path_i = os.path.join(result_path, '{}.pt'.format(model_tag))
        if os.path.exists(base_result_path_i):
            base_result = load(base_result_path_i)
            if model_tag == '0_ML100K_user_explicit_ae_0_random-9_constant-0.1_constant':
                print(base_result['logger']['test'].history)
                exit()
            for k in base_result['logger']['test'].mean:
                metric_name = k.split('/')[1]
                if metric_name not in processed_result_exp:
                    processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
                    processed_result_history[metric_name] = {'history': [None for _ in range(num_experiments)]}
                processed_result_exp[metric_name]['exp'][exp_idx] = base_result['logger']['test'].mean[k]
                processed_result_history[metric_name]['history'][exp_idx] = base_result['logger']['train'].history[k]
        else:
            print('Missing {}'.format(base_result_path_i))
    else:
        if control[1] not in processed_result_exp:
            processed_result_exp[control[1]] = {}
            processed_result_history[control[1]] = {}
        extract_result([control[0]] + control[2:], model_tag, processed_result_exp[control[1]],
                       processed_result_history[control[1]])
    return


def summarize_result(processed_result):
    if 'exp' in processed_result:
        pivot = 'exp'
        processed_result[pivot] = np.stack(processed_result[pivot], axis=0)
        processed_result['mean'] = np.mean(processed_result[pivot], axis=0).item()
        processed_result['std'] = np.std(processed_result[pivot], axis=0).item()
        processed_result['max'] = np.max(processed_result[pivot], axis=0).item()
        processed_result['min'] = np.min(processed_result[pivot], axis=0).item()
        processed_result['argmax'] = np.argmax(processed_result[pivot], axis=0).item()
        processed_result['argmin'] = np.argmin(processed_result[pivot], axis=0).item()
        processed_result[pivot] = processed_result[pivot].tolist()
    elif 'history' in processed_result:
        pivot = 'history'
        processed_result[pivot] = np.stack(processed_result[pivot], axis=0)
        processed_result['mean'] = np.mean(processed_result[pivot], axis=0)
        processed_result['std'] = np.std(processed_result[pivot], axis=0)
        processed_result['max'] = np.max(processed_result[pivot], axis=0)
        processed_result['min'] = np.min(processed_result[pivot], axis=0)
        processed_result['argmax'] = np.argmax(processed_result[pivot], axis=0)
        processed_result['argmin'] = np.argmin(processed_result[pivot], axis=0)
        processed_result[pivot] = processed_result[pivot].tolist()
    else:
        for k, v in processed_result.items():
            summarize_result(v)
        return
    return


def extract_processed_result(extracted_processed_result, processed_result, control):
    if 'exp' in processed_result or 'history' in processed_result:
        exp_name = '_'.join(control[:-1])
        metric_name = control[-1]
        if exp_name not in extracted_processed_result:
            extracted_processed_result[exp_name] = defaultdict()
        extracted_processed_result[exp_name]['{}_mean'.format(metric_name)] = processed_result['mean']
        extracted_processed_result[exp_name]['{}_std'.format(metric_name)] = processed_result['std']
    else:
        for k, v in processed_result.items():
            extract_processed_result(extracted_processed_result, v, control + [k])
    return


def make_df_exp(extracted_processed_result_exp):
    df = defaultdict(list)
    for exp_name in extracted_processed_result_exp:
        control = exp_name.split('_')
        if len(control) == 3:
            data_name, model_name, num_supervised = control
            index_name = ['1']
            df_name = '_'.join([data_name, model_name, num_supervised])
        else:
            data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode, \
            local_epoch, gm, all_sbn, = control
            index_name = ['_'.join([local_epoch, gm])]
            df_name = '_'.join(
                [data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode, all_sbn])
        df[df_name].append(pd.DataFrame(data=extracted_processed_result_exp[exp_name], index=index_name))
    startrow = 0
    writer = pd.ExcelWriter('{}/result_exp.xlsx'.format(result_path), engine='xlsxwriter')
    for df_name in df:
        df[df_name] = pd.concat(df[df_name])
        df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1)
        writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
        startrow = startrow + len(df[df_name].index) + 3
    writer.save()
    return df


def make_df_history(extracted_processed_result_history):
    df = defaultdict(list)
    for exp_name in extracted_processed_result_history:
        control = exp_name.split('_')
        if len(control) == 3:
            data_name, model_name, num_supervised = control
            index_name = ['1']
            for k in extracted_processed_result_history[exp_name]:
                df_name = '_'.join([data_name, model_name, num_supervised, k])
                df[df_name].append(
                    pd.DataFrame(data=extracted_processed_result_history[exp_name][k].reshape(1, -1), index=index_name))
        else:
            data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode, \
            local_epoch, gm, all_sbn = control
            index_name = ['_'.join([local_epoch, gm])]
            for k in extracted_processed_result_history[exp_name]:
                df_name = '_'.join(
                    [data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode, all_sbn,
                     k])
                df[df_name].append(
                    pd.DataFrame(data=extracted_processed_result_history[exp_name][k].reshape(1, -1), index=index_name))
    startrow = 0
    writer = pd.ExcelWriter('{}/result_history.xlsx'.format(result_path), engine='xlsxwriter')
    for df_name in df:
        df[df_name] = pd.concat(df[df_name])
        df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1)
        writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
        startrow = startrow + len(df[df_name].index) + 3
    writer.save()
    return df


def make_vis(df):
    data_split_mode_dict = {'iid': 'IID', 'non-iid-l-2': 'Non-IID, $K=2$',
                            'non-iid-d-0.1': 'Non-IID, $\operatorname{Dir}(0.1)$',
                            'non-iid-d-0.3': 'Non-IID, $\operatorname{Dir}(0.3)$'}
    color = {'5_0.5': 'red', '1_0.5': 'orange', '5_0': 'dodgerblue', '5_0.9': 'blue', '5_0.5_nomixup': 'green',
             'iid': 'red', 'non-iid-l-2': 'orange', 'non-iid-d-0.1': 'dodgerblue', 'non-iid-d-0.3': 'green'}
    linestyle = {'5_0.5': '-', '1_0.5': '--', '5_0': ':', '5_0.5_nomixup': '-.', '5_0.9': (0, (1, 5)),
                 'iid': '-', 'non-iid-l-2': '--', 'non-iid-d-0.1': '-.', 'non-iid-d-0.3': ':'}
    loc_dict = {'Accuracy': 'lower right', 'Loss': 'upper right'}
    fontsize = {'legend': 16, 'label': 16, 'ticks': 16}
    fig = {}
    reorder_fig = []
    for df_name in df:
        df_name_list = df_name.split('_')
        if len(df_name_list) == 5:
            data_name, model_name, num_supervised, metric_name, stat = df_name.split('_')
            if stat == 'std':
                continue
            df_name_std = '_'.join([data_name, model_name, num_supervised, metric_name, 'std'])
            fig_name = '_'.join([data_name, model_name, num_supervised, metric_name])
            fig[fig_name] = plt.figure(fig_name)
            for ((index, row), (_, row_std)) in zip(df[df_name].iterrows(), df[df_name_std].iterrows()):
                y = row.to_numpy()
                yerr = row_std.to_numpy()
                x = np.arange(len(y))
                plt.plot(x, y, color='r', linestyle='-')
                plt.fill_between(x, (y - yerr), (y + yerr), color='r', alpha=.1)
                plt.xlabel('Communication Rounds', fontsize=fontsize['label'])
                plt.ylabel(metric_name, fontsize=fontsize['label'])
                plt.xticks(fontsize=fontsize['ticks'])
                plt.yticks(fontsize=fontsize['ticks'])

        else:
            data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode, all_sbn, \
            metric_name, stat = df_name.split('_')
            if stat == 'std':
                continue
            df_name_std = '_'.join(
                [data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, data_split_mode, all_sbn,
                 metric_name, 'std'])
            for ((index, row), (_, row_std)) in zip(df[df_name].iterrows(), df[df_name_std].iterrows()):
                y = row.to_numpy()
                yerr = row_std.to_numpy()
                x = np.arange(len(y))
                if index == '5_0.5' and loss_mode == 'fix-mix':
                    fig_name = '_'.join(
                        [data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, all_sbn, metric_name])
                    reorder_fig.append(fig_name)
                    style = data_split_mode
                    fig[fig_name] = plt.figure(fig_name)
                    label_name = '{}'.format(data_split_mode_dict[data_split_mode])
                    plt.plot(x, y, color=color[style], linestyle=linestyle[style], label=label_name)
                    plt.fill_between(x, (y - yerr), (y + yerr), color=color[style], alpha=.1)
                    plt.legend(loc=loc_dict[metric_name], fontsize=fontsize['legend'])
                    plt.xlabel('Communication Rounds', fontsize=fontsize['label'])
                    plt.ylabel(metric_name, fontsize=fontsize['label'])
                    plt.xticks(fontsize=fontsize['ticks'])
                    plt.yticks(fontsize=fontsize['ticks'])
                if data_split_mode in ['iid', 'non-iid-l-2']:
                    fig_name = '_'.join(
                        [data_name, model_name, num_supervised, num_clients, active_rate, data_split_mode, all_sbn,
                         metric_name])
                    reorder_fig.append(fig_name)
                    fig[fig_name] = plt.figure(fig_name)
                    local_epoch, gm = index.split('_')
                    if loss_mode == 'fix':
                        label_name = '$E={}$, $\\beta_g={}$, No mixup'.format(local_epoch, gm)
                        style = '{}_nomixup'.format(index)
                    else:
                        label_name = '$E={}$, $\\beta_g={}$'.format(local_epoch, gm)
                        style = index
                    plt.plot(x, y, color=color[style], linestyle=linestyle[style], label=label_name)
                    plt.fill_between(x, (y - yerr), (y + yerr), color=color[style], alpha=.1)
                    plt.legend(loc=loc_dict[metric_name], fontsize=fontsize['legend'])
                    plt.xlabel('Communication Rounds', fontsize=fontsize['label'])
                    plt.ylabel(metric_name, fontsize=fontsize['label'])
                    plt.xticks(fontsize=fontsize['ticks'])
                    plt.yticks(fontsize=fontsize['ticks'])
    for fig_name in reorder_fig:
        data_name, model_name, num_supervised, loss_mode, num_clients, active_rate, all_sbn, metric_name = fig_name.split(
            '_')
        plt.figure(fig_name)
        handles, labels = plt.gca().get_legend_handles_labels()
        if len(handles) == 4:
            handles = [handles[0], handles[3], handles[2], handles[1]]
            labels = [labels[0], labels[3], labels[2], labels[1]]
            plt.legend(handles, labels, loc=loc_dict[metric_name], fontsize=fontsize['legend'])
        if len(handles) == 5:
            handles = [handles[0], handles[4], handles[2], handles[3], handles[1]]
            labels = [labels[0], labels[4], labels[2], labels[3], labels[1]]
            plt.legend(handles, labels, loc=loc_dict[metric_name], fontsize=fontsize['legend'])
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        plt.grid()
        fig_path = '{}/{}.{}'.format(vis_path, fig_name, save_format)
        makedir_exist_ok(vis_path)
        plt.savefig(fig_path, dpi=500, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


if __name__ == '__main__':
    main()
