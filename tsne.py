import glob
import os
from os.path import basename
import re
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

import config_util
import load_data
from load_data import DataLoader, get_x, get_y

file_to_process = [
    {
        'name_pattern': '{}/{}_{}_scaledData_*.csv',
        'process': True
    }
]


def tsne_experiment(dataset, problem, problem_to_process_1, problem_to_process_2=None):
    if problem_to_process_2 is not None:
        # populate input paths
        problem_input_path = load_data.get_ds_comb_prob_path('output', dataset['path'], problem_to_process_1['path'],
                                                             problem_to_process_2['path'])
        problem_output_path = load_data.get_ds_comb_prob_path('output', dataset['path'], problem_to_process_1['path'],
                                                              problem_to_process_2['path'])
        file_name_pattern = '{}/{}_{}_{}_scaledData_*.csv'.format(problem_input_path, dataset['path'],
                                                                  problem_to_process_1['path'],
                                                                  problem_to_process_2['path'])
    else:
        # populate input paths
        problem_input_path = load_data.get_ds_prob_path('output', dataset['path'], problem_to_process_1['path'])
        problem_output_path = load_data.get_ds_prob_path('output', dataset['path'], problem_to_process_1['path'])
        file_name_pattern = '{}/{}_{}_scaledData_*.csv'.format(problem_input_path, dataset['path'],
                                                               problem_to_process_1['path'])

    if not os.path.exists(problem_output_path):
        os.makedirs(problem_output_path)

    file_paths = glob.glob(file_name_pattern)
    print("Files {}".format(file_paths))

    if len(file_paths) != 0:
        for file_path in file_paths:
            base_file_name = basename(file_path)
            if problem_to_process_2 is not None:
                output_file_name_regex = re.compile('([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)_([0-9]+)\.csv')
                _, _, _, param_name, k = output_file_name_regex.search(base_file_name).groups()
            else:
                output_file_name_regex = re.compile('([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)_([0-9]+)\.csv')
                _, _, param_name, k = output_file_name_regex.search(base_file_name).groups()

            df = pd.read_csv(file_path, header=0)
            training_x = get_x(df)
            training_y = get_y(df)
            training_x_2d = TSNE(verbose=10, random_state=config_util.random_state, learning_rate=500,
                                 n_iter=3000).fit_transform(training_x)

            training_y_T = np.array([training_y]).T
            scaled_val = np.concatenate((training_x_2d, training_y_T), axis=1)
            scaled_df = pd.DataFrame(scaled_val, columns=['tsne_0', 'tsne_1', 'label'])
            if problem_to_process_2 is not None:
                scaled_df_name = (problem_output_path + '{}_{}_{}_{}_{}_{}.csv').format(dataset['path'],
                                                                                        problem_to_process_1['path'],
                                                                                        problem_to_process_2['path'],
                                                                                        problem['path'], param_name,
                                                                                        str(k))
            else:
                scaled_df_name = (problem_output_path + '{}_{}_{}_{}_{}.csv').format(dataset['path'],
                                                                                     problem_to_process_1['path'],
                                                                                     problem['path'], param_name,
                                                                                     str(k))
            scaled_df.to_csv(scaled_df_name, index=False)
            print("Creating {} ".format(scaled_df_name))


if __name__ == '__main__':
    prob = {
        'name': 'T-SNE',
        'path': 'tsne'
    }

    if config_util.dr_enabled:
        for ds in config_util.datasets:
            if not ds['process']:
                continue
            for prob_to_process_1 in config_util.problem_to_process:
                if not prob_to_process_1['process'] or prob_to_process_1['type'] != 'Clustering':
                    continue
                for prob_to_process_2 in config_util.problem_to_process:
                    if not prob_to_process_2['process'] or prob_to_process_2['type'] != 'Dim-Red':
                        continue

                    print("================================")
                    print("Dataset:{}, Problem: {}, Problem_to_process:{},{} ".format(ds['name'], prob['name'],
                                                                                      prob_to_process_1['name'],
                                                                                      prob_to_process_2['name']))
                    tsne_experiment(ds, prob, prob_to_process_1, prob_to_process_2)

    else:
        for ds in config_util.datasets:
            if not ds['process']:
                continue
            for prob_to_process in config_util.problem_to_process:
                if not prob_to_process['process']:
                    continue
                print("================================")
                print("Dataset: {}, Problem: {}, Problem_to_process:{}".format(ds["name"], prob['name'],
                                                                               prob_to_process['name']))

                tsne_experiment(ds, prob, prob_to_process)
