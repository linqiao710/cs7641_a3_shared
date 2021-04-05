import os
from time import clock

import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.linalg import pinv

from sklearn.decomposition import PCA, FastICA
from sklearn.model_selection import cross_val_score

import config_util
import load_data
from load_data import DataLoader, get_x, get_y


def ica_experiment(dl, problem):
    dataset_path = dl.dataset_path
    problem_path = problem['path']
    problem_output_path = load_data.get_ds_prob_path('output', dataset_path, problem_path)

    if not os.path.exists(problem_output_path):
        os.makedirs(problem_output_path)

    train_x = get_x(dl.get_data())
    dims = np.arange(2, train_x.shape[1] + 1)

    kurt_mean = []

    for k in dims:
        ica = FastICA(n_components=k, random_state=config_util.random_state)
        ica.fit(train_x)

        components_df = pd.DataFrame(ica.components_)
        kurt_df = components_df.kurt(axis=1)
        kurt_mean.append([k, kurt_df.abs().mean()])

    kurt_mean_df = pd.DataFrame(kurt_mean, columns=['numberOfComponents', 'kurtosis']).set_index(
        'numberOfComponents')
    kurt_mean_df.to_csv((problem_output_path + '{}_{}_{}.csv').format(dataset_path, problem_path, 'kurtosis'))
    print("Creating {} ".format(
        (problem_output_path + '{}_{}_{}.csv').format(dataset_path, problem_path, 'kurtosis')))


def ica_scale_data(dl, problem, best_param_val):
    dataset_path = dl.dataset_path
    problem_path = problem['path']
    problem_output_path = load_data.get_ds_prob_path('output', dataset_path, problem_path)

    if not os.path.exists(problem_output_path):
        os.makedirs(problem_output_path)

    # best hyper-parameter
    best_k = best_param_val
    train_x = get_x(dl.get_data())
    train_y = get_y(dl.get_data())

    col_names = ["{}_{}".format(problem_path, i) for i in range(best_k)]
    col_names.append('label')

    ica = FastICA(n_components=best_k, random_state=config_util.random_state)
    ica_scaled_x = ica.fit_transform(train_x)

    train_y_t = np.array([train_y]).T
    ica_scaled_val = np.concatenate((ica_scaled_x, train_y_t), axis=1)
    ica_scaled_df = pd.DataFrame(ica_scaled_val, columns=col_names)

    ica_scaled_df_name = (problem_output_path + '{}_{}_{}_{}.csv').format(dataset_path, problem_path, 'scaledData', str(best_k))
    ica_scaled_df.to_csv(ica_scaled_df_name, index=False)
    print("Creating {} ".format(ica_scaled_df_name))

    components_df = pd.DataFrame(ica.components_)
    kurt_se = components_df.kurt(axis=1)
    kurt_se = kurt_se.abs()

    kurt_df = pd.DataFrame(kurt_se.values, columns=['kurtosis'], index=range(1, best_param_val + 1))
    kurt_df.index.name = 'numberOfComponents'
    kurt_df_name = (problem_output_path + '{}_{}_{}.csv').format(dataset_path, problem_path, 'kurtoses')
    kurt_df.to_csv(kurt_df_name)
    print("Creating {} ".format(kurt_df_name))

    # nn data
    nn_data_df_name = (problem_output_path + '{}_{}_{}_{}_{}.csv').format(dataset_path, problem_path, 'nn',
                                                                          'data', str(best_k))

    ica_scaled_df.to_csv(nn_data_df_name, index=False)
    print("Creating {} ".format(nn_data_df_name))


def ica_valuate(dl, problem, best_param_value):
    dataset_path = dl.dataset_path
    problem_path = problem['path']
    problem_output_path = load_data.get_ds_prob_path('output', dataset_path, problem_path)

    if not os.path.exists(problem_output_path):
        os.makedirs(problem_output_path)

    # best hyper-parameter
    best_k = best_param_value

    # get data
    train_x = get_x(dl.get_data())
    train_y = get_y(dl.get_data())

    # ICA
    ica = FastICA(n_components=best_k, random_state=config_util.random_state)
    start = clock()
    ica.fit(train_x)
    elapsed = clock() - start

    val_cores = []

    reconstruction_error = load_data.reconstruction_error(ica, train_x)

    val_cores.append([reconstruction_error, elapsed])
    val_cores_df = pd.DataFrame(val_cores, columns=['reconstruction_error', 'fit_time'])
    val_cores_df.to_csv(
        (problem_output_path + '{}_{}_{}_{}.csv').format(dataset_path, problem_path, 'scores', str(best_k)),
        index=False)
    print("Creating {} ".format(
        (problem_output_path + '{}_{}_{}_{}.csv').format(dataset_path, problem_path, 'scores', str(best_k))))


if __name__ == '__main__':
    prob = {
        'name': 'ICA',
        'path': 'ica'
    }
    for ds in config_util.datasets:
        if not ds['process']:
            continue
        data = DataLoader(ds)
        data.load_and_process()
        print("================================")
        print("Dataset: {}, Problem: {} ".format(ds["name"], prob['name']))

        for prob_to_process in config_util.problem_to_process:
            if prob_to_process['path'] == prob['path']:
                prob = prob_to_process
                break

        best_param = prob[ds['path']]

        ica_experiment(data, prob)
        ica_scale_data(data, prob, best_param)
        ica_valuate(data, prob, best_param)
