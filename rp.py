import os
from collections import defaultdict
from itertools import product
from time import clock

import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.linalg import pinv

from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.random_projection import SparseRandomProjection

import config_util
import load_data
from load_data import DataLoader, get_x, get_y

seeds = np.arange(1, 1001)


def rp_experiment(dl, problem):
    dataset_path = dl.dataset_path
    problem_path = problem['path']
    problem_output_path = load_data.get_ds_prob_path('output', dataset_path, problem_path)

    if not os.path.exists(problem_output_path):
        os.makedirs(problem_output_path)

    train_x = get_x(dl.get_data())
    dims = np.arange(2, train_x.shape[1] + 1)

    col_names = []
    for i in range(len(seeds)):
        col_names.append('trial_{}'.format(str(i + 1)))

    reconstruction_error_val = defaultdict(dict)
    for seed, dim in product(seeds, dims):
        rp = SparseRandomProjection(random_state=seed, n_components=dim)
        rp.fit(train_x)
        reconstruction_error_val[dim][seed] = load_data.reconstruction_error(rp, train_x)

    reconstruction_error_df = pd.DataFrame(reconstruction_error_val).T
    reconstruction_error_df = pd.DataFrame(reconstruction_error_df.values, index=dims, columns=col_names)
    reconstruction_error_df.index.name = 'numberOfComponents'
    reconstruction_error_df['mean'] = reconstruction_error_df.mean(axis=1)
    reconstruction_error_df['std'] = reconstruction_error_df.std(axis=1)

    reconstruction_error_df.to_csv(
        (problem_output_path + '{}_{}_{}.csv').format(dataset_path, problem_path, 'reconstructionErrors'))
    print("Creating {} ".format(
        (problem_output_path + '{}_{}_{}.csv').format(dataset_path, problem_path, 'reconstructionErrors')))


def rp_scale_data(dl, problem, best_param_value):
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

    col_names = ["{}_{}".format(problem_path, i) for i in range(best_k)]
    col_names.append('label')

    rp = SparseRandomProjection(random_state=config_util.random_state, n_components=best_k)
    rp_scaled_x = rp.fit_transform(train_x)

    train_y_t = np.array([train_y]).T
    rp_scaled_val = np.concatenate((rp_scaled_x, train_y_t), axis=1)
    rp_scaled_df = pd.DataFrame(rp_scaled_val, columns=col_names)
    rp_scaled_df_name = (problem_output_path + '{}_{}_{}_{}.csv').format(dataset_path, problem_path, 'scaledData',
                                                                         str(best_k))
    rp_scaled_df.to_csv(rp_scaled_df_name, index=False)
    print("Creating {} ".format(rp_scaled_df_name))

    reconstruction_error_val = []
    for seed in seeds:
        rp = SparseRandomProjection(random_state=seed, n_components=best_k)
        rp.fit(train_x)
        reconstruction_error_val.append(load_data.reconstruction_error(rp, train_x))
    reconstruction_error_df = pd.DataFrame(reconstruction_error_val, columns=['reconstructionErrors'])
    reconstruction_error_df_name = (problem_output_path + '{}_{}_{}_{}.csv').format(dataset_path, problem_path,
                                                                                    'reconstructionErrors',
                                                                                    str(best_k))
    reconstruction_error_df.to_csv(reconstruction_error_df_name, index=False)
    print("Creating {} ".format(reconstruction_error_df_name))

    # nn data
    nn_data_df_name = (problem_output_path + '{}_{}_{}_{}_{}.csv').format(dataset_path, problem_path, 'nn',
                                                                          'data', str(best_k))

    rp_scaled_df.to_csv(nn_data_df_name, index=False)
    print("Creating {} ".format(nn_data_df_name))


def rp_valuate(dl, problem, best_param_value):
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

    # RP
    rp = SparseRandomProjection(random_state=config_util.random_state, n_components=best_k)
    start = clock()
    rp.fit(train_x)
    elapsed = clock() - start

    val_cores = []

    reconstruction_error = load_data.reconstruction_error(rp, train_x)

    val_cores.append([reconstruction_error, elapsed])
    val_cores_df = pd.DataFrame(val_cores, columns=['reconstruction_error', 'fit_time'])
    val_cores_df.to_csv(
        (problem_output_path + '{}_{}_{}_{}.csv').format(dataset_path, problem_path, 'scores', str(best_k)),
        index=False)
    print("Creating {} ".format(
        (problem_output_path + '{}_{}_{}_{}.csv').format(dataset_path, problem_path, 'scores', str(best_k))))


if __name__ == '__main__':
    prob = {
        'name': 'Random-Projection',
        'path': 'rp'
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

        rp_experiment(data, prob)
        rp_scale_data(data, prob, best_param)
        rp_valuate(data, prob, best_param)
