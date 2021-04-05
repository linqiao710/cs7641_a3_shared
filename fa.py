import os
from collections import defaultdict
from time import clock

from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA, SparsePCA, FactorAnalysis
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, adjusted_mutual_info_score, \
    fowlkes_mallows_score
import pandas as pd
import numpy as np
import scipy.sparse as sps
from scipy.linalg import pinv

from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

import config_util
import load_data
from load_data import get_x, get_y, DataLoader


def fa_experiment(dl, problem):
    dataset_path = dl.dataset_path
    problem_path = problem['path']
    problem_output_path = load_data.get_ds_prob_path('output', dataset_path, problem_path)

    if not os.path.exists(problem_output_path):
        os.makedirs(problem_output_path)

    train_x = get_x(dl.get_data())

    dims = np.arange(2, train_x.shape[1] + 1)

    reconstructionError_val = []
    cv_val = []

    for k in dims:
        fa = FactorAnalysis(n_components=k, random_state=config_util.random_state)
        fa.fit(train_x)
        reconstructionError_val.append([k, load_data.reconstruction_error(fa, train_x)])
        cv_val.append([k, np.mean(cross_val_score(fa, train_x))])

    reconstructionError_df = pd.DataFrame(reconstructionError_val,
                                          columns=['numberOfComponents', 'reconstructionError']).set_index(
        'numberOfComponents')
    reconstructionError_df_name = (problem_output_path + '{}_{}_{}.csv').format(dataset_path, problem_path,
                                                                                'reconstructionError')
    reconstructionError_df.to_csv(reconstructionError_df_name)
    print("Creating {} ".format(reconstructionError_df_name))

    cv_df = pd.DataFrame(cv_val, columns=['numberOfComponents', 'crossValidationScores']).set_index(
        'numberOfComponents')
    cv_df_name = (problem_output_path + '{}_{}_{}.csv').format(dataset_path, problem_path,
                                                               'crossValidationScores')
    cv_df.to_csv(cv_df_name)
    print("Creating {} ".format(cv_df_name))


def fa_scale_data(dl, problem, best_param_val):
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

    fa = FactorAnalysis(n_components=best_k, random_state=config_util.random_state)
    fa_scaled_x = fa.fit_transform(train_x)

    train_y_t = np.array([train_y]).T
    fa_scaled_val = np.concatenate((fa_scaled_x, train_y_t), axis=1)
    fa_scaled_df = pd.DataFrame(fa_scaled_val, columns=col_names)
    fa_scaled_df.to_csv(
        (problem_output_path + '{}_{}_{}_{}.csv').format(dataset_path, problem_path, 'scaledData', str(best_k)),
        index=False)

    print("Creating {} ".format(
        (problem_output_path + '{}_{}_{}_{}.csv').format(dataset_path, problem_path, 'scaledData', str(best_k))))

    # nn data
    nn_data_df_name = (problem_output_path + '{}_{}_{}_{}_{}.csv').format(dataset_path, problem_path, 'nn',
                                                                          'data', str(best_k))

    fa_scaled_df.to_csv(nn_data_df_name, index=False)
    print("Creating {} ".format(nn_data_df_name))


def fa_valuate(dl, problem, best_param_value):
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
    fa = FactorAnalysis(n_components=best_k, random_state=config_util.random_state)
    start = clock()
    fa.fit(train_x)
    elapsed = clock() - start

    val_cores = []

    reconstruction_error = load_data.reconstruction_error(fa, train_x)

    val_cores.append([reconstruction_error, elapsed])
    val_cores_df = pd.DataFrame(val_cores, columns=['reconstruction_error', 'fit_time'])
    val_cores_df.to_csv(
        (problem_output_path + '{}_{}_{}_{}.csv').format(dataset_path, problem_path, 'scores', str(best_k)),
        index=False)
    print("Creating {} ".format(
        (problem_output_path + '{}_{}_{}_{}.csv').format(dataset_path, problem_path, 'scores', str(best_k))))


if __name__ == '__main__':
    prob = {
        'name': 'Factor-Analysis',
        'path': 'fa'
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

        fa_experiment(data, prob)
        fa_scale_data(data, prob, best_param)
        fa_valuate(data, prob, best_param)
