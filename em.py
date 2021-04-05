import glob
import os
import re
from collections import defaultdict
from os.path import basename
from time import clock

from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, adjusted_mutual_info_score, \
    fowlkes_mallows_score, accuracy_score, homogeneity_score, adjusted_rand_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

import config_util
import load_data
from load_data import get_x, get_y, DataLoader

clusters = range(2, 20)

file_to_process = [
    {
        'name_pattern': '{}/{}_{}_scaledData_*.csv',
        'process': True
    }
]


def em_experiment(dl, problem, problem_to_process=None):
    dataset_path = dl.dataset_path
    problem_path = problem['path']
    if problem_to_process is not None:
        # populate input paths
        problem_input_path = load_data.get_ds_prob_path('output', dataset_path, problem_to_process['path'])
        problem_output_path = load_data.get_ds_comb_prob_path('output', dataset_path, problem_path,
                                                              problem_to_process['path'])

        for cur_file in file_to_process:
            if not cur_file['process']:
                continue

            file_paths = glob.glob(
                cur_file['name_pattern'].format(problem_input_path, dataset_path, problem_to_process['path']))
            print("Files {}".format(file_paths))

            if len(file_paths) == 0:
                continue

            for file_path in file_paths:
                base_file_name = basename(file_path)
                output_file_name_regex = re.compile('([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)_([0-9]+)\.csv')
                _, _, param_name, k = output_file_name_regex.search(base_file_name).groups()

                df = pd.read_csv(file_path, header=0)
                train_x = get_x(df)
    else:
        problem_output_path = load_data.get_ds_prob_path('output', dataset_path, problem_path)
        train_x = get_x(dl.get_data())

    if not os.path.exists(problem_output_path):
        os.makedirs(problem_output_path)

    information_criterion_val = []
    ll_val = []

    j = 0
    times = [0]
    for index, k in enumerate(clusters):
        gmm = mixture.GaussianMixture(n_components=k, random_state=config_util.random_state)

        start = clock()
        gmm.fit(train_x)
        elapsed = clock() - start
        times.append(times[-1] + elapsed)

        information_criterion_val.append([k, gmm.aic(train_x), gmm.bic(train_x)])

        ll_val.append([k, gmm.score(train_x)])

    information_criterion_df = pd.DataFrame(information_criterion_val,
                                            columns=['numberOfClusters', 'AIC', 'BIC']).set_index('numberOfClusters')
    ll_df = pd.DataFrame(ll_val, columns=['numberOfClusters', 'logLikelihood']).set_index('numberOfClusters')

    if problem_to_process is not None:
        information_criterion_df_name = (problem_output_path + '{}_{}_{}_{}.csv').format(dataset_path, problem_path,
                                                                                         problem_to_process['path'],
                                                                                         'informationCriterion')
        ll_df_name = (problem_output_path + '{}_{}_{}_{}.csv').format(dataset_path, problem_path,
                                                                      problem_to_process['path'],
                                                                      'logLikelihood')
    else:
        information_criterion_df_name = (problem_output_path + '{}_{}_{}.csv').format(dataset_path, problem_path,
                                                                                      'informationCriterion')
        ll_df_name = (problem_output_path + '{}_{}_{}.csv').format(dataset_path, problem_path, 'logLikelihood')

    information_criterion_df.to_csv(information_criterion_df_name)
    print("Creating {} ".format(information_criterion_df_name))

    ll_df.to_csv(ll_df_name)
    print("Creating {} ".format(ll_df_name))


def em_scale_data(dl, problem, best_param_value, problem_to_process=None):
    dataset_path = dl.dataset_path
    problem_path = problem['path']
    best_k = best_param_value
    if problem_to_process is not None:
        # populate input paths
        problem_input_path = load_data.get_ds_prob_path('output', dataset_path, problem_to_process['path'])
        problem_output_path = load_data.get_ds_comb_prob_path('output', dataset_path, problem_path,
                                                              problem_to_process['path'])

        for cur_file in file_to_process:
            if not cur_file['process']:
                continue

            file_paths = glob.glob(
                cur_file['name_pattern'].format(problem_input_path, dataset_path, problem_to_process['path']))
            print("Files {}".format(file_paths))

            if len(file_paths) == 0:
                continue

            for file_path in file_paths:
                base_file_name = basename(file_path)
                output_file_name_regex = re.compile('([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)_([0-9]+)\.csv')
                _, _, param_name, k = output_file_name_regex.search(base_file_name).groups()

                df = pd.read_csv(file_path, header=0)
                train_x = get_x(df)
    else:
        problem_output_path = load_data.get_ds_prob_path('output', dataset_path, problem_path)
        train_x = get_x(dl.get_data())

    if not os.path.exists(problem_output_path):
        os.makedirs(problem_output_path)

    col_names = ["{}_{}".format(problem_path, i) for i in range(train_x.shape[1])]
    col_names.append('label')

    gmm = mixture.GaussianMixture(n_components=best_k, random_state=config_util.random_state)

    scaled_predict_y = gmm.fit_predict(train_x)
    scaled_predict_y_t = np.array([scaled_predict_y]).T

    gmm_scaled_val = np.concatenate((train_x, scaled_predict_y_t), axis=1)
    gmm_scaled_df = pd.DataFrame(gmm_scaled_val, columns=col_names)

    if problem_to_process is not None:
        gmm_scaled_df_name = (problem_output_path + '{}_{}_{}_{}_{}.csv').format(dataset_path, problem_path,
                                                                                 problem_to_process['path'],
                                                                                 'scaledData', str(best_k))
    else:
        gmm_scaled_df_name = (problem_output_path + '{}_{}_{}_{}.csv').format(dataset_path, problem_path, 'scaledData',
                                                                              str(best_k))
    gmm_scaled_df.to_csv(gmm_scaled_df_name, index=False)
    print("Creating {} ".format(gmm_scaled_df_name))

    if problem_to_process is None:
        # nn data
        nn_data_y = get_y(dl.get_data())
        nn_data_y_t = np.array([nn_data_y]).T

        lb = LabelBinarizer()
        nn_data_scaled_x = lb.fit_transform(scaled_predict_y_t)

        col_names = ["{}_{}".format(problem_path, i) for i in range(nn_data_scaled_x.shape[1])]
        col_names.append('label')

        nn_data_val = np.concatenate((nn_data_scaled_x, nn_data_y_t), axis=1)
        nn_data_df = pd.DataFrame(nn_data_val, columns=col_names)

        nn_data_df_name = (problem_output_path + '{}_{}_{}_{}_{}.csv').format(dataset_path, problem_path, 'nn',
                                                                              'data', str(best_k))

        nn_data_df.to_csv(nn_data_df_name, index=False)
        print("Creating {} ".format(nn_data_df_name))


def em_valuate(dl, problem, best_param_value, problem_to_process=None):
    dataset_path = dl.dataset_path
    problem_path = problem['path']
    best_k = best_param_value

    if problem_to_process is not None:
        # populate input paths
        problem_input_path = load_data.get_ds_prob_path('output', dataset_path, problem_to_process['path'])
        problem_output_path = load_data.get_ds_comb_prob_path('output', dataset_path, problem_path,
                                                              problem_to_process['path'])

        for cur_file in file_to_process:
            if not cur_file['process']:
                continue

            file_paths = glob.glob(
                cur_file['name_pattern'].format(problem_input_path, dataset_path, problem_to_process['path']))
            print("Files {}".format(file_paths))

            if len(file_paths) == 0:
                continue

            for file_path in file_paths:
                base_file_name = basename(file_path)
                output_file_name_regex = re.compile('([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)_([0-9]+)\.csv')
                _, _, param_name, k = output_file_name_regex.search(base_file_name).groups()

                df = pd.read_csv(file_path, header=0)
                train_x = get_x(df)
                train_y = get_y(df)
    else:
        problem_output_path = load_data.get_ds_prob_path('output', dataset_path, problem_path)
        # get data
        train_x = get_x(dl.get_data())
        train_y = get_y(dl.get_data())
        val_0_x = get_x(dl.get_validation_data_0())
        val_1_x = get_x(dl.get_validation_data_1())

    if not os.path.exists(problem_output_path):
        os.makedirs(problem_output_path)

    # gmm
    gmm = mixture.GaussianMixture(n_components=best_k, random_state=config_util.random_state)
    start = clock()
    scaled_predict_y = gmm.fit_predict(train_x)
    elapsed = clock() - start

    val_cores = []
    # internal: davies_bouldin_score
    dbs_train = davies_bouldin_score(X=train_x, labels=scaled_predict_y)

    # external: adjusted_mutual_info_score
    ami_train = adjusted_mutual_info_score(train_y, scaled_predict_y)
    # adjusted_rand_score
    ar_train = adjusted_rand_score(train_y, scaled_predict_y)
    # external: homogeneity_score
    homogeneity_train = homogeneity_score(train_y, scaled_predict_y)

    if problem_to_process is not None:

        val_cores.append([dbs_train, ami_train, ar_train, homogeneity_train, elapsed])

        val_cores_df = pd.DataFrame(val_cores,
                                    columns=['davies_bouldin_score',
                                             'adjusted_mutual_info_score_ex',
                                             'adjusted_rand_score',
                                             'homogeneity_score_ex',
                                             'fit_time'])
        val_cores_df_name = (problem_output_path + '{}_{}_{}_{}_{}.csv').format(dataset_path, problem_path,
                                                                                problem_to_process['path'], 'scores',
                                                                                str(best_k))

    else:
        # twin-sample validation
        gmm_0 = mixture.GaussianMixture(n_components=best_k, random_state=config_util.random_state)
        scaled_predict_y_0 = gmm_0.fit_predict(val_0_x)

        gmm_1 = mixture.GaussianMixture(n_components=best_k, random_state=config_util.random_state)
        scaled_predict_y_1 = gmm_1.fit_predict(val_1_x)

        # twin-sample validation: adjusted_mutual_info_score
        adjusted_mutual_info_score_twin_val = adjusted_mutual_info_score(scaled_predict_y_0, scaled_predict_y_1)

        val_cores.append(
            [dbs_train, ami_train, ar_train, homogeneity_train, adjusted_mutual_info_score_twin_val, elapsed])

        val_cores_df = pd.DataFrame(val_cores,
                                    columns=['davies_bouldin_score',
                                             'adjusted_mutual_info_score_ex',
                                             'adjusted_rand_score',
                                             'homogeneity_score_ex',
                                             'adjusted_mutual_info_score_twin_val', 'fit_time'])
        val_cores_df_name = (problem_output_path + '{}_{}_{}_{}.csv').format(dataset_path, problem_path, 'scores',
                                                                             str(best_k))

    val_cores_df.to_csv(val_cores_df_name, index=False)
    print("Creating {} ".format(val_cores_df_name))


if __name__ == '__main__':
    prob = {
        'name': 'EM',
        'path': 'em'
    }
    if config_util.dr_enabled:
        for ds in config_util.datasets:
            if not ds['process']:
                continue
            data = DataLoader(ds)
            data.load_and_process()
            for prob_to_process in config_util.problem_to_process:
                if not prob_to_process['process'] or prob_to_process['type'] != 'Dim-Red':
                    continue

                print("================================")
                print("Dataset:{}, Problem: {}, Problem_to_process:{}, ".format(ds['name'], prob['name'],
                                                                                prob_to_process['name']))
                em_experiment(dl=data, problem=prob, problem_to_process=prob_to_process)
                best_param = prob_to_process["{}_{}".format(ds['path'], prob['path'])]
                print("Best Param:{}".format(best_param))
                em_scale_data(dl=data, problem=prob, best_param_value=best_param, problem_to_process=prob_to_process)
                em_valuate(dl=data, problem=prob, best_param_value=best_param, problem_to_process=prob_to_process)

    else:
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
            print("Best Param:{}".format(best_param))

            em_experiment(dl=data, problem=prob)
            em_scale_data(dl=data, problem=prob, best_param_value=best_param)
            em_valuate(dl=data, problem=prob, best_param_value=best_param)
