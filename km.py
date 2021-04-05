import glob
import os
import re
from os.path import basename
from time import clock
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_mutual_info_score, f1_score, \
    davies_bouldin_score, fowlkes_mallows_score, jaccard_score, homogeneity_score, adjusted_rand_score
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer

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


def k_means_experiment(dl, problem, problem_to_process=None):
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

    inertia_val = []
    sil_score_mean = []
    sil_score_coefficient = []

    j = 0

    for index, k in enumerate(clusters):
        km = KMeans(n_clusters=k, random_state=config_util.random_state)
        km.fit(train_x)
        km_labels = km.predict(train_x)

        inertia_val.append([k, km.inertia_])
        sil_score_mean.append([k, silhouette_score(X=train_x, labels=km_labels)])
        sil_score_sample = silhouette_samples(X=train_x, labels=km_labels)

        for i, x in enumerate(sil_score_sample):
            sil_score_coefficient.append([k, round(x, 6), sil_score_mean[index][1], km_labels[i]])
            j += 1

    inertia_df = pd.DataFrame(inertia_val, columns=['numberOfClusters', 'inertia']).set_index('numberOfClusters')
    sil_score_mean_df = pd.DataFrame(sil_score_mean, columns=['numberOfClusters', 'silhouetteScore']).set_index(
        'numberOfClusters')
    sil_score_coefficient_df = pd.DataFrame(sil_score_coefficient,
                                            columns=['numberOfClusters', 'score', 'score_mean', 'label']).set_index(
        'numberOfClusters')

    if problem_to_process is not None:
        inertia_df_name = (problem_output_path + '{}_{}_{}_{}.csv').format(dataset_path, problem_path,
                                                                           problem_to_process['path'], 'inertia')
        sil_score_mean_df_name = (problem_output_path + '{}_{}_{}_{}.csv').format(dataset_path, problem_path,
                                                                                  problem_to_process['path'],
                                                                                  'silhouetteScore')
        sil_score_coefficient_df_name = (problem_output_path + '{}_{}_{}_{}.csv').format(dataset_path, problem_path,
                                                                                         problem_to_process['path'],
                                                                                         'silhouetteCoefficient')
    else:
        inertia_df_name = (problem_output_path + '{}_{}_{}.csv').format(dataset_path, problem_path, 'inertia')
        sil_score_mean_df_name = (problem_output_path + '{}_{}_{}.csv').format(dataset_path, problem_path,
                                                                               'silhouetteScore')
        sil_score_coefficient_df_name = (problem_output_path + '{}_{}_{}.csv').format(dataset_path, problem_path,
                                                                                      'silhouetteCoefficient')

    inertia_df.to_csv(inertia_df_name)
    print("Creating {} ".format(inertia_df_name))

    sil_score_mean_df.to_csv(sil_score_mean_df_name)
    print("Creating {} ".format(sil_score_mean_df_name))

    sil_score_coefficient_df.to_csv(sil_score_coefficient_df_name)
    print("Creating {} ".format(sil_score_coefficient_df_name))


def k_means_scale_data(dl, problem, best_param_value, problem_to_process=None):
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

    km = KMeans(n_clusters=best_k, random_state=config_util.random_state)
    scaled_predict_y = km.fit_predict(train_x)
    scaled_predict_y_t = np.array([scaled_predict_y]).T
    km_scaled_val = np.concatenate((train_x, scaled_predict_y_t), axis=1)
    km_scaled_df = pd.DataFrame(km_scaled_val, columns=col_names)

    if problem_to_process is not None:
        km_scaled_df_name = (problem_output_path + '{}_{}_{}_{}_{}.csv').format(dataset_path, problem_path,
                                                                                problem_to_process['path'],
                                                                                'scaledData', str(best_k))
    else:
        km_scaled_df_name = (problem_output_path + '{}_{}_{}_{}.csv').format(dataset_path, problem_path, 'scaledData',
                                                                             str(best_k))

    km_scaled_df.to_csv(km_scaled_df_name, index=False)
    print("Creating {} ".format(km_scaled_df_name))

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


def k_means_valuate(dl, problem, best_param_value, problem_to_process=None):
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

    # k-means
    km = KMeans(n_clusters=best_k, random_state=config_util.random_state)
    start = clock()
    scaled_predict_y = km.fit_predict(train_x)
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
        km_0 = KMeans(n_clusters=best_k, random_state=config_util.random_state)
        scaled_predict_y_0 = km_0.fit_predict(val_0_x)

        km_1 = KMeans(n_clusters=best_k, random_state=config_util.random_state)
        scaled_predict_y_1 = km_1.fit_predict(val_1_x)

        # twin-sample validation: adjusted_mutual_info_score
        adjusted_mutual_info_score_twin_val = adjusted_mutual_info_score(scaled_predict_y_0, scaled_predict_y_1)

        val_cores.append([dbs_train, ami_train, ar_train, homogeneity_train, adjusted_mutual_info_score_twin_val, elapsed])

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
        'name': 'K-Means',
        'path': 'km'
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
                k_means_experiment(dl=data, problem=prob, problem_to_process=prob_to_process)
                best_param = prob_to_process["{}_{}".format(ds['path'], prob['path'])]
                print("Best Param:{}".format(best_param))
                k_means_scale_data(dl=data, problem=prob, best_param_value=best_param,
                                   problem_to_process=prob_to_process)
                k_means_valuate(data, prob, best_param, problem_to_process=prob_to_process)
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
            k_means_experiment(dl=data, problem=prob)
            k_means_scale_data(dl=data, problem=prob, best_param_value=best_param)
            k_means_valuate(dl=data, problem=prob, best_param_value=best_param)
