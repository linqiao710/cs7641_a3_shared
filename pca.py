import os
from time import clock

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelBinarizer

import config_util
import load_data
from load_data import DataLoader, get_x, get_y


def pca_experiment(dl, problem):
    dataset_path = dl.dataset_path
    problem_path = problem['path']
    problem_output_path = load_data.get_ds_prob_path('output', dataset_path, problem_path)

    if not os.path.exists(problem_output_path):
        os.makedirs(problem_output_path)

    train_x = get_x(dl.get_data())
    dim = train_x.shape[1]

    pca = PCA(n_components=dim, random_state=config_util.random_state)
    pca.fit(train_x)

    scree_df = pd.DataFrame(data=pca.explained_variance_, index=range(1, pca.explained_variance_.shape[0] + 1),
                            columns=['eigenValue'])
    scree_df.index.name = 'numberOfComponents'
    scree_df.to_csv(
        (problem_output_path + '{}_{}_{}.csv').format(dataset_path, problem_path, 'eigenValue'))
    print("Creating {} ".format(
        (problem_output_path + '{}_{}_{}.csv').format(dataset_path, problem_path, 'eigenValue')))

    cum_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    cum_scree_df = pd.DataFrame(data=cum_variance_ratio, index=range(1, pca.explained_variance_.shape[0] + 1),
                                columns=['cumulativeVarianceRatio'])
    cum_scree_df.index.name = 'numberOfComponents'
    cum_scree_df.to_csv(
        (problem_output_path + '{}_{}_{}.csv').format(dataset_path, problem_path, 'cumulativeVarianceRatio'))
    print("Creating {} ".format(
        (problem_output_path + '{}_{}_{}.csv').format(dataset_path, problem_path, 'cumulativeVarianceRatio')))


def pca_scale_data(dl, problem, best_param_value):
    dataset_path = dl.dataset_path
    problem_path = problem['path']
    problem_output_path = load_data.get_ds_prob_path('output', dataset_path, problem_path)

    if not os.path.exists(problem_output_path):
        os.makedirs(problem_output_path)

    # best hyper-parameter
    best_k = best_param_value
    train_x = get_x(dl.get_data())
    train_y = get_y(dl.get_data())

    col_names = ["{}_{}".format(problem_path, i) for i in range(best_k)]
    col_names.append('label')

    pca = PCA(n_components=best_k, random_state=config_util.random_state)
    pca_scaled_x = pca.fit_transform(train_x)

    train_y_t = np.array([train_y]).T
    pca_scaled_val = np.concatenate((pca_scaled_x, train_y_t), axis=1)

    pca_scaled_df = pd.DataFrame(pca_scaled_val, columns=col_names)
    pca_scaled_df_name = (problem_output_path + '{}_{}_{}_{}.csv').format(dataset_path, problem_path, 'scaledData',
                                                                          str(best_k))
    pca_scaled_df.to_csv(pca_scaled_df_name, index=False)

    print("Creating {} ".format(pca_scaled_df_name))

    scree_df = pd.DataFrame(data=pca.explained_variance_, index=range(1, pca.explained_variance_.shape[0] + 1),
                            columns=['eigenValue'])
    scree_df.index.name = 'numberOfComponents'
    scree_df_name = (problem_output_path + '{}_{}_{}.csv').format(dataset_path, problem_path, 'eigenValues')
    scree_df.to_csv(scree_df_name)
    print("Creating {} ".format(scree_df_name))

    # nn data
    nn_data_df_name = (problem_output_path + '{}_{}_{}_{}_{}.csv').format(dataset_path, problem_path, 'nn',
                                                                          'data', str(best_k))

    pca_scaled_df.to_csv(nn_data_df_name, index=False)
    print("Creating {} ".format(nn_data_df_name))


def pca_valuate(dl, problem, best_param_value):
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

    # PCA
    pca = PCA(n_components=best_k, random_state=config_util.random_state)
    start = clock()
    pca.fit(train_x)
    elapsed = clock() - start

    val_cores = []

    reconstruction_error = load_data.reconstruction_error(pca, train_x)

    val_cores.append([reconstruction_error, elapsed])
    val_cores_df = pd.DataFrame(val_cores, columns=['reconstruction_error', 'fit_time'])
    val_cores_df.to_csv(
        (problem_output_path + '{}_{}_{}_{}.csv').format(dataset_path, problem_path, 'scores', str(best_k)),
        index=False)
    print("Creating {} ".format(
        (problem_output_path + '{}_{}_{}_{}.csv').format(dataset_path, problem_path, 'scores', str(best_k))))


if __name__ == '__main__':
    prob = {
        'name': 'PCA',
        'path': 'pca'
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
        print("Best Param:{}".format(best_param))
        pca_experiment(data, prob)
        pca_scale_data(data, prob, best_param)
        pca_valuate(data, prob, best_param)
