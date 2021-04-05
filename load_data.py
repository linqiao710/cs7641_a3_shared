import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.tree._criterion import MSE

import config_util
import scipy.sparse as sps
from scipy.linalg import pinv


def reconstruction_error(projections, x):
    w = projections.components_
    if sps.issparse(w):
        w = w.todense()
    p = pinv(w)
    reconstructed = ((p @ w) @ (x.T)).T  # Unproject projected data
    return mean_squared_error(x, reconstructed)


def get_reconstructed_x(projections, x):
    w = projections.components_
    if sps.issparse(w):
        w = w.todense()
    p = pinv(w)
    reconstructed = ((p @ w) @ (x.T)).T  # Unproject projected data
    return reconstructed


def get_ds_prob_path(in_out_type, ds_name, prob_to_process_name):
    return config_util.path_pattern.format(in_out_type, ds_name, prob_to_process_name)


def get_ds_comb_prob_path(in_out_type, ds_name, prob, prob_to_process_name):
    return config_util.com_prob_path_pattern.format(in_out_type, ds_name, prob, prob_to_process_name)


def get_x(df):
    return np.array(df.iloc[:, 0:-1])


def get_y(df):
    return np.array(df.iloc[:, -1])


class DataLoader:
    def __init__(self, dataset, problem=None, problem_to_process_1=None):
        if problem is not None and problem_to_process_1 is not None:
            self.dataset_name = dataset['name']
            self.dataset_path = dataset['path']

            input_path = get_ds_prob_path('output', dataset['path'], problem_to_process_1['path'])
            self.input_train_path = '{}/{}_{}_nn_data_train.csv'.format(input_path, dataset['path'],
                                                                        problem_to_process_1['path'])
            self.input_test_path = '{}/{}_{}_nn_data_test.csv'.format(input_path, dataset['path'],
                                                                      problem_to_process_1['path'])

            self.problem_to_process_name = problem_to_process_1['name']
            self.problem_to_process_path = problem_to_process_1['path']

            self.random_state = config_util.random_state
        else:
            self.dataset_name = dataset['name']
            self.dataset_path = dataset['path']
            self.problem_name = 'Original'
            self.problem_path = 'original'
            self.random_state = config_util.random_state
            self.input_train_path = None
            self.input_test_path = None

        self.train_data = None
        self.test_data = None
        self.validation_data_0 = None
        self.validation_data_1 = None
        self.data = None

        self.load_and_process()

    def load_and_process(self):
        if self.input_train_path is not None and self.input_test_path is not None:
            self.train_data = pd.read_csv(self.input_train_path, header=0)
            self.test_data = pd.read_csv(self.input_test_path, header=0)
        else:
            # populate input paths
            ds_input_path = get_ds_prob_path('output', self.dataset_path, self.problem_path)

            train_input_path = os.path.join(ds_input_path, config_util.train_data_file.format(self.dataset_name))
            test_input_path = os.path.join(ds_input_path, config_util.test_data_file.format(self.dataset_name))
            val_input_path_0 = os.path.join(ds_input_path, config_util.val_data_file_0.format(self.dataset_name))
            val_input_path_1 = os.path.join(ds_input_path, config_util.val_data_file_1.format(self.dataset_name))
            data_file_path = os.path.join(ds_input_path, "{}_original_scaledData_1.csv".format(self.dataset_path))

            self.train_data = pd.read_csv(train_input_path, header=0)
            self.test_data = pd.read_csv(test_input_path, header=0)
            self.validation_data_0 = pd.read_csv(val_input_path_0, header=0)
            self.validation_data_1 = pd.read_csv(val_input_path_1, header=0)
            self.validation_data_1 = self.validation_data_1[:-1]

            self.data = pd.read_csv(data_file_path, header=0)

    def get_data(self):
        return self.data

    def get_test_data(self):
        return self.test_data

    def get_train_data(self):
        return self.train_data

    def get_validation_data_0(self):
        return self.validation_data_0

    def get_validation_data_1(self):
        return self.validation_data_1


if __name__ == '__main__':
    for ds in config_util.datasets:
        data = DataLoader(ds)
