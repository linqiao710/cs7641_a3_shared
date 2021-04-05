import os
from collections import Counter

import pandas as pd
import numpy as np
from sklearn import preprocessing, utils
import sklearn.model_selection as ms
from scipy.sparse import isspmatrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

import config_util
import load_data
import nn


def is_balanced(seq):
    n = len(seq)
    classes = [(clas, float(count)) for clas, count in Counter(seq).items()]
    k = len(classes)

    H = -sum([(count / n) * np.log((count / n)) for clas, count in classes])
    return H / np.log(k) > 0.75


class DataPreprocessor:
    def __init__(self, dataset, problem=None, problem_to_process_1=None):
        if problem is not None and problem_to_process_1 is not None:
            best_k = problem_to_process_1[dataset['path']]
            self.dataset_name = dataset['name']
            self.dataset_path = dataset['path']

            input_path = load_data.get_ds_prob_path('output', dataset['path'], problem_to_process_1['path'])
            self.input_path = '{}/{}_{}_nn_data_{}.csv'.format(input_path, dataset['path'],
                                                               problem_to_process_1['path'], best_k)
            self.output_path = load_data.get_ds_prob_path('output', dataset['path'], problem_to_process_1['path'])

            self.problem_to_process_name = problem_to_process_1['name']
            self.problem_to_process_path = problem_to_process_1['path']

            self.random_state = config_util.random_state
        else:
            self.dataset_name = dataset['name']
            self.dataset_path = dataset['path']
            self.input_file = dataset['input_file']
            self.input_path = os.path.join('data', dataset['path'], self.input_file)
            self.output_path = load_data.get_ds_prob_path('output', dataset['path'], 'original')
            self.random_state = config_util.random_state

        self.features = None
        self.classes = None
        self.testing_x = None
        self.testing_y = None
        self.training_x = None
        self.training_y = None
        self.binary = False
        self.balanced = False
        self._data = pd.DataFrame()

    def _load_data(self):
        self._data = pd.read_csv(self.input_path, header=0)

    def load_and_process(self, verbose=True):
        # read from file
        self._load_data()
        print("Processing {} Path: {}, Dimensions: {}".format(self.dataset_name, self.input_path, self._data.shape))
        if verbose:
            old_max_rows = pd.options.display.max_rows
            pd.options.display.max_rows = 10
            print("Before Shuffle")
            print("Data Sample:\n{}".format(self._data))
            pd.options.display.max_rows = old_max_rows

        self.shuffle_df(3)

        if verbose:
            old_max_rows = pd.options.display.max_rows
            pd.options.display.max_rows = 10
            print("After Shuffle")
            print("Data Sample:\n{}".format(self._data))
            pd.options.display.max_rows = old_max_rows

        # set features
        self.get_features()
        # set labels
        self.get_classes()

    def shuffle_df(self, times):
        for i in range(times):
            self._data = shuffle(self._data, random_state=self.random_state)

    def get_features(self):
        if self.features is None:
            print("Pulling features")
            self.features = np.array(self._data.iloc[:, 0:-1])

        return self.features

    def get_classes(self):
        if self.classes is None:
            print("Pulling classes")
            self.classes = np.array(self._data.iloc[:, -1])

        return self.classes

    def split_test_train(self, test_size, val_size):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # split train and test sets
        train_x, test_x, train_y, test_y = ms.train_test_split(self.features, self.classes,
                                                               test_size=test_size,
                                                               random_state=self.random_state,
                                                               stratify=self.classes)

        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)

        if config_util.nn_enabled:
            col_names = ["{}_{}".format(self.problem_to_process_path, i) for i in range(train_x.shape[1])]
            col_names.append('label')

            train_output_path = os.path.join(self.output_path, config_util.nn_train_data_file.format(self.dataset_path,
                                                                                                     self.problem_to_process_path))
            test_output_path = os.path.join(self.output_path, config_util.nn_test_data_file.format(self.dataset_path,
                                                                                                   self.problem_to_process_path))
        else:
            col_names = ["{}_{}".format(self.dataset_name, i) for i in range(train_x.shape[1])]
            col_names.append('label')

            train_output_path = os.path.join(self.output_path, config_util.train_data_file.format(self.dataset_path))
            test_output_path = os.path.join(self.output_path, config_util.test_data_file.format(self.dataset_path))

        train_df = pd.concat([pd.DataFrame(train_x), pd.DataFrame(train_y)], axis=1)
        test_df = pd.concat([pd.DataFrame(test_x), pd.DataFrame(test_y)], axis=1)
        print("train dimension: {}".format(train_df.shape))
        print("test dimension: {}".format(test_df.shape))

        train_df.columns = col_names
        test_df.columns = col_names

        train_df.to_csv(train_output_path, index=False)
        test_df.to_csv(test_output_path, index=False)

        print("Generating: {}".format(train_output_path))
        print("Generating: {}".format(test_output_path))

    def scale_data(self, val_size):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        data_x = load_data.get_x(self._data)
        data_y = load_data.get_y(self._data)

        scaler = StandardScaler()
        data_x = scaler.fit_transform(data_x)

        col_names = ["{}_{}".format(self.dataset_name, i) for i in range(data_x.shape[1])]
        col_names.append('label')

        data_df = pd.concat([pd.DataFrame(data_x), pd.DataFrame(data_y)], axis=1)
        data_df.columns = col_names

        print("data_df dimension: {}".format(data_df.shape))

        data_df_name = "{}_original_scaledData_1.csv".format(self.dataset_path)
        data_output_path = os.path.join(self.output_path, data_df_name)

        data_df.to_csv(data_output_path, index=False)

        print("Generating: {}".format(data_output_path))

        # split twin validation sets
        val_x_0, val_x_1, val_y_0, val_y_1 = ms.train_test_split(data_x, data_y,
                                                                 test_size=val_size,
                                                                 random_state=self.random_state,
                                                                 stratify=data_y)

        val_output_path_0 = os.path.join(self.output_path, config_util.val_data_file_0.format(self.dataset_name))
        val_output_path_1 = os.path.join(self.output_path, config_util.val_data_file_1.format(self.dataset_name))

        val_0_df = pd.concat([pd.DataFrame(val_x_0), pd.DataFrame(val_y_0)], axis=1)
        val_1_df = pd.concat([pd.DataFrame(val_x_1), pd.DataFrame(val_y_1)], axis=1)

        print("validation_0 dimension: {}".format(val_0_df.shape))
        print("validation_1 dimension: {}".format(val_1_df.shape))

        col_names = ["{}_{}".format(self.dataset_name, i) for i in range(data_x.shape[1])]
        col_names.append('label')

        val_0_df.columns = col_names
        val_1_df.columns = col_names

        val_0_df.to_csv(val_output_path_0, index=False)
        val_1_df.to_csv(val_output_path_1, index=False)

        print("Generating: {}".format(val_output_path_0))
        print("Generating: {}".format(val_output_path_1))


if __name__ == '__main__':
    if config_util.nn_enabled:
        prob = {
            'name': 'Neural Networks',
            'path': 'nn'
        }
        for ds in config_util.datasets:
            if not ds['process']:
                continue
            for prob_to_process in config_util.problem_to_process:
                if not prob_to_process['process']:
                    continue
                print("================================")
                print("Dataset: {}, Problem: {}, Problem_to_process:{}".format(ds["name"], prob['name'],
                                                                               prob_to_process['name']))
                data = DataPreprocessor(ds, prob, prob_to_process)
                data.load_and_process(verbose=False)
                data.split_test_train(test_size=0.2, val_size=0.5)
    else:
        for ds in config_util.datasets:
            print("================================")
            print("Processing Dataset: {}".format(ds['name']))
            data = DataPreprocessor(ds)
            data.load_and_process(verbose=False)
            data.scale_data(val_size=0.5)
