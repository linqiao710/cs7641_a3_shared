import glob
import os
import time
from collections import defaultdict
from os.path import basename
import re
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
from sklearn.neural_network import MLPClassifier

import config_util
import load_data
from load_data import DataLoader, get_x, get_y


def nn_experiment(dl, dataset, problem, problem_to_process_1):
    # populate input paths
    problem_output_path = load_data.get_ds_comb_prob_path('output', dataset['path'], problem_to_process_1['path'],
                                                          problem['path'])
    if not os.path.exists(problem_output_path):
        os.makedirs(problem_output_path)

    train_x = get_x(dl.get_train_data())
    train_y = get_y(dl.get_train_data())

    hidden_layer_sizes = list((10,) * ele for ele in range(1, 11, 2))
    max_iter = 100

    loss_val = defaultdict(dict)
    accuracy_val = []
    for hidden_layer_size in hidden_layer_sizes:
        mlp = MLPClassifier(activation="relu", max_iter=max_iter, early_stopping=False,
                            hidden_layer_sizes=hidden_layer_size,
                            random_state=config_util.random_state)

        mlp.fit(train_x, train_y)
        accuracy_val.append([len(hidden_layer_size), mlp.score(train_x, train_y)])

        for iter in range(max_iter):
            if iter < len(mlp.loss_curve_):
                loss_val[iter][str(len(hidden_layer_size))] = mlp.loss_curve_[iter]
            else:
                loss_val[iter][str(len(hidden_layer_size))] = loss_val[iter - 1][str(len(hidden_layer_size))]
        pass

    col_names = []
    for i in range(1, 11, 2):
        col_names.append('(10,)*{}'.format(str(i)))

    loss_df = pd.DataFrame(loss_val).T
    loss_df = pd.DataFrame(loss_df.values, index=range(max_iter), columns=col_names)
    loss_df.index.name = 'iterations'

    loss_df_name = (problem_output_path + '{}_{}_{}_{}.csv').format(dataset['path'],
                                                                    problem_to_process_1['path'],
                                                                    problem['path'], "loss")
    loss_df.to_csv(loss_df_name, index=True)
    print("Creating {} ".format(loss_df_name))

    accuracy_df = pd.DataFrame(accuracy_val, columns=['layers', 'accuracy']).set_index(
        'layers')
    accuracy_df_name = (problem_output_path + '{}_{}_{}_{}.csv').format(dataset['path'],
                                                                        problem_to_process_1['path'],
                                                                        problem['path'], "accuracy")
    accuracy_df.to_csv(accuracy_df_name)
    print("Creating {} ".format(accuracy_df_name))


def nn_valuate(dl, dataset, problem, best_param_value, problem_to_process_1):
    # populate input paths
    problem_output_path = load_data.get_ds_comb_prob_path('output', dataset['path'], problem_to_process_1['path'],
                                                          problem['path'])
    if not os.path.exists(problem_output_path):
        os.makedirs(problem_output_path)

    x_train = get_x(dl.get_train_data())
    y_train = get_y(dl.get_train_data())
    x_test = get_x(dl.get_test_data())
    y_test = get_y(dl.get_test_data())

    max_iter = 100
    mlp = MLPClassifier(activation="relu", max_iter=max_iter, early_stopping=False,
                        hidden_layer_sizes=best_param_value,
                        random_state=config_util.random_state)

    print("Best Param: {}".format(best_param_value))

    # train
    train_start = time.time()
    mlp.fit(x_train, y_train)
    train_end = time.time()

    # predict
    predict_start = time.time()
    train_predict = mlp.predict(x_train)
    test_predict = mlp.predict(x_test)
    predict_end = time.time()

    # accuracy
    train_score = accuracy_score(y_train, train_predict)
    test_score = accuracy_score(y_test, test_predict)

    # error rate
    train_error_rate = 1.0 - train_score
    test_error_rate = 1.0 - test_score

    # confusion matrix
    train_cm = confusion_matrix(y_train, train_predict).ravel()
    test_cm = confusion_matrix(y_test, test_predict).ravel()

    # recall
    train_recall = recall_score(y_train, train_predict).ravel()
    test_recall = recall_score(y_test, test_predict).ravel()

    # precision
    train_precision = precision_score(y_train, train_predict).ravel()
    test_precision = precision_score(y_test, test_predict).ravel()

    scores_df_name = (problem_output_path + '{}_{}_{}_{}.csv').format(dataset['path'],
                                                                      problem_to_process_1['path'],
                                                                      problem['path'], "scores")
    f = open(scores_df_name, "w")

    f.write("Train Results\n")
    f.write(f"Accuracy: {train_score}\n")
    f.write(f"Error Rate: {train_error_rate}\n")
    f.write(f"Confusion Matrix: {train_cm}\n")
    f.write(f"Recall: {train_recall}\n")
    f.write(f"Precision: {train_precision}\n")
    f.write("==================\n")
    f.write("Test Results\n")
    f.write(f"Accuracy: {test_score}\n")
    f.write(f"Error Rate: {test_error_rate}\n")
    f.write(f"Confusion Matrix: {test_cm}\n")
    f.write(f"Recall: {test_recall}\n")
    f.write(f"Precision: {test_precision}\n")
    f.write("==================\n")
    f.write("Time\n")
    f.write(f"Training Time: {train_end - train_start}\n")
    f.write(f"Prediction Time: {predict_end - predict_start}\n")

    f.close()
    print("Creating {} ".format(scores_df_name))


if __name__ == '__main__':

    for ds in config_util.datasets:
        if not ds['process']:
            continue
        for prob_to_process in config_util.problem_to_process:
            if not prob_to_process['process']:
                continue
            prob = config_util.nn_prob
            print("================================")
            print("Dataset: {}, Problem: {}, Problem_to_process:{}".format(ds["name"], prob['name'],
                                                                           prob_to_process['name']))
            data = DataLoader(ds, prob, prob_to_process)
            nn_experiment(data, ds, prob, prob_to_process)

            best_param = prob["{}_{}".format(ds['path'], prob_to_process['path'])]
            nn_valuate(dl=data, dataset=ds, problem=prob, best_param_value=best_param,
                       problem_to_process_1=prob_to_process)
