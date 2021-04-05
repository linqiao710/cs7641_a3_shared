import os
import glob
import re
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.axes as maxes
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import NullFormatter, FormatStrFormatter, MaxNLocator
from os.path import basename

import config_util
import load_data
from load_data import get_x

file_to_process = [
    {
        'name_pattern': '{}/*_inertia.csv',
        'index_name': 'numberOfClusters',
        'plot_cols': ['inertia'],
        'plot_type': 'line_mark',
        'process': True
    },
    {
        'name_pattern': '{}/*_silhouetteScore.csv',
        'index_name': 'numberOfClusters',
        'plot_cols': ['silhouetteScore'],
        'plot_type': 'line_mark',
        'process': True
    },
    {
        'name_pattern': '{}/*_silhouetteCoefficient.csv',
        'index_name': 'numberOfClusters',
        'plot_cols': [],
        'plot_type': 'sil_sample',
        'process': True
    },
    {
        'name_pattern': '{}/*_informationCriterion.csv',
        'index_name': 'numberOfClusters',
        'plot_cols': ['AIC', 'BIC'],
        'plot_type': 'line_mark',
        'process': True
    },
    {
        'name_pattern': '{}/*_logLikelihood.csv',
        'index_name': 'numberOfClusters',
        'plot_cols': ['logLikelihood'],
        'plot_type': 'line_mark',
        'process': True
    },
    {
        'name_pattern': '{}/*_tsne_scaledData_*.csv',
        'index_name': 'numberOfClusters',
        'plot_cols': ['tsne_0', 'tsne_1', 'label'],
        'plot_type': 'tsne',
        'process': True
    },
    {
        'name_pattern': '{}/*_tsne_reconstructedData_*.csv',
        'index_name': 'numberOfClusters',
        'plot_cols': ['tsne_0', 'tsne_1', 'label'],
        'plot_type': 'tsne',
        'process': True
    },
    {
        'name_pattern': '{}/*_eigenValue.csv',
        'index_name': 'numberOfComponents',
        'plot_cols': ['eigenValue'],
        'plot_type': 'bar',
        'process': True
    },
    {
        'name_pattern': '{}/*_eigenValues.csv',
        'index_name': 'numberOfComponents',
        'plot_cols': ['eigenValue'],
        'plot_type': 'bar',
        'process': True
    },
    {
        'name_pattern': '{}/*_cumulativeVarianceRatio.csv',
        'index_name': 'numberOfComponents',
        'plot_cols': ['cumulativeVarianceRatio'],
        'plot_type': 'line_mark',
        'process': True
    },
    {
        'name_pattern': '{}/*_kurtosis.csv',
        'index_name': 'numberOfComponents',
        'plot_cols': ['kurtosis'],
        'plot_type': 'bar',
        'process': True
    },
    {
        'name_pattern': '{}/*_kurtoses.csv',
        'index_name': 'numberOfComponents',
        'plot_cols': ['kurtosis'],
        'plot_type': 'bar',
        'process': True
    },
    {
        'name_pattern': '{}/*_kurtosisMean.csv',
        'index_name': 'numberOfComponents',
        'plot_cols': ['kurtosisMean'],
        'plot_type': 'bar',
        'process': True
    },
    {
        'name_pattern': '{}/*_reconstructionError.csv',
        'index_name': 'numberOfComponents',
        'plot_cols': [],
        'plot_type': 'line_mark_multi',
        'process': True
    },
    {
        'name_pattern': '{}/*_reconstructionErrors.csv',
        'index_name': 'numberOfComponents',
        'plot_cols': [],
        'plot_type': 'line_mark_mean_std',
        'process': True
    },
    {
        'name_pattern': '{}/*_nn_accuracy.csv',
        'index_name': 'layers',
        'plot_cols': ['accuracy'],
        'plot_type': 'line_mark',
        'process': True
    },
    {
        'name_pattern': '{}/*_nn_loss.csv',
        'index_name': 'iterations',
        'plot_cols': ["(10,)*1", "(10,)*3", "(10,)*5", "(10,)*7", "(10,)*9"],
        'plot_type': 'line_mark_multi',
        'process': True
    },
    {
        'name_pattern': '{}/*_crossValidationScores.csv',
        'index_name': 'numberOfComponents',
        'plot_cols': ['crossValidationScores'],
        'plot_type': 'line_mark',
        'process': True
    }
]


def plot_line(df, column_name, problem_path, title, x_label, y_label, clear_existing=True, mark=True, std=False):
    if clear_existing:
        plt.close()
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title(title, fontsize=16)

        plt.xlabel(x_label, fontsize=16)
        plt.ylabel(y_label, fontsize=16)
        plt.grid()
        plt.tight_layout()
    if mark:
        plt.plot(df.index.values, df[column_name], 'o-', linewidth=3, markersize=5,
                 label="{}_{}".format(problem_path, column_name))
    else:
        plt.plot(df.index.values, df[column_name], linewidth=3,
                 label="{}_{}".format(problem_path, column_name))

    if std:
        plt.fill_between(df.index.values, df[column_name] - df["std"],
                         df[column_name] + df["std"], alpha=0.3)
    plt.legend(loc="best")
    return plt


def plot_sil_samples(title, df, y_label, x_label, n_clusters):
    plt.close()
    plt.figure().set_size_inches(12, 10)

    df = df[df['numberOfClusters'] == n_clusters]
    ax = plt.gca()

    sample_silhouette_values = df['score'].astype(np.double)
    ax.set_xlim([-0.1, 0.8])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax.set_ylim([0, df.shape[0] + (n_clusters + 1) * 12])
    silhouette_mean = df['score_mean'].mean()
    cluster_labels = df['label'].astype(np.float).values

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i].values
        size_cluster_i = ith_cluster_silhouette_values.shape[0]

        ith_cluster_silhouette_values.sort()

        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.axvline(x=silhouette_mean, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8])

    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.grid()
    plt.tight_layout()

    return plt


def plot_tsne(title, df, column_names):
    plt.close()
    plt.figure()

    possible_clusters = list(set(df['label']))

    ax = plt.gca()
    for g in possible_clusters:
        g = int(g)
        index = df[column_names[2]] == g
        ix = df[column_names[0]][index.values]
        iy = df[column_names[1]][index.values]
        ax.scatter(x=ix, y=iy, label=g, s=10)

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())

    ax.legend()
    plt.title(title, fontsize=16)
    plt.xlabel(column_names[0], fontsize=16)
    plt.ylabel(column_names[1], fontsize=16)
    plt.tight_layout()

    return plt


def plot_bar(title, df, column_names, y_label, x_label):
    plt.close()
    plt.figure()
    ax = plt.gca()

    y = df[column_names[0]].values
    x = df.index.values
    ax.bar(x, y)
    ax.set_xticks(x)
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.xticks(rotation='horizontal')
    plt.tight_layout()

    return plt


def plot_hist(title, df, column_names, y_label, x_label):
    plt.close()
    plt.figure()

    y = df[column_names[0]].values
    counts, bins = np.histogram(y)
    plt.hist(bins[:-1], bins, weights=counts)

    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.xticks(rotation='horizontal')
    plt.tight_layout()

    return plt


def read_and_plot_line(files, output_dir, index_name, cur_dataset, cur_problem_1, cur_problem_2, col_names):
    for cur_file in files:
        base_file_name = basename(cur_file)
        if cur_problem_2 is not None:
            output_file_name_regex = re.compile('([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)\.csv')
            _, _, _, param_name = output_file_name_regex.search(base_file_name).groups()
        else:
            output_file_name_regex = re.compile('([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)\.csv')
            _, _, param_name = output_file_name_regex.search(base_file_name).groups()

        df = pd.read_csv(cur_file).set_index(index_name)

        for i, col_name in enumerate(col_names):
            if i == 0:
                clear_existing = True
                if cur_problem_2 is not None:
                    title = '{}: {}+{} \n{} vs {}'.format(cur_dataset['name'],
                                                          cur_problem_2['name'], cur_problem_1['name'],
                                                          param_name, index_name)
                else:
                    title = '{}: {} [ {} ] \n{} vs {}'.format(cur_dataset['name'], cur_problem_1['type'],
                                                              cur_problem_1['name'],
                                                              param_name, index_name)
                y_label = param_name
                x_label = index_name
            else:
                clear_existing = False
            p = plot_line(df=df, problem_path=cur_problem_1['path'], column_name=col_name, title=title, y_label=y_label,
                          x_label=x_label, clear_existing=clear_existing, mark=True)

        if cur_problem_2 is not None:
            out_file_name = '{}/{}_{}_{}_{}.png'.format(output_dir, cur_dataset['path'], cur_problem_1['path'],
                                                        cur_problem_2['path'], param_name)
        else:
            out_file_name = '{}/{}_{}_{}.png'.format(output_dir, cur_dataset['path'], cur_problem_1['path'], param_name)
        p.savefig(out_file_name)
        print("Plotting file {} to {}".format(cur_file, out_file_name))


def read_and_plot_line_multi(files, output_dir, index_name, cur_dataset, cur_problem_1, cur_problem_2):
    for cur_file in files:
        base_file_name = basename(cur_file)
        if cur_problem_2 is not None:
            output_file_name_regex = re.compile('([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)\.csv')
            _, _, _, param_name = output_file_name_regex.search(base_file_name).groups()
        else:
            output_file_name_regex = re.compile('([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)\.csv')
            _, _, param_name = output_file_name_regex.search(base_file_name).groups()

        df = pd.read_csv(cur_file).set_index(index_name)
        col_names = df.columns.values

        for i, col_name in enumerate(col_names):
            if i == 0:
                clear_existing = True
                if cur_problem_2 is not None:
                    title = '{}: {}+{} \n{} vs {}'.format(cur_dataset['name'],
                                                          cur_problem_2['name'], cur_problem_1['name'],
                                                          param_name, index_name)
                else:
                    title = '{}: {} [ {} ] \n{} vs {}'.format(cur_dataset['name'], cur_problem_1['type'],
                                                              cur_problem_1['name'],
                                                              param_name, index_name)
                y_label = param_name
                x_label = index_name
            else:
                clear_existing = False
            p = plot_line(df=df, problem_path=cur_problem_1['path'], column_name=col_name, title=title, y_label=y_label,
                          x_label=x_label, clear_existing=clear_existing, mark=True)

        if cur_problem_2 is not None:
            out_file_name = '{}/{}_{}_{}_{}.png'.format(output_dir, cur_dataset['path'], cur_problem_1['path'],
                                                        cur_problem_2['path'], param_name)
        else:
            out_file_name = '{}/{}_{}_{}.png'.format(output_dir, cur_dataset['path'], cur_problem_1['path'], param_name)
        p.savefig(out_file_name)
        print("Plotting file {} to {}".format(cur_file, out_file_name))


def read_and_plot_mean_std(files, output_dir, index_name, cur_dataset, cur_problem_1, cur_problem_2):
    for cur_file in files:
        base_file_name = basename(cur_file)
        if cur_problem_2 is not None:
            output_file_name_regex = re.compile('([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)\.csv')
            _, _, _, param_name = output_file_name_regex.search(base_file_name).groups()
        else:
            output_file_name_regex = re.compile('([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)\.csv')
            _, _, param_name = output_file_name_regex.search(base_file_name).groups()

        df = pd.read_csv(cur_file).set_index(index_name)
        col_names = ['mean']

        for i, col_name in enumerate(col_names):
            if i == 0:
                clear_existing = True
                if cur_problem_2 is not None:
                    title = '{}: {}+{} \n{} vs {}'.format(cur_dataset['name'],
                                                          cur_problem_2['name'], cur_problem_1['name'],
                                                          param_name, index_name)
                else:
                    title = '{}: {} [ {} ] \n{} vs {}'.format(cur_dataset['name'], cur_problem_1['type'],
                                                              cur_problem_1['name'],
                                                              param_name, index_name)
                y_label = param_name
                x_label = index_name
            else:
                clear_existing = False
            p = plot_line(df=df, problem_path=cur_problem_1['path'], column_name=col_name, title=title, y_label=y_label,
                          x_label=x_label, clear_existing=clear_existing, mark=True,
                          std=True)

        if cur_problem_2 is not None:
            out_file_name = '{}/{}_{}_{}_{}.png'.format(output_dir, cur_dataset['path'], cur_problem_1['path'],
                                                        cur_problem_2['path'], param_name)
        else:
            out_file_name = '{}/{}_{}_{}.png'.format(output_dir, cur_dataset['path'], cur_problem_1['path'], param_name)
        p.savefig(out_file_name)
        print("Plotting file {} to {}".format(cur_file, out_file_name))


def read_and_plot_sil_samples(files, output_dir, cur_dataset, cur_problem_1, cur_problem_2):
    for cur_file in files:
        base_file_name = basename(cur_file)
        if cur_problem_2 is not None:
            output_file_name_regex = re.compile('([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)\.csv')
            _, _, _, param_name = output_file_name_regex.search(base_file_name).groups()
        else:
            output_file_name_regex = re.compile('([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)\.csv')
            _, _, param_name = output_file_name_regex.search(base_file_name).groups()

        y_label = "Cluster"
        x_label = "silhouetteCoefficient"

        df = pd.read_csv(cur_file)
        cluster_sizes = list(set(df['numberOfClusters']))
        for k in cluster_sizes:
            print(" - Processing k={}".format(k))
            if cur_problem_2 is not None:
                title = '{}: {}+{} [k={}] \nCluster vs silhouetteCoefficient'.format(cur_dataset['name'],
                                                                                     cur_problem_2['name'],
                                                                                     cur_problem_1['name'], k)
            else:
                title = '{}: {} [{}: k={}] \n Cluster vs silhouetteCoefficient'.format(cur_dataset['name'],
                                                                                       cur_problem_1['type'],
                                                                                       cur_problem_1['name'], k)
            p = plot_sil_samples(title=title, df=df, y_label=y_label, x_label=x_label, n_clusters=k)

            if cur_problem_2 is not None:
                out_file_name = '{}/{}_{}_{}_{}_{}.png'.format(output_dir, cur_dataset['path'], cur_problem_1['path'],
                                                               cur_problem_2['path'], param_name, k)
            else:
                out_file_name = '{}/{}_{}_{}_{}.png'.format(output_dir, cur_dataset['path'], cur_problem_1['path'],
                                                            param_name, k)
            p.savefig(out_file_name)
            print("Plotting file {} to {}".format(cur_file, out_file_name))


def read_and_plot_tsne(files, output_dir, cur_dataset, cur_problem_1, cur_problem_2, col_names):
    for cur_file in files:
        base_file_name = basename(cur_file)
        if cur_problem_2 is not None:
            output_file_name_regex = re.compile(
                '([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)_([0-9]+)\.csv')
            _, _, _, _, param_name, k = output_file_name_regex.search(base_file_name).groups()

            title = '{}: {}+{} [cluster#={}] \n {} vs {}'.format(cur_dataset['name'],
                                                                 cur_problem_2['name'], cur_problem_1['name'], k,
                                                                 col_names[1],
                                                                 col_names[0])
        else:
            output_file_name_regex = re.compile('([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)_([0-9]+)\.csv')
            _, _, _, param_name, k = output_file_name_regex.search(base_file_name).groups()

            if cur_problem_1['type'] == 'Unmodified':
                title = '{}: Original Data \n {} vs {}'.format(cur_dataset['name'], col_names[1], col_names[0])
            elif cur_problem_1['type'] == 'Dim-Red':
                title = '{}: {} [{}: dim={}] \n {} vs {}'.format(cur_dataset['name'], cur_problem_1['type'],
                                                                 cur_problem_1['name'], k,
                                                                 col_names[1],
                                                                 col_names[0])
            else:
                title = '{}: {} [{}: cluster#={}] \n {} vs {}'.format(cur_dataset['name'], cur_problem_1['type'],
                                                                      cur_problem_1['name'], k,
                                                                      col_names[1],
                                                                      col_names[0])

        df = pd.read_csv(cur_file)
        p = plot_tsne(title=title, df=df, column_names=col_names)
        if cur_problem_2 is not None:
            out_file_name = '{}/{}_{}_{}_{}_{}.png'.format(output_dir, cur_dataset['path'], cur_problem_1['path'],
                                                           cur_problem_2['path'],
                                                           param_name, k)
        else:
            out_file_name = '{}/{}_{}_{}_{}.png'.format(output_dir, cur_dataset['path'], cur_problem_1['path'],
                                                        param_name, k)
        p.savefig(out_file_name)
        print("Plotting file {} to {}".format(cur_file, out_file_name))


def read_and_plot_bar(files, output_dir, index_name, cur_dataset, cur_problem_1, cur_problem_2, col_names):
    for cur_file in files:
        base_file_name = basename(cur_file)
        if cur_problem_2 is not None:
            output_file_name_regex = re.compile('([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)\.csv')
            _, _, _, param_name = output_file_name_regex.search(base_file_name).groups()
        else:
            output_file_name_regex = re.compile('([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)\.csv')
            _, _, param_name = output_file_name_regex.search(base_file_name).groups()

        y_label = param_name
        x_label = index_name

        if cur_problem_2 is not None:
            title = '{}: {}+{} \n{} vs {}'.format(cur_dataset['name'],
                                                  cur_problem_2['name'], cur_problem_1['name'],
                                                  param_name, index_name)
        else:
            title = '{}: {} [ {} ] \n {} vs {}'.format(cur_dataset['name'], cur_problem_1['type'],
                                                       cur_problem_1['name'],
                                                       y_label, x_label)

        df = pd.read_csv(cur_file).set_index(index_name)
        p = plot_bar(title=title, df=df, column_names=col_names, y_label=y_label, x_label=x_label)

        if cur_problem_2 is not None:
            out_file_name = '{}/{}_{}_{}_{}.png'.format(output_dir, cur_dataset['path'], cur_problem_1['path'],
                                                        cur_problem_2['path'], param_name)
        else:
            out_file_name = '{}/{}_{}_{}.png'.format(output_dir, cur_dataset['path'], cur_problem_1['path'], param_name)
        p.savefig(out_file_name)
        print("Plotting file {} to {}".format(cur_file, out_file_name))


def read_and_plot_hist(files, output_dir, index_name, cur_dataset, cur_problem_1, cur_problem_2, col_names):
    for cur_file in files:
        base_file_name = basename(cur_file)
        if cur_problem_2 is not None:
            output_file_name_regex = re.compile('([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)_([0-9]+)\.csv')
            _, _, _, param_name, k = output_file_name_regex.search(base_file_name).groups()
        else:
            output_file_name_regex = re.compile('([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)_([0-9]+)\.csv')
            _, _, param_name, k = output_file_name_regex.search(base_file_name).groups()

        y_label = 'Count'
        x_label = param_name

        if cur_problem_2 is not None:
            title = '{}: {}+{} [dim={}] \n{} vs {}'.format(cur_dataset['name'],
                                                           cur_problem_2['name'], cur_problem_1['name'], k, y_label,
                                                           x_label)
        else:
            title = '{}: {} [{}: dim={}] \n{} vs {}'.format(cur_dataset['name'], cur_problem_1['type'],
                                                            cur_problem_1['name'], k, y_label, x_label)

        df = pd.read_csv(cur_file)
        p = plot_hist(title=title, df=df, column_names=col_names, y_label=y_label, x_label=x_label)

        if cur_problem_2 is not None:
            out_file_name = '{}/{}_{}_{}_{}_{}.png'.format(output_dir, cur_dataset['path'], cur_problem_1['path'],
                                                           cur_problem_2['path'], param_name, str(k))
        else:
            out_file_name = '{}/{}_{}_{}_{}.png'.format(output_dir, cur_dataset['path'], cur_problem_1['path'],
                                                        param_name, str(k))
        p.savefig(out_file_name)
        print("Plotting file {} to {}".format(cur_file, out_file_name))


def read_and_plot_problem(dataset, problem_to_process_1, problem_to_process_2=None):
    if problem_to_process_2 is not None:
        # populate input paths
        problem_input_path = load_data.get_ds_comb_prob_path('output', dataset['path'], problem_to_process_1['path'],
                                                             problem_to_process_2['path'])
        problem_output_path = load_data.get_ds_comb_prob_path('plots', dataset['path'], problem_to_process_1['path'],
                                                              problem_to_process_2['path'])
    else:
        # populate input paths
        problem_input_path = load_data.get_ds_prob_path('output', dataset['path'], problem_to_process_1['path'])
        problem_output_path = load_data.get_ds_prob_path('plots', dataset['path'], problem_to_process_1['path'])

    if not os.path.exists(problem_output_path):
        os.makedirs(problem_output_path)

    for cur_file in file_to_process:
        file_paths = glob.glob(cur_file['name_pattern'].format(problem_input_path))

        if len(file_paths) == 0 or not cur_file['process']:
            continue

        if cur_file['plot_type'] == 'line_mark':
            read_and_plot_line(files=file_paths, output_dir=problem_output_path,
                               index_name=cur_file['index_name'], cur_dataset=dataset,
                               cur_problem_1=problem_to_process_1,
                               cur_problem_2=problem_to_process_2, col_names=cur_file['plot_cols'])

        elif cur_file['plot_type'] == 'sil_sample':
            read_and_plot_sil_samples(files=file_paths, output_dir=problem_output_path, cur_dataset=dataset,
                                      cur_problem_1=problem_to_process_1, cur_problem_2=problem_to_process_2)

        elif cur_file['plot_type'] == 'tsne':
            read_and_plot_tsne(files=file_paths, output_dir=problem_output_path, cur_dataset=dataset,
                               cur_problem_1=problem_to_process_1, cur_problem_2=problem_to_process_2,
                               col_names=cur_file['plot_cols'])

        elif cur_file['plot_type'] == 'bar':
            read_and_plot_bar(files=file_paths, output_dir=problem_output_path, index_name=cur_file['index_name'],
                              cur_dataset=dataset, cur_problem_1=problem_to_process_1,
                              cur_problem_2=problem_to_process_2, col_names=cur_file['plot_cols'])

        elif cur_file['plot_type'] == 'hist':
            read_and_plot_hist(files=file_paths, output_dir=problem_output_path, index_name=cur_file['index_name'],
                               cur_dataset=dataset, cur_problem_1=problem_to_process_1,
                               cur_problem_2=problem_to_process_2, col_names=cur_file['plot_cols'])

        elif cur_file['plot_type'] == 'line_mark_mean_std':
            read_and_plot_mean_std(files=file_paths, output_dir=problem_output_path,
                                   index_name=cur_file['index_name'], cur_dataset=dataset,
                                   cur_problem_1=problem_to_process_1, cur_problem_2=problem_to_process_2)

        elif cur_file['plot_type'] == 'line_mark_multi':
            read_and_plot_line_multi(files=file_paths, output_dir=problem_output_path,
                                     index_name=cur_file['index_name'], cur_dataset=dataset,
                                     cur_problem_1=problem_to_process_1, cur_problem_2=problem_to_process_2)


if __name__ == '__main__':
    if config_util.dr_enabled or config_util.nn_enabled:
        for ds in config_util.datasets:
            if not ds['process']:
                continue

            for prob_to_process_1 in config_util.problem_to_process:
                if config_util.nn_enabled:
                    if not prob_to_process_1['process']:
                        continue
                    prob_to_process_2 = config_util.nn_prob
                    print("================================")
                    print("Dataset:{},Problem_to_process:{},{} ".format(ds['name'],
                                                                        prob_to_process_1['name'],
                                                                        prob_to_process_2['name']))
                    read_and_plot_problem(ds, prob_to_process_1, prob_to_process_2)

                elif config_util.dr_enabled:
                    if not prob_to_process_1['process'] or prob_to_process_1['type'] != 'Clustering':
                        continue
                    for prob_to_process_2 in config_util.problem_to_process:
                        if not prob_to_process_2['process'] or prob_to_process_2['type'] != 'Dim-Red':
                            continue

                        print("================================")
                        print("Dataset:{},Problem_to_process:{},{} ".format(ds['name'],
                                                                            prob_to_process_1['name'],
                                                                            prob_to_process_2['name']))
                        read_and_plot_problem(ds, prob_to_process_1, prob_to_process_2)
    else:
        for ds in config_util.datasets:
            if not ds['process']:
                continue
            for prob_to_process in config_util.problem_to_process:
                if not prob_to_process['process']:
                    continue

                print("================================")
                print("Plotting Dataset: {}, Problem_to_process: {}, ".format(ds['name'], prob_to_process['name']))
                read_and_plot_problem(ds, problem_to_process_1=prob_to_process)
