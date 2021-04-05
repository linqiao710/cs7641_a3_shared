datasets = [
    {
        'name': 'Abalone',
        'input_file': 'abalone.csv',
        'path': 'abalone',
        'process': True
    },
    {
        'name': 'Wine',
        'input_file': 'wine_quality_red.csv',
        'path': 'wine',
        'process': False
    }
]

problem_to_process = [
    {
        'name': 'Original',
        'type': 'Unmodified',
        'path': 'original',
        'process': False
    },
    {
        'name': 'k-Means',
        'type': 'Clustering',
        'path': 'km',
        'abalone': 7,
        'wine': 4,
        'process': True
    },
    {
        'name': 'EM',
        'type': 'Clustering',
        'path': 'em',
        'abalone': 10,
        'wine': 7,
        'process': True
    },
    {
        'name': 'PCA',
        'type': 'Dim-Red',
        'path': 'pca',
        'abalone': 6,
        'wine': 10,
        'abalone_km': 8,
        'abalone_em': 16,
        'wine_km': 2,
        'wine_em': 5,
        'process': False
    },
    {
        'name': 'ICA',
        'type': 'Dim-Red',
        'path': 'ica',
        'abalone': 8,
        'wine': 4,
        'abalone_km': 5,
        'abalone_em': 13,
        'wine_km': 6,
        'wine_em': 6,
        'process': False
    },
    {
        'name': 'Random-Projection',
        'type': 'Dim-Red',
        'path': 'rp',
        'abalone': 9,
        'wine': 10,
        'abalone_km': 2,
        'abalone_em': 14,
        'wine_km': 2,
        'wine_em': 6,
        'process': False
    },
    {
        'name': 'Factor-Analysis',
        'type': 'Dim-Red',
        'path': 'fa',
        'abalone': 7,
        'wine': 10,
        'abalone_km': 3,
        'abalone_em': 13,
        'wine_km': 5,
        'wine_em': 6,
        'process': False
    }
]

nn_prob = {
    'name': 'Neural Networks',
    'path': 'nn',
    'abalone_em': (10,)*3,
    'abalone_km': (10,)*1,
    'abalone_pca': (10,)*7,
    'abalone_ica': (10,)*7,
    'abalone_rp': (10,)*5,
    'abalone_fa': (10,)*9,
    'abalone_original': (10,)*9
}

train_data_file = '{}_train.csv'
test_data_file = '{}_test.csv'
val_data_file_0 = '{}_validate_0.csv'
val_data_file_1 = '{}_validate_1.csv'

nn_train_data_file = '{}_{}_nn_data_train.csv'
nn_test_data_file = '{}_{}_nn_data_test.csv'

path_pattern = '{}/{}/{}/'
com_prob_path_pattern = '{}/{}/{}/{}/'

dr_enabled = False

nn_enabled = True

random_state = 42
