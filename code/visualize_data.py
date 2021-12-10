import argparse
import os

import numpy as np
import pandas as pd
import ref
import seaborn as sns
from data_preprocess import DataPreprocess
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.manifold import TSNE
from utils import remove_mean, wavelets_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-base_classifiers_cat',
                        default='svm_cat',
                        help='Comma-separated list of classifiers to use.')
    parser.add_argument('-base_classifiers_mat',
                        default='svm_mat',
                        help='Comma-separated list of classifiers to use.')
    parser.add_argument('-data_dir',
                        default='data',
                        help='The directory contains data files.')
    parser.add_argument('-use_groupbyID',
                        default='False',
                        help='Use the single ObjID data')
    parser.add_argument('-output_dir',
                        default='output',
                        help='The directory where the outputs are stored.')
    parser.add_argument('-ensemble',
                        action='store_true',
                        default=False,
                        help='Use ensemble method.')
    parser.add_argument('-grid_search',
                        action='store_true',
                        default=False,
                        help='Use grid search when training.')
    parser.add_argument('-train_material_classifier',
                        action='store_true',
                        default=False,
                        help='Train material classifier as well.')
    parser.add_argument(
        '-wavelets_transform',
        action='store_true',
        default=False,
        help='Apply wavelets transform before sent into classifier.')
    parser.add_argument(
        '-remove_mean',
        action='store_true',
        default=False,
        help='Subtract mean value before sent into classifier.')
    parser.add_argument('-use_gpu',
                        action='store_true',
                        default=False,
                        help='Use NVIDIA GPU to accelerate xgboost training.')
    parser.add_argument(
        '-use_standardScaler',
        action='store_true',
        default=False,
        help='Use sklearn.preprocessing.StandardScaler to process wavelength.')
    parser.add_argument(
        '-use_labelencoder',
        action='store_true',
        default=False,
        help='Use sklearn.preprocessing.LabelEncoder to encode label.')
    parser.add_argument('-train_without_shape',
                        action='store_true',
                        default=False,
                        help='Train model without shape data.')
    parser.add_argument('-train_without_wavelength',
                        action='store_true',
                        default=False,
                        help='Train model without wavelength data.')
    parser.add_argument('-skip_data_preprocess',
                        action='store_true',
                        default=False,
                        help='The flag to skip running data_preprocess.')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not args.skip_data_preprocess:
        data_preprocess = DataPreprocess(data_dir=args.data_dir, test=False)
        data_preprocess.run_preprocess()
    else:
        print('WARN: Data preprocessing was skipped')

    model_info = {}
    model_info['base_classifiers_cat'] = args.base_classifiers_cat
    model_info['base_classifiers_mat'] = args.base_classifiers_mat
    model_info['train_material_classifier'] = args.train_material_classifier
    model_info['wavelets_transform'] = args.wavelets_transform
    model_info['remove_mean'] = args.remove_mean
    model_info['use_standardScaler'] = args.use_standardScaler
    model_info['use_labelencoder'] = args.use_labelencoder
    model_info['train_without_shape'] = args.train_without_shape
    model_info['train_without_wavelength'] = args.train_without_wavelength

    print('Loading Dataset...')
    if args.use_groupbyID:
        data = pd.read_csv(os.path.join(
            args.data_dir, 'AllEmbracingDataset_groupbyObjID.csv'),
                           dtype=ref.data_dtype['AllEmbrace'])
    else:
        data = pd.read_csv(os.path.join(args.data_dir,
                                        'AllEmbracingDataset.csv'),
                           dtype=ref.data_dtype['AllEmbrace'])

    X_wavelength_columns = [
        col_name for col_name in data.columns if 'absorbance' in col_name
    ] + ['MaxIntensity']

    ScanChart_name_list = [
        'OriginalPointCount', 'ScanYAvg', 'ScanYAvgWeighted', 'ScanYStdDev',
        'ScanMass', 'ScanDensity', 'ScanMassSimple', 'ScanDensitySimple',
        'ScanMassRatio'
    ]
    OutlineChart_name_list = [
        'OutlinePointCount', 'OutlineYAvg', 'OutlineYAvgWeighted',
        'OutlineYStdDev', 'OutlineMass', 'OutlineDensity', 'OutlineMassSimple',
        'OutlineDensitySimple', 'OutlineMassRatio'
    ]
    XChart_name_list = ['MassRatio', 'MassRatioSimple', 'IsOutlineSameAsScan']
    X_shape_columns = [
        'Length'
    ] + ScanChart_name_list + OutlineChart_name_list + XChart_name_list

    y_columns = ['Category', 'Material']
    data = data.loc[data['Material'] != 563]  # 563 is background

    X_wavelength = data[X_wavelength_columns]
    X_shape = data[X_shape_columns]
    y = data[y_columns]

    if args.wavelets_transform:
        X_wavelength = wavelets_transform(X_wavelength, 'db4')
    if args.remove_mean:
        X_wavelength = remove_mean(X_wavelength)

    if args.use_standardScaler:
        scaler = preprocessing.StandardScaler()
        X_wavelength = scaler.fit_transform(X_wavelength)
        scaler = preprocessing.StandardScaler()
        X_shape = scaler.fit_transform(X_shape)

    if args.train_without_shape:
        X = X_wavelength
    elif args.train_without_wavelength:
        X = X_shape
    else:
        X = np.concatenate((X_wavelength, X_shape), axis=1)

    print(X.shape, y.shape)
    print('tsne vis')
    tsne = TSNE()
    X_embedded = tsne.fit_transform(X_wavelength)
    df = pd.DataFrame()
    df['Category'] = data['Category']
    df['Material'] = data['Material']
    df['comp-1'] = X_embedded[:, 0]
    df['comp-2'] = X_embedded[:, 1]

    plt.figure(figsize=(12, 12))
    fig1 = sns.scatterplot(x='comp-1',
                           y='comp-2',
                           hue=df['Category'],
                           palette=sns.color_palette('bright', 4),
                           legend='full',
                           data=df)

    figpath1 = os.path.join(args.output_dir, 'cat.png')
    scatter_fig1 = fig1.get_figure()
    scatter_fig1.savefig(figpath1, dpi=400)

    # plot one vs one
    cat_vis = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    cat_vis = np.array(cat_vis).astype(np.int8)

    for ind, cat_ind in enumerate(cat_vis):
        df_cat = df[df['Category'].isin(cat_ind)]
        print(df_cat.shape)
        plt.figure(figsize=(12, 12))
        fig = sns.scatterplot(x='comp-1',
                              y='comp-2',
                              hue=df_cat['Category'],
                              palette=sns.color_palette('bright', 2),
                              data=df_cat)
        figpath = os.path.join(args.output_dir, str(ind) + '_cat.png')
        scatter_fig = fig.get_figure()
        scatter_fig.savefig(figpath, dpi=400)


if __name__ == '__main__':
    main()
