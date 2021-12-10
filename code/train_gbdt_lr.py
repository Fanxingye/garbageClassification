import argparse
import gc
import os
import pickle

import numpy as np
import pandas as pd
import ref
from data_preprocess import DataPreprocess
from estimator import SklearnEstimator
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from utils import mkdir, write_log, get_background
from gbdt_feature import LightGBMFeatureTransformer
from feature_engineering import feature_engineering


<<<<<<< HEAD
def train_test(mode, gbdt_params, lr_params, X_train, y_train, X_test, y_test, log, output_dir, fitall = True):
=======
def train_test(mode, gbdt_params, lr_params, X_train, y_train, X_test, y_test, log, output_dir):
>>>>>>> c3ca679cd70ab23bd47dc17139a63c2c82dec2d7
    gbdt = LightGBMFeatureTransformer(
            task='classification',
            params=gbdt_params
            )
    if mode == "Category":
        lr = SklearnEstimator("lr", "classification", params=lr_params)
    elif mode == "Material" or "Backgroud":
        lr = SklearnEstimator("lr", "classification")
    else:
        raise Exception("mode must be 'Category' or 'Material'")
    gbdt.fit(X_train, y_train)
    X_train = gbdt.dense_transform(X_train, keep_original=False)
    X_test = gbdt.dense_transform(X_test, keep_original=False)
    lr.fit(X_train, y_train)
    pre_test = lr.predict(X_test)
    acc = accuracy_score(y_test, pre_test)
    write_log(log, 'Accuracy of {}: {}.'.format(mode, acc))
    save_dir_cat = os.path.join(output_dir, 'model/{}'.format(mode))
    mkdir(save_dir_cat)
    pickle.dump(
        gbdt,
        open(os.path.join(save_dir_cat, 'gbdt.pkl'), 'wb'))
    pickle.dump(
        lr,
        open(os.path.join(save_dir_cat, 'lr.pkl'), 'wb'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir',
                        default='data',
                        help='The directory contains data files.')
    parser.add_argument('-use_groupbyID',
                        default='True',
                        help='Use the single ObjID data')
    parser.add_argument('-output_dir',
                        default='output',
                        help='The directory where the outputs are stored.')
    parser.add_argument('-skip_data_preprocess',
                        action='store_true',
                        default=False,
                        help='The flag to skip running data_preprocess.')
    args = parser.parse_args()

    # Run data_preprocess.py
    if not args.skip_data_preprocess:
        data_preprocess = DataPreprocess(data_dir=args.data_dir, test=False)
        data_preprocess.run_preprocess()
    else:
        print('WARN: Data preprocessing was skipped')

    mkdir(os.path.join(args.output_dir, 'log'))
    log = open(args.output_dir + '/log/log.txt', 'w')

    print('Loading Dataset...')
    if args.use_groupbyID:
        data = pd.read_csv(os.path.join(args.data_dir,
                                        'AllEmbracingDataset.csv'),
                           dtype=ref.data_dtype['AllEmbrace'])
    else:
        data = pd.read_csv(os.path.join(args.data_dir,
                                        'AllEmbracingDataset_original.csv'),
                           dtype=ref.data_dtype['AllEmbrace'])
    y_columns = ['Category', 'Material']
    data_sub = data.loc[data['Material'] != 563]  # 563 is background

    # use the select 50 class
    select_50_mat = np.array(ref.select_50_material).astype(np.int32)
    data_sub = data_sub[data_sub['Material'].isin(select_50_mat)]

    X, y = feature_engineering(data_sub, y_columns)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        stratify=y,
                                                        random_state=2021)

    y_cat_train = y_train['Category']
    y_cat_test = y_test['Category']
    y_mat_train = y_train['Material']
    y_mat_test = y_test['Material']

    X = pd.concat([X_train, X_test], axis=0)
    y_cat = pd.concat([y_cat_train, y_cat_test], axis=0)
    y_mat = pd.concat([y_mat_train, y_mat_test], axis=0)

    # LabelEncoder
    mkdir(os.path.join(args.output_dir, 'tool'))
    le = preprocessing.LabelEncoder()
    y_mat = le.fit_transform(y_mat)
    y_mat_train = le.transform(y_mat_train)
    print(
        f'min y_mat_train: {min(y_mat_train)}, max y_mat_train: {max(y_mat_train)}'
    )
    y_mat_test = le.transform(y_mat_test)
    write_log(log, 'LabelEncoder.classes_: {}.'.format(le.classes_))
    pickle.dump(
        le,
        open(os.path.join(args.output_dir, 'tool/labelEncoder.pkl'), 'wb'))
    gc.collect()

    drop_cols = ["absorbance_min", "absorbance_max"]
    cat_cols = [i for i in X_train.columns if i not in drop_cols]
    X_cat_train = X_train[cat_cols]
    X_cat_test = X_test[cat_cols]
    X_mat_train = X_train
    X_mat_test = X_test
    gbdt_params = {
                'n_estimators': 100,
                'max_depth': 3
            }
    lr_params = {'C': 0.01}
    # GBDT+lr for Category
    train_test("Category", gbdt_params, lr_params, X_cat_train, y_cat_train, X_cat_test, y_cat_test, log, args.output_dir)
<<<<<<< HEAD
    train_test("Category", gbdt_params, lr_params, X[cat_cols], y_cat, X[cat_cols], y_cat, log, args.output_dir)
    # GBDT+lr for Material
    train_test("Material", gbdt_params, lr_params, X_mat_train, y_mat_train, X_mat_test, y_mat_test, log, args.output_dir)
    train_test("Material", gbdt_params, lr_params, X, y_mat, X, y_mat, log, args.output_dir)
=======
    # GBDT+lr for Material
    train_test("Material", gbdt_params, lr_params, X_mat_train, y_mat_train, X_mat_test, y_mat_test, log, args.output_dir)
>>>>>>> c3ca679cd70ab23bd47dc17139a63c2c82dec2d7
    # GBDT+lr for Backgroud
    data_bg = data
    data_bg.loc[:, 'Material'] = data_bg.apply(get_background, axis=1)
    X, y = feature_engineering(data_bg, y_columns)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        stratify=y,
                                                        random_state=2021)

    y_bg_train = y_train['Material']
    y_bg_test = y_test['Material']
    train_test("Backgroud", gbdt_params, lr_params, X_train, y_bg_train, X_test, y_bg_test, log, args.output_dir)
<<<<<<< HEAD
    train_test("Backgroud", gbdt_params, lr_params, X, y['Material'], X, y['Material'], log, args.output_dir)
=======
>>>>>>> c3ca679cd70ab23bd47dc17139a63c2c82dec2d7
    write_log(log, 'Train and test finished !')


if __name__ == '__main__':
    main()
