import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import ref
from data_preprocess import DataPreprocess
from utils import mkdir
from feature_engineering import feature_engineering


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-output_dir',
                        default='output',
                        help='The path to the directory of a trained model.')
    parser.add_argument('-data_dir',
                        default='data',
                        help='The directory contains data files.')
    parser.add_argument('-use_groupbyID',
                        default='True',
                        help='Use the single ObjID data')
    parser.add_argument(
        '-save_dir',
        default='output/test_result',
        help='The directory where the test results are stored.')
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

    print('Loading Dataset...')
    if args.use_groupbyID:
        data = pd.read_csv(os.path.join(args.data_dir,
                                        'AllEmbracingDataset.csv'),
                           dtype=ref.data_dtype['AllEmbrace'])
    else:
        data = pd.read_csv(os.path.join(args.data_dir,
                                        'AllEmbracingDataset_original.csv'),
                           dtype=ref.data_dtype['AllEmbrace'])

    X = feature_engineering(data)
    preds = pd.DataFrame(data[['ObjID']])
    preds['Category'] = data['Category'].map(ref.id2category)
    preds['Material'] = data['Material']

    bg_dir = os.path.join(args.output_dir, "model/Backgroud")
    cat_dir = os.path.join(args.output_dir, "model/Category")
    mat_dir = os.path.join(args.output_dir, "model/Material")
    le_dir = os.path.join(args.output_dir, "tool")

    bg_gbdt = pickle.load(
        open(os.path.join(bg_dir, "gbdt.pkl"), 'rb'))
    bg_lr = pickle.load(
        open(os.path.join(bg_dir, "lr.pkl"), 'rb'))
    cat_gbdt = pickle.load(
        open(os.path.join(cat_dir, "gbdt.pkl"), 'rb'))
    cat_lr = pickle.load(
        open(os.path.join(cat_dir, "lr.pkl"), 'rb'))
    mat_gbdt = pickle.load(
        open(os.path.join(mat_dir, "gbdt.pkl"), 'rb'))
    mat_lr = pickle.load(
        open(os.path.join(mat_dir, "lr.pkl"), 'rb'))
    le = pickle.load(
        open(os.path.join(le_dir, "labelEncoder.pkl"), 'rb'))

    X_bg_gbdt = bg_gbdt.dense_transform(X, keep_original=False)
    ped_bg = bg_lr.predict(X_bg_gbdt)
    drop_cols = ["absorbance_min", "absorbance_max"]
    cat_cols = [i for i in X.columns if i not in drop_cols]
    X_cat_gbdt = cat_gbdt.dense_transform(X[cat_cols], keep_original=False)
    ped_cat = cat_lr.predict(X_cat_gbdt)
    X_mat_gbdt = mat_gbdt.dense_transform(X, keep_original=False)
    ped_mat = mat_lr.predict(X_mat_gbdt)

    preds["pred_background"] = pd.Series(ped_bg, dtype=np.uint8)
    preds["pred_Category"] = pd.Series(ped_cat, dtype=np.uint8).map(ref.id2category)
    preds["pred_Material"] = pd.Series(ped_mat, dtype=np.uint32)
    preds['pred_Material'] = le.inverse_transform(preds['pred_Material'])
    preds.loc[preds['pred_background'] == 0, 'pred_Material'] = 563
    preds.loc[preds['pred_background'] == 0, 'pred_Category'] = 'RESIDUAL'

    id_zero = ref.id_background_zero
    preds_category_grouped = preds['pred_Category'].groupby(preds['ObjID']) \
        .agg(lambda x: x.value_counts().index[0] if x.value_counts().index[0] != id_zero or x.value_counts().shape[0] == 1 else x.value_counts().index[1]).reset_index()
    preds_material_grouped = preds['pred_Material'].groupby(preds['ObjID']) \
        .agg(lambda x: x.value_counts().index[0] if x.value_counts().index[0] != id_zero or x.value_counts().shape[0] == 1 else x.value_counts().index[1]).reset_index()
    preds_category_grouped.columns = pd.Index(['ObjID', 'pred_Category_final'],
                                              dtype=object)
    preds_material_grouped.columns = pd.Index(['ObjID', 'pred_Material_final'],
                                              dtype=object)
    preds_merge1 = pd.merge(preds,
                            preds_category_grouped,
                            on='ObjID',
                            how='inner')
    preds_merge2 = pd.merge(preds_merge1,
                            preds_material_grouped,
                            on='ObjID',
                            how='inner')
    print('Saving prediction results...')
    mkdir(args.save_dir)

    preds_merge2.to_csv(os.path.join(args.save_dir, 'predictions.csv'),
                        index=False)
    print('Saving completed! Predictions were saved at this location: ')
    print('{}'.format(os.path.join(args.save_dir, 'predictions.csv')))


if __name__ == '__main__':
    main()
