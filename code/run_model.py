import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import ref
from data_preprocess import DataPreprocess
from utils import mkdir, remove_mean, wavelets_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-output_dir',
                        default='output',
                        help='The path to the directory of a trained model.')
    parser.add_argument('-output_background_dir',
                        default='output_background',
                        help='The path to the directory of a trained model.')
    parser.add_argument('-data_dir',
                        default='data',
                        help='The directory contains data files.')
    parser.add_argument(
        '-save_dir',
        default='output/test_result',
        help='The directory where the test results are stored.')
    parser.add_argument('-skip_data_preprocess',
                        action='store_true',
                        default=False,
                        help='The flag to skip running data_preprocess.')
    parser.add_argument(
        '-review_background_zero',
        action='store_true',
        default=False,
        help='The flag to review predicted materials who are background zeros.'
    )
    args = parser.parse_args()

    # Run data_preprocess.py
    if not args.skip_data_preprocess:
        data_preprocess = DataPreprocess(data_dir=args.data_dir, test=False)
        data_preprocess.run_preprocess()
    else:
        print('WARN: Data preprocessing was skipped')

    with open(os.path.join(args.output_dir, 'model/model_info.json'),
              'r') as f:
        model_info = json.load(f)
    with open(
            os.path.join(args.output_background_dir, 'model/model_info.json'),
            'r') as f:
        model_background_info = json.load(f)
        model_background = pickle.load(
            open(model_background_info['model_mat_dir'], 'rb'))

    model_cat = pickle.load(open(model_info['model_cat_dir'], 'rb'))
    if model_info['train_material_classifier']:
        model_mat = pickle.load(open(model_info['model_mat_dir'], 'rb'))
    else:
        model_mat = None

    X_wavelength_columns = [
        'absorbance_' + str(i) + '_mean' for i in range(228)
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

    print('Loading Dataset...')
    data = pd.read_csv(os.path.join(args.data_dir,
                                    'AllEmbracingDataset_groupbyObjID2.csv'),
                       dtype=ref.data_dtype['AllEmbrace'])

    X_wavelength = data[X_wavelength_columns]
    print(X_wavelength)
    X_shape = data[X_shape_columns]

    preds = pd.DataFrame(data[['ObjID']])
    preds['Category'] = data['Category'].map(ref.id2category)
    preds['Material'] = data['Material']

    print('Preprocessing Dataset...')
    if model_info['wavelets_transform']:
        X_wavelength = wavelets_transform(X_wavelength, 'db4')
    if model_info['remove_mean']:
        X_wavelength = remove_mean(X_wavelength)
        X_shape_columns = remove_mean(X_wavelength)
    if model_info['use_standardScaler']:
        scaler_wave = pickle.load(
            open(os.path.join(args.output_dir, 'tool/standardScaler_wave.pkl'),
                 'rb'))
        X_wavelength = scaler_wave.transform(X_wavelength)
        scaler_shape = pickle.load(
            open(
                os.path.join(args.output_dir, 'tool/standardScaler_shape.pkl'),
                'rb'))
        X_shape = scaler_shape.transform(X_shape)

    if model_info['train_without_shape']:
        X = X_wavelength
    elif model_info['train_without_wavelength']:
        X = X_shape
    else:
        X = np.concatenate((X_wavelength, X_shape), axis=1)

    print('Predicting classification results...')
    preds['pred_background'] = pd.Series(
        model_background.predict(X_wavelength), dtype=np.uint8)
    preds['pred_Category'] = pd.Series(model_cat.predict(X),
                                       dtype=np.uint8).map(ref.id2category)
    if model_mat:
        preds['pred_Material'] = pd.Series(model_mat.predict(X),
                                           dtype=np.uint32)
    print('Transforming labels...')
    if model_info['use_labelencoder']:
        le = pickle.load(
            open(os.path.join(args.output_dir, 'tool/labelEncoder.pkl'), 'rb'))
        preds['pred_Material'] = le.inverse_transform(preds['pred_Material'])

    preds.loc[preds['pred_background'] == 0, 'pred_Material'] = 563
    preds.loc[preds['pred_background'] == 0, 'pred_Category'] = 'RESIDUAL'

    if args.review_background_zero:
        print('Reviewing background zero...')
        print('Background zero ID: {}'.format(ref.id_background_zero))
        id_zero = ref.id_background_zero
        preds_category_grouped = preds['pred_Category'].groupby(preds['ObjID']) \
            .agg(lambda x: x.value_counts().index[0] if x.value_counts().index[0] != id_zero or x.value_counts().shape[0] == 1 else x.value_counts().index[1]).reset_index()
        preds_material_grouped = preds['pred_Material'].groupby(preds['ObjID']) \
            .agg(lambda x: x.value_counts().index[0] if x.value_counts().index[0] != id_zero or x.value_counts().shape[0] == 1 else x.value_counts().index[1]).reset_index()
    else:
        preds_category_grouped = preds['pred_Category'].groupby(preds['ObjID']) \
            .agg(lambda x: x.value_counts().index[0]).reset_index()
        preds_material_grouped = preds['pred_Material'].groupby(preds['ObjID']) \
            .agg(lambda x: x.value_counts().index[0]).reset_index()
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
    # preds.to_csv(os.path.join(args.save_dir, 'predictions.csv'), index=False)
    print('Saving completed! Predictions were saved at this location: ')
    print('{}'.format(os.path.join(args.save_dir, 'predictions.csv')))


if __name__ == '__main__':
    main()
