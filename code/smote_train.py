import os

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    data_dir = 'D:/datasets/garbage/'
    data_train = pd.read_csv(data_dir + 'train.csv')
    data_test = pd.read_csv(data_dir + 'test.csv')
    num_train = len(data_train)
    data_all = pd.concat([data_train, data_test], axis=0)
    drop_columns = [
        'ObjID', 'SamplingPointID', 'MaxIntensity', 'MaterialCount',
        'Category', 'Material'
    ]
    wavelength_columns = [
        col_name for col_name in data_all.columns if 'absorbance' in col_name
    ] + ['MaxIntensity']
    label = ['Material']
    # X = data_all.drop(drop_columns, axis = 1)
    X = data_all[wavelength_columns]
    X = pd.get_dummies(X)
    Y = data_all[label]
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    x_train = X[:num_train]
    y_train = Y[:num_train]
    x_test = X[num_train:]
    y_test = Y[num_train:]
    upsample = False
    if upsample:
        ov = SMOTE(random_state=2021)
        x_train, y_train = ov.fit_resample(x_train, y_train)
    # classifier = SVC(kernel='rbf',
    #                 class_weight='balanced',
    #                 probability=True)
    classifier = RandomForestClassifier(n_estimators=200,
                                        random_state=2021,
                                        class_weight='balanced')
    sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)
    scores_train = []
    scores_val = []

    x_train = pd.DataFrame(x_train)
    y_train = pd.DataFrame(y_train, columns=label)
    data_train = pd.concat([x_train, y_train], axis=1)
    data_train = data_train.drop_duplicates(keep='first')
    x_train = data_train.drop('Material', axis=1)
    y_train = data_train['Material']
    x_test = pd.DataFrame(x_test)
    y_test = pd.DataFrame(y_test)

    k = 0
    for train_ind, val_ind in sk.split(x_train, y_train):
        k = k + 1
        train_x = x_train.iloc[train_ind].values
        train_y = y_train.iloc[train_ind].values
        val_x = x_train.iloc[val_ind].values
        val_y = y_train.iloc[val_ind].values
        classifier.fit(train_x, train_y)
        pred_train = classifier.predict(train_x)
        pred_val = classifier.predict(val_x)

        score_train = accuracy_score(train_y, pred_train)
        score_val = accuracy_score(val_y, pred_val)
        scores_train.append(score_train)
        scores_val.append(score_val)
        print(f'fold: {k}, Accuracy of training: {score_train}')
        print(f'fold: {k}, Accuracy of validation: {score_val}')
    print(f'Mean Accuracy of training: {np.mean(scores_train)}')
    print(f'Mean Accuracy of validation: {np.mean(scores_val)}')

    pred_test = classifier.predict(x_test)
    score_test = accuracy_score(y_test, pred_test)
    print(f'Accuracy of test set: {score_test}')
