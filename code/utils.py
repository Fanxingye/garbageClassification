import itertools
import os
import random
import statistics
from datetime import datetime
from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import ref
import tqdm
# from autofe.optuna_tuner.registry import MULTICLASS_CLASSIFICATION
# from autofe.optuna_tuner.rf_optuna import RandomForestOptuna
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def split_train_test(objects, test_percent=0.15, random_state=1):
    length = len(objects)
    idx = np.arange(length)
    n = max(1, int(length * test_percent))
    random.seed(random_state)
    random.shuffle(idx)
    return {'train': idx[:-n], 'test': idx[-n:]}


def get_estimator(classifier, class_weight='balanced'):
    if classifier == 'xgboost':
        estimator = XGBClassifier()
    elif classifier == 'randomforest':
        estimator = RandomForestClassifier()
    elif classifier == 'randomforestoptuna':
        estimator = RandomForestOptuna(task=MULTICLASS_CLASSIFICATION)
    elif classifier == 'svm':
        print('Class weights: {}'.format(class_weight))
        estimator = SVC(kernel='rbf',
                        class_weight=class_weight,
                        probability=True)
    else:
        print('Unsupported classifier: {}'.format(classifier))
        exit()
    return estimator


def update_learner_params(learner_params, best):
    for k, v in best.items():
        if k in learner_params.keys():
            learner_params[k] = v
    return learner_params


def grid_search(classifier,
                X,
                y,
                class_weight='balanced',
                cv=None,
                use_gpu=False,
                write_log='training_log.txt'):
    classifier_prefix = classifier.split('_')[0]
    estimator = get_estimator(classifier_prefix, class_weight)
    grid = ref.grid[classifier]
    if use_gpu and 'xgboost' in classifier:
        grid['tree_method'] = ['gpu_hist']
        grid['gpu_id'] = [0]
    gs = GridSearchCV(estimator=estimator,
                      param_grid=grid,
                      cv=cv,
                      scoring='accuracy',
                      verbose=2)
    gs.fit(X, y)
    with open(write_log, 'w') as f:
        f.write('Best parameters: %s\n' % gs.best_params_)
        f.write('CV Accuracy: %.3f\n' % gs.best_score_)
    return gs.best_estimator_.fit(X, y)


def wavelets_transform(data, wavelet, level=1):
    for i in range(level):
        data, _ = pywt.dwt(data, wavelet)
    return data


def remove_mean(data: pd.DataFrame) -> pd.DataFrame:
    data_ = data - np.mean(data, axis=-1, keepdims=True)
    # data = np.concatenate([data, data_], axis=-1)
    return data_


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,
                 i,
                 format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        print('%s has been created, will use the  origin dir' % (dir))


def write_log(logfile, content):
    t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logfile.write('{}\t{}\n'.format(t, content))
    print(content)


def groupby_zero(data):
    result = [list(g) for k, g in groupby(data, lambda x: x == 0) if not k]
    length = len(result)
    return length


def get_std(data):
    std = statistics.stdev(data)
    return std


def get_background(x):
    if x['Material'] != 563:
        return 1
    else:
        return 0


def collect_focus_material(data_dir):
    """collect the 50 focus class."""
    data_file = os.path.join(data_dir, 'Focus50Class.csv')
    data = pd.read_csv(data_file)
    focus50_list = data['MaterialID'].values.tolist()
    return focus50_list


def parse_material_mapping(material_mapping_path):
    cat2mat = {}
    mat2cat = {}
    with open(material_mapping_path, 'r', encoding='utf-8') as f:
        idx = 0
        for line in f.readlines():
            idx += 1
            ID, Desc, Chinese, Category, Comments = line.strip().split(',')
            if idx == 1:
                continue
            cat2mat[str(ID)] = Category
            if Category in mat2cat:
                mat2cat[Category].append(ID)
            else:
                mat2cat[Category] = [ID]
    return cat2mat, mat2cat


class Ensemble():
    def __init__(self,
                 base_classifiers=None,
                 n_estimators=5,
                 class_weight='balanced',
                 grid_search=True,
                 gs_folds=5):
        self.base_classifiers = base_classifiers.split(',')
        if len(self.base_classifiers) == 1:
            self.n_estimators = n_estimators
        else:
            self.n_estimators = len(self.base_classifiers)
        self.class_weight = class_weight
        self.grid_search = grid_search
        self.gs_folds = gs_folds

    def fit(self, X, y, groups, random_state=1):
        self.fitted_estimators = []
        if len(self.base_classifiers) == 1:
            classifier = self.base_classifiers[0]
            group_kfold = StratifiedGroupKFold(n_splits=self.n_estimators,
                                               random_state=random_state,
                                               shuffle=True)
            qbar = tqdm(group_kfold.split(X, y, groups=groups))
            for k, (train, test) in enumerate(qbar):
                qbar.set_description('Training Estimator {}'.format(k + 1))
                if self.grid_search:
                    estimator = grid_search(classifier.split('_')[0],
                                            X,
                                            y,
                                            self.class_weight,
                                            cv=[(train, test)])
                else:
                    estimator = get_estimator(
                        classifier.split('_')[0], self.class_weight)
                    estimator.set_params(**ref.params[classifier])
                    estimator.fit(X[train], y[train])
                self.fitted_estimators.append(estimator)
        else:
            for classifier in self.base_classifiers:
                if self.grid_search:
                    estimator = grid_search(classifier.split('_')[0],
                                            X,
                                            y,
                                            class_weight=self.class_weight,
                                            cv=self.gs_folds)
                else:
                    estimator = get_estimator(
                        classifier.split('_')[0], self.class_weight)
                    estimator.set_params(**ref.params[classifier])
                    estimator.fit(X, y)
                self.fitted_estimators.append(estimator)

    def refit(self, X, y):
        refitted_estimators = []
        for estimator in self.fitted_estimators:
            estimator.fit(X, y)
            refitted_estimators.append(estimator)
        self.fitted_estimators = refitted_estimators

    def predict(self, X):
        y_preds = []
        for estimator in self.fitted_estimators:
            y_pred = estimator.predict(X)
            y_preds.append(y_pred)
        y_preds = np.array(y_preds).T
        maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)),
                                       axis=1,
                                       arr=y_preds)
        return maj_vote
