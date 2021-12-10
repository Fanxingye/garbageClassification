import numpy as np

params = {
    'svm_cat': {
        'C': 10000.0,
        'gamma': 0.1,
        'tol': 1e-3,
        'probability': True,
        'shrinking': True,
        'class_weight': 'balanced'
    },
    'svm_mat': {
        'C': 10000.0,
        'gamma': 0.1,
        'tol': 1e-3,
        'probability': True,
        'shrinking': True,
        'class_weight': 'balanced'
    },
    'randomforest_cat': {
        'n_estimators': 200
    },
    'randomforest_mat': {
        'n_estimators': 200
    },
    'xgboost_cat': {
        'objective': 'multi:softmax',
        'num_class': 4,
        # 'n_estimators': 30,
        # 'max_depth': 4,
        # 'subsample': 0.5,
        'gamma': 0.01,
        'eta': 0.01
    },

    'xgboost_mat': {
        'objective': 'multi:softmax',
        'num_class': 136,
        # 'n_estimators': 30,
        # 'max_depth': 34,
        # 'subsample': 1.0,
        'gamma': 0.01,
        'eta': 0.3
    },
    'lightgbm_cat': {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 4,
        'metric': 'multi_logloss',
        'verbose': -1,
        'num_leaves': 64,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'subsample_freq': 1,
        'learning_rate': 0.01,
        'n_jobs': -1,
        'num_boost_round': 30,
        'verbosity': 1
    },
    'lightgbm_mat': {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 136,
        'metric': 'multi_logloss',
        'verbose': -1,
        'num_leaves': 64,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'subsample_freq': 1,
        'learning_rate': 0.01,
        'num_boost_round': 15,
        'n_jobs': -1,
        'verbosity': 1
    }
}

gpu_params = {
    'tree_method': 'gpu_hist',
    'updater': 'grow_gpu_hist,prune',
    'gpu_id': 0
}

grid = {
    'svm_cat': {
        'C': np.logspace(4, 6, 5),
        'gamma': np.logspace(-2, 0, 3),
        'tol': [1e-3, 1e-6],
        'shrinking': [True],
        'probability': [True]
    },
    'svm_mat': {
        'C': np.logspace(4, 6, 5),
        'gamma': np.logspace(-2, 0, 3),
        'tol': [1e-3, 1e-6],
        'shrinking': [True],
        'probability': [True]
    },
    'xgboost_cat': {
        'objective': ['multi:softmax'],
        'num_class': [4],
        'n_estimators': [10, 30, 100],
        'max_depth': [4, 6, 8],
        'subsample': [0.5, 0.8, 1.0],
        'gamma': [0.01, 0.1, 1.0],
        'eta': [0.1, 0.3, 1.0],
    },
    'xgboost_mat': {
        'objective': ['multi:softmax'],
        'num_class': [86],
        'n_estimators': [10, 30, 100],
        'max_depth': [4, 6, 8],
        'subsample': [0.5, 0.8, 1.0],
        'gamma': [0.01, 0.1, 1.0],
        'eta': [0.1, 0.3, 1.0],
    }
}

class_weight = 'balanced'

data_dtype = {
    'TestRecordDesc': {
        'ObjID': np.uint64,
        'SamplingPointID': np.uint32,
        'MappingIDCorrect': np.uint32,
        'MappingIDVoted': np.uint32,
        'MappingIDItem': np.uint32,
        'Internal': np.uint32,
    },
    'Boundaries': {
        'ObjID': np.uint64,
        'Direction': np.str,
        'Length': np.int,
        'ScanChar': np.str,
        'OutlineChar': np.str,
        'XChart': np.str
    },
    'AllEmbrace': {
        'FileName': np.str,
        'ObjID': np.uint64,
        'SamplingPointID': np.uint32,
        'Material': np.uint32,
        'Category': np.uint8,
        'Direction': np.str,
        'Length': np.uint32,
        'OriginalPointCount': np.float,
        'ScanYAvg': np.float,
        'ScanYAvgWeighted': np.float,
        'ScanYStdDev': np.float,
        'ScanMass': np.float,
        'ScanDensity': np.float,
        'ScanMassSimple': np.float,
        'ScanDensitySimple': np.float,
        'ScanMassRatio': np.float,
        'OutlinePointCount': np.float,
        'OutlineYAvg': np.float,
        'OutlineYAvgWeighted': np.float,
        'OutlineYStdDev': np.float,
        'OutlineMass': np.float,
        'OutlineDensity': np.float,
        'OutlineMassSimple': np.float,
        'OutlineDensitySimple': np.float,
        'OutlineMassRatio': np.float,
        'MassRatio': np.float,
        'MassRatioSimple': np.float,
        'IsOutlineSameAsScan': bool
    }
}

category2id = {'HAZARDOUS': 0, 'RECYCLABLE': 1, 'HOUSEHOLD': 2, 'RESIDUAL': 3}

id2category = {0: 'HAZARDOUS', 1: 'RECYCLABLE', 2: 'HOUSEHOLD', 3: 'RESIDUAL'}

id_background_zero = 563

# only focus the 50 class
select_50_material = [
    203, 559, 207, 562, 244, 584, 583, 4, 3, 102, 80, 205, 338, 586, 255, 257,
    343, 346, 539, 139, 187, 188, 561, 2, 223, 222, 570, 1, 468, 567, 582, 588,
    330, 14, 219, 221, 7, 8, 383, 217, 51, 392, 247, 564, 249, 569, 281, 163,
    142, 336
]
