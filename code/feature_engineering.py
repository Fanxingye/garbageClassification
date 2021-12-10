import pandas as pd
<<<<<<< HEAD
=======
import numpy as np
import ref
>>>>>>> c3ca679cd70ab23bd47dc17139a63c2c82dec2d7


def feature_engineering(data : pd.DataFrame, label_columns = None):
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

    print(X_wavelength_columns.remove('MaxIntensity'))
    data["absorbance_mean"] = data[X_wavelength_columns].mean(axis = 1)
    data["absorbance_std"] = data[X_wavelength_columns].std(axis = 1)
    data["absorbance_min"] = data[X_wavelength_columns].min(axis = 1)
    data["absorbance_max"] = data[X_wavelength_columns].max(axis = 1)
    data["absorbance_skew"] = data[X_wavelength_columns].skew(axis = 1)
    data["absorbance_kurtosis"] = data[X_wavelength_columns].kurtosis(axis = 1)
    data[X_wavelength_columns] = data[X_wavelength_columns].apply(
        lambda x: (x - data["absorbance_mean"]) / data["absorbance_std"])
    X_wavelength_columns = X_wavelength_columns + ['MaxIntensity', "absorbance_mean", "absorbance_std",
        "absorbance_min", "absorbance_max", "absorbance_skew", "absorbance_kurtosis"]
    X_wavelength = data[X_wavelength_columns].reset_index(drop=True)
    X_shape = data[X_shape_columns].reset_index(drop=True)

    X = pd.concat((X_wavelength, X_shape), axis=1)
    if label_columns is not None:
        y = data[label_columns]
        return X, y
    else:
        return X
