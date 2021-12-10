#！/bin/sh
python code/data_preprocess.py -data_dir D:/datasets/garbage \
                        -test

python code/data_preprocess.py -data_dir D:/datasets/garbage \
                        -test \
                        -groupbyObjID


python code/train_background_model.py \
    -data_dir D:/datasets/garbage \
    -use_groupbyID True \
    -output_dir output3/background_model/ \
    -train_material_classifier   \
    -wavelets_transform \
    -use_standardScaler  \
    -use_labelencoder  \
    -train_without_shape \
    -skip_data_preprocess

## Model A
python code/train_model.py -data_dir D:/datasets/garbage \
                    -use_groupbyID True \
                    -output_dir output3/modelA/ \
                    -train_material_classifier \
                    -kfold \
                    -wavelets_transform \
                    -use_standardScaler  \
                    -use_labelencoder  \
                    -train_without_shape  \
                    -skip_data_preprocess

python code/train_model.py -data_dir D:/datasets/garbage \
                    -use_groupbyID True \
                    -base_classifiers_cat xgboost_cat \
                    -base_classifiers_mat xgboost_mat  \
                    -wavelets_transform \
                    -output_dir output3/modelA/ \
                    -train_material_classifier \
                    -use_standardScaler  \
                    -use_labelencoder  \
                    -train_without_shape  \
                    -skip_data_preprocess

python code/train_model.py -data_dir D:/datasets/garbage \
                    -use_groupbyID True \
                    -base_classifiers_cat lightgbm_cat \
                    -base_classifiers_mat lightgbm_mat  \
                    -wavelets_transform \
                    -kfold \
                    -output_dir output3/modelA/ \
                    -train_material_classifier \
                    -use_standardScaler  \
                    -use_labelencoder  \
                    -train_without_shape  \
                    -skip_data_preprocess

python code/train_model.py -data_dir D:/datasets/garbage \
                    -use_groupbyID True \
                    -base_classifiers_cat lightgbm_cat \
                    -base_classifiers_mat lightgbm_mat  \
                    -output_dir output3/modelA/ \
                    -train_material_classifier \
                    -use_standardScaler  \
                    -use_labelencoder  \
                    -train_without_shape  \
                    -skip_data_preprocess

# lightgbm
python code/train_model_lgb_mat.py -data_dir D:/datasets/garbage \
                    -use_groupbyID True \
                    -output_dir output3/modelA/ \
                    -train_material_classifier \
                    -wavelets_transform \
                    -use_standardScaler  \
                    -use_labelencoder  \
                    -train_without_shape  \
                    -skip_data_preprocess


## Model B
python train_model2.py  -data_dir /media/robin/DATA/datatsets/garbagedata/data_v5/data \
                    -data_version 3 \
                    -output_dir output/modelB/ \
                    -train_material_classifier \
                    -use_standardScaler  \
                    -use_labelencoder  \
                    -skip_data_preprocess

## Model C
python train_model2.py  -data_dir /media/robin/DATA/datatsets/garbagedata/data_v5/data \
                    -data_version 3 \
                    -output_dir output/modelC/ \
                    -train_material_classifier \
                    -use_standardScaler  \
                    -use_labelencoder  \
                    -train_without_wavelength \
                    -skip_data_preprocess

## run predict
python run_model_boost.py -data_dir /media/robin/DATA/datatsets/garbagedata/data_v5/data \
                    -output_dir output \
                    -data_version 3 \
                    -save_dir output \
                    -review_background_zero  \
                    -skip_data_preprocess


### run model predict
### modelA
python code/run_model.py \
        -data_dir D:/datasets/garbage \
        -output_dir output3/modelA/ \
        -output_background_dir output3/background_model/ \
        -save_dir output3/modelA/test_result/  \
        -skip_data_preprocess

#### modelB
python run_model.py \
        -data_dir /media/robin/DATA/datatsets/garbagedata/data_v5/data \
        -output_dir output/modelB/ \
        -output_background_dir output/background_model/ \
        -save_dir output/modelB/test_result  \
        -model_version 3 \
        -data_version 3 \
        -skip_data_preprocess

#### modelC
python run_model.py \
        -data_dir /media/robin/DATA/datatsets/garbagedata/data_v5/data \
        -output_dir output/modelC/  \
        -output_background_dir output/background_model/ \
        -save_dir output/modelC/test_result/ \
        -model_version 3 \
        -data_version 3 \
        -skip_data_preprocess

#### run boosting
python run_model_boost.py -data_dir /media/robin/DATA/datatsets/garbagedata/data_v5/data \
            -output_dir output \
            -data_version 3 \
            -save_dir output \
            -review_background_zero  \
            -skip_data_preprocess

### train GBDT+LR
python code/train_gbdt_lr.py -data_dir D:/datasets/garbage \
                    -use_groupbyID True \
                    -output_dir output3/model_cat/ \
                    -use_labelencoder  \
                    -skip_data_preprocess