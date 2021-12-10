## 垃圾分类模型使用说明

### 1.包含以下数据文件

| 文件                     | 描述                                         |
| ------------------------ | -------------------------------------------- |
| data/MaterialMapping.csv | 物体以及其归类的信息                         |
| data/TestRecords         | 光谱原始测试数据 CSV 文件                    |
| data/TestRecordDesc.zip  | CSV 文件描述文件                             |
| data/Boundaries.csv      | 物体轮廓信息                                 |

### 2.**包含以下模型文件**

| 文件夹            | 描述                                                         |
| ----------------- | ------------------------------------------------------------ | 
| output/Category/ | 包含预测大类别的分类模型 |
| output/Material/            | 包含预测大类别（4类）的分类模型              |
| output/Backgroud/      | 包含预测小类别（50类）的分类模型  |


### 3.环境配置

&emsp; 进入`garbage`路径，在anaconda命令行运行`pip install -r requirements.txt`

### 4.**数据预处理**

&emsp; 在`anaconda`命令行运行`python data_preprocess.py`，即可在data文件夹中生成`AllEmbracingDataset.csv`。若将来更新数据，按照和原来相同的格式和路径保存在data文件夹中，即可用`data_preprocess.py`生成更新后的数据集


- 运行数据预处理Python脚本，将上述数据的信息集合到一个数据文件中

```shell
python code/data_preprocess.py -data_dir D:/datasets/garbage \
                        -test \
                        -groupbyObjID
```

运行脚本生成的数据文件 `datasets/AllEmbracingDataset.csv`  数据集


### 5.模型训练Python脚本

```
python code/train_gbdt_lr.py -data_dir D:/datasets/garbage/ \
                    -use_groupbyID True \
                    -output_dir output/ \
                    -skip_data_preprocess
```

其他 `Python`脚本说明：
- feature_engineering.py 特征工程代码
- ref.py	数据处理和模型推理所需的配置文件
- utils.py	数据处理所需的一些函数
- gbdt_feature.py   用gbdt模型生成特征


### 6.模型推理Python脚本

```
python code/predict_gbdt_lr.py -data_dir D:/datasets/garbage/ \
                    -use_groupbyID True \
                    -output_dir output/ \
                    -skip_data_preprocess \
                    -save_dir output/ 
```

&emsp; 注1：只要同一个`ObjID`的多条数据的预测结果有一个不是背景零，最终预测结果就不是背景零。

&emsp; 注2：预测出的Material只会是在训练数据中出现过的唯一标记号。这次数据中不同的唯一标记号共有148个，具体可参见output/log/log.txt中的`LabelEncoder.classes`

- 预测结果文件(predictions.csv)说明：对每个物体（即每个ObjID，通常对应多条测试记录）给出多个预测结果汇总后的预测结果。

| **#** | **域名**      | **意义**                                     |
| ----- | ------------- | -------------------------------------------- |
| **1** | ObjID         | 被测物体唯一标记。同一物体会对应多条测试记录 |
| **2** | Category      | 物体分类，从训练数据中获取                   |
| **3** | Material      | 物体对应的唯一标识号，从训练数据中获取       |
| **4** | pred_Category | 模型所预测出的物体分类                       |
| **5** | pred_Material | 模型所预测出的物体唯一标识号                 |
| **6** | pred_background | 模型预测的背景和物体 （背景标记为 0，物体标记为 1） |
| **7** | pred_Category_final | 模型所预测出的物体分类                |
| **8** | pred_Material_final | 模型所预测出的物体材料分类 | 

### 7. 模型精度
&emsp; 对于Category、Material和Background三种场景的预测，我们均使用GBDT+LR模型。尝试过SVM、XGBoost、LightGBM和GBDT+LR模型，对比之下，GBDT+LR模型表现最好。
&emsp; 在测试集上的Accuracy如下：
| **场景** | **Accuracy** |
| ----- | ----- |
| Category | 0.7583130575831306 |
| Material | 0.6042173560421735 |
| Background | 0.996044825313118 |