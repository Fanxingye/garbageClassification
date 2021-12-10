# Garbage Classification

[toc]
## 1.包含以下数据文件

| 文件                     | 描述                      |
| ------------------------ | ------------------------- |
| data/MaterialMapping.csv | 物体以及其归类的信息      |
| data/TestRecords         | 光谱原始测试数据 CSV 文件 |
| data/TestRecordDesc.zip  | CSV 文件描述文件          |
| data/Boundaries.csv      | 物体轮廓信息              |

## 2. 光谱数据可视化
### No wavelets_transform
![](./output/no_wavelets_transform/cat.png)

### Use wavelets_transform
![](./output/wavelets_transform/cat.png)

### wavelets_transform standard scale
![](./output/wavelets_transform_standard/cat.png)


### HAZARDOUS

易混淆类品: RECYCLABLE, RESIDUAL

![](./output/wavelets_transform_standard/0_cat.png)
![](./output/wavelets_transform_standard/1_cat.png)
![](./output/wavelets_transform_standard/2_cat.png)



### RECYCLABLE

易混淆类品: RESIDUAL

![](./output/wavelets_transform_standard/0_cat.png)
![](./output/wavelets_transform_standard/3_cat.png)
![](./output/wavelets_transform_standard/4_cat.png)
### HOUSEHOLD

易混淆类品: RESIDUAL

![](./output/wavelets_transform_standard/1_cat.png)
![](./output/wavelets_transform_standard/3_cat.png)
![](./output/wavelets_transform_standard/5_cat.png)

### RESIDUAL

易混淆类品: RECYCLABLE

![](./output/wavelets_transform_standard/2_cat.png)
![](./output/wavelets_transform_standard/4_cat.png)
![](./output/wavelets_transform_standard/5_cat.png)

## 3. 数据处理和模型分析
### 3.1 数据预处理(一)

- 合并所有数据集
- 1. 光谱数据小波变换
- 2. 光谱数据 标准化
- 3. 形状数据 标准化
- 4. 对标签进行 LabelEncoder
### 3.2 模型结果分析(一)

- SVM model:
- Test Accuracy of Category Classifier: 0.6978496599889726
- Confusion matrix, without normalization:

|            | HAZARDOUS | RECYCLABLE | HOUSEHOLD | RESIDUAL | Count | Presion |
| ---------- | :-------: | :--------: | :-------: | :------: | :---: | :-----: |
| HAZARDOUS  |    174    |     71     |     3     |    89    |  337  |  0.516  |
| RECYCLABLE |    60     |    1345    |    24     |   545    | 1974  | 0.6813  |
| HOUSEHOLD  |     4     |     61     |    713    |   115    |  893  | 0.7984  |
| RESIDUAL   |    73     |    525     |    74     |   1565   | 2237  | 0.6995  |
| **Count**  |    311    |    2002    |    814    |   2314   |  xxx  |   xxx   |
| Recall     |   0.559   |   0.671    |  0.87592  |  0.6763  |  xxx  |   xxx   |

category2id

|            | HAZARDOUS |
| ---------- | :-------: |
| HAZARDOUS  |     1     |
| RECYCLABLE |     2     |
| HOUSEHOLD  |     3     |
| RESIDUAL   |     4     |

### 3.3 数据预处理(二)

- 合并所有数据集
- 1. 光谱数据小波变换
- 2. 光谱数据 标准化
- 3. 形状数据 标准化
- 4. 对标签进行 LabelEncoder

发现 光谱数据对同一个 Object 进行了多次测量, 原本只有 6019 个 Object, 合并之后产生 29000 多条记录. 改进策略:
- 将同一 Object 的数据合并, 用均值代表之前的多条记录.
- 训练测试集划分准则: 根据 object 不同随机进行划分.

### 3.2 模型结果分析(二)

#### SVM
##### Model A

- SVM model:
- Test Accuracy of Category Classifier: 0.619
- Confusion matrix, without normalization:
```sh
Test Accuracy of Category Classifier: 0.619
              precision    recall  f1-score   support

   HAZARDOUS       0.39      0.35      0.37       124
  RECYCLABLE       0.52      0.56      0.54       376
   HOUSEHOLD       0.85      0.79      0.82       201
    RESIDUAL       0.66      0.66      0.66       503

    accuracy                           0.62      1204
   macro avg       0.61      0.59      0.60      1204
weighted avg       0.62      0.62      0.62      1204

Confusion matrix, without normalization
[[ 44  46   2  32]
 [ 39 212   8 117]
 [  3  20 159  19]
 [ 27 129  17 330]]

```

##### Model B

- SVM model:
- Test Accuracy of Category Classifier: 0.646
- Confusion matrix, without normalization:
```sh
Test Accuracy of Category Classifier: 0.646
              precision    recall  f1-score   support

   HAZARDOUS       0.50      0.37      0.43       124
  RECYCLABLE       0.63      0.62      0.62       376
   HOUSEHOLD       0.84      0.67      0.75       201
    RESIDUAL       0.62      0.73      0.67       503

    accuracy                           0.65      1204
   macro avg       0.65      0.60      0.62      1204
weighted avg       0.65      0.65      0.64      1204

Confusion matrix, without normalization
[[ 46  30   3  45]
 [ 22 232   7 115]
 [  1   6 135  59]
 [ 23  99  16 365]]
```
##### Model C
- SVM model:
- Test Accuracy of Category Classifier: 0.477
- Confusion matrix, without normalization:

```sh
Test Accuracy of Category Classifier: 0.477
              precision    recall  f1-score   support

   HAZARDOUS       0.32      0.41      0.36       124
  RECYCLABLE       0.50      0.60      0.54       376
   HOUSEHOLD       0.38      0.32      0.35       201
    RESIDUAL       0.55      0.47      0.50       503

    accuracy                           0.48      1204
   macro avg       0.44      0.45      0.44      1204
weighted avg       0.48      0.48      0.48      1204

Confusion matrix, without normalization
[[ 51  17  24  32]
 [ 34 224  20  98]
 [ 19  56  65  61]
 [ 56 150  63 234]]
```