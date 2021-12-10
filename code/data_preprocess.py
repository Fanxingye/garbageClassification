import argparse
import os

import numpy as np
import pandas as pd
import ref
from tqdm import tqdm


class DataPreprocess():
    """load and preprocess the garbage datasets:

    1: collect_absorbance
    2: collect_label
    3: collect_desc
    4: collect_boundaries
    """

    def __init__(self,
                 data_dir: str,
                 test: bool,
                 groupbyObjID: bool = True) -> None:
        self.data_dir = data_dir
        self.test = test
        self.groupbyObjID = groupbyObjID

        self.load_ref()

    def collect_absorbance(self) -> pd.DataFrame:
        """collect the absorbance data."""
        data_absorbance_dir = os.path.join(self.data_dir, 'TestRecords')
        list_df_absorbance = []
        total_records = int(len(os.listdir(data_absorbance_dir)))
        with tqdm(total=total_records) as pbar:
            pbar.set_description('Processing TestRecords')
            for idx, filename in enumerate(os.listdir(data_absorbance_dir)):
                if '.' in filename and filename.split('.')[-1] == 'csv':
                    data = pd.read_csv(os.path.join(data_absorbance_dir,
                                                    filename),
                                       skiprows=19)['Absorbance (AU)']
                    df = pd.DataFrame({idx: data}).T
                    df['filename'] = filename
                    list_df_absorbance.append(df)
                pbar.update(idx)
        data = pd.concat(list_df_absorbance, axis=0).reset_index(drop=True)
        feats_names = [
            'absorbance_' + str(i) for i in range(data.shape[1] - 1)
        ] + ['FileName']
        data.columns = feats_names
        return data

    def collect_label(self) -> pd.DataFrame:
        """collect the label data."""
        data_file = os.path.join(self.data_dir, 'MaterialMapping.csv')
        data = pd.read_csv(data_file, dtype={
            'ID': np.uint32
        }).drop(['Chinese', 'Desc', 'Comments'],
                axis=1).rename(columns={'ID': 'Material'})
        data['Category'] = data['Category'].map(self.category2id).astype(
            np.uint8)
        return data

    def collect_desc(self) -> pd.DataFrame:
        """collect the desc data."""
        data_file = os.path.join(self.data_dir, 'TestRecordDesc.csv')
        data = pd.read_csv(data_file,
                           dtype=self.data_types['TestRecordDesc']).rename(
                               columns={'MappingIDCorrect': 'Material'})
        data['FileName'] = data['FileName'].apply(lambda x: x + '.csv')
        return data

    def collect_boundaries(self) -> pd.DataFrame:
        """collect the boundaries."""
        data_file = os.path.join(self.data_dir, 'Boundaries.csv')
        data = pd.read_csv(data_file, dtype=self.data_types['Boundaries'])

        data['ScanChart_split'] = data['ScanChart'].str.split('|').apply(
            lambda row: [float(s) for s in row])
        ScanChart_name_list = [
            'OriginalPointCount', 'ScanYAvg', 'ScanYAvgWeighted',
            'ScanYStdDev', 'ScanMass', 'ScanDensity', 'ScanMassSimple',
            'ScanDensitySimple', 'ScanMassRatio'
        ]
        for idx in range(len(ScanChart_name_list)):
            data[ScanChart_name_list[idx]] = data['ScanChart_split'].apply(
                lambda row: row[idx])

        data['OutlineChart_split'] = data['OutlineChart'].str.split('|').apply(
            lambda row: [float(s) for s in row])

        OutlineChart_name_list = [
            'OutlinePointCount', 'OutlineYAvg', 'OutlineYAvgWeighted',
            'OutlineYStdDev', 'OutlineMass', 'OutlineDensity',
            'OutlineMassSimple', 'OutlineDensitySimple', 'OutlineMassRatio'
        ]

        for idx in range(len(OutlineChart_name_list)):
            data[OutlineChart_name_list[idx]] = data[
                'OutlineChart_split'].apply(lambda row: row[idx])

        data['XChart_split'] = data['XChart'].str.split('|').apply(
            lambda row: [str(s) for s in row])

        XChart_name_list = [
            'MassRatio', 'MassRatioSimple', 'IsOutlineSameAsScan'
        ]

        for idx in range(len(XChart_name_list)):
            data[XChart_name_list[idx]] = data['XChart_split'].apply(
                lambda row: row[idx])

        data[['MassRatio',
              'MassRatioSimple']] = data[['MassRatio',
                                          'MassRatioSimple']].astype('float')
        data['IsOutlineSameAsScan'] = data['IsOutlineSameAsScan'].astype(
            'bool')

        data = data.drop([
            'ScanChart', 'ScanChart_split', 'OutlineChart',
            'OutlineChart_split', 'XChart', 'XChart_split'
        ],
                         axis=1)
        return data

    def load_ref(self) -> None:
        self.category2id = ref.category2id
        self.id2category = ref.id2category
        self.data_types = ref.data_dtype

    def process_data(self) -> None:
        self.data_absorbance = self.collect_absorbance()
        self.data_desc = self.collect_desc()
        self.data_label = self.collect_label()
        self.data_boundaries = self.collect_boundaries()

        self.data_merge1 = pd.merge(self.data_absorbance,
                                    self.data_desc,
                                    on='FileName',
                                    how='inner')
        self.data_merge2 = pd.merge(self.data_merge1,
                                    self.data_label,
                                    on='Material',
                                    how='inner')
        self.data_merge3 = pd.merge(self.data_merge2,
                                    self.data_boundaries,
                                    on='ObjID',
                                    how='inner')

        self.all_embrace = self.data_merge3.drop([
            'LibVer', 'LibModel', 'ProcessBackground', 'MappingIDVoted',
            'MappingIDItem', 'InternalID'
        ],
                                                 axis=1).dropna(axis=0)

        if self.groupbyObjID:
            idx = self.all_embrace.groupby("ObjID")["MaxIntensity"].idxmax()
            self.all_embrace = self.all_embrace.iloc[idx]
            self.all_embrace = self.all_embrace.drop(['FileName'], axis=1)

        if self.test:
            print(self.data_absorbance.head())
            print(self.data_desc.head())
            print(self.data_label.head())
            print(self.data_boundaries.head())
            print('self.data_absorbance Shape: {}'.format(
                self.data_absorbance.shape))
            print('self.data_desc Shape: {}'.format(self.data_desc.shape))
            print('self.data_label Shape: {}'.format(self.data_label.shape))
            print('self.data_boundaries Shape: {}'.format(
                self.data_boundaries.shape))
            print('self.data_merge1 Shape: {}'.format(self.data_merge1.shape))
            print('self.data_merge2 Shape: {}'.format(self.data_merge2.shape))
            print('self.data_merge3 Shape: {}'.format(self.data_merge3.shape))
            print('self.all_embrace Shape: {}'.format(self.all_embrace.shape))
            print('self.all_embrace', self.all_embrace.head())

    def save_data(self) -> None:
        if not self.groupbyObjID:
            file_name = 'AllEmbracingDataset_original.csv'
        else:
            file_name = 'AllEmbracingDataset.csv'
        self.all_embrace.to_csv(os.path.join(self.data_dir, file_name),
                                index=False)

    def run_preprocess(self) -> None:
        self.process_data()
        self.save_data()
        if self.test:
            print('Shape: {}'.format(self.all_embrace.shape))
            print('Material Count: {}'.format(
                self.all_embrace['Material'].nunique()))
            print('Category Count: {}'.format(
                self.all_embrace['Category'].nunique()))

    def reset_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = pd.Index(np.arange(df.shape[1]), dtype=object)
        return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir',
                        default='data/',
                        help='The directory contains data files.')
    parser.add_argument('-test',
                        action='store_true',
                        default=False,
                        help='The option to test the preprocess code.')
    parser.add_argument('-groupbyObjID',
                        action='store_true',
                        default=True,
                        help='The option to groupby the data by ObjID')
    args = parser.parse_args()

    data_preprocess = DataPreprocess(data_dir=args.data_dir,
                                     test=args.test,
                                     groupbyObjID=args.groupbyObjID)

    data_preprocess.run_preprocess()


if __name__ == '__main__':
    main()
