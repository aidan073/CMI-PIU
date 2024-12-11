import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from matplotlib.ticker import PercentFormatter

class Preprocessor:
    def __init__(self, test_path, train_path):
        self.test = pd.read_csv(test_path)
        self.train = pd.read_csv(train_path)

    # pure visualization intended for training set, does not modify training set
    def visualization(self):
        usable = self.train[self.train['sii'].notnull()] # remove any samples with no sii

        # visualize nulls per column
        missing_count = usable.isnull().sum().reset_index().rename(columns={0: 'null_count', 'index': 'feature'}).sort_values('null_count', ascending=False).assign(null_ratio=lambda df: df['null_count'] / len(usable))
        plt.figure(figsize=(6, 15))
        plt.title(f'Missing values over the {len(usable)} samples which have a target')
        plt.barh(np.arange(len(missing_count)), (missing_count['null_ratio']), color='coral', label='missing')
        plt.barh(np.arange(len(missing_count)), 
                1 - missing_count['null_ratio'],
                left=missing_count['null_ratio'],
                color='darkseagreen', label='available')
        plt.yticks(np.arange(len(missing_count)), missing_count['feature'])
        plt.gca().xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        plt.xlim(0, 1)
        plt.legend()
        plt.savefig("visualizations/null_counts.png", bbox_inches="tight")

        # correlation matrix between features and PCIAT total (pretty much target variable)
        plt.figure(figsize=(14, 12))
        corr_matrix = usable[[
            'PCIAT-PCIAT_Total', 'Basic_Demos-Age', 'Basic_Demos-Sex', 'Physical-BMI', 
            'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference',
            'Physical-Diastolic_BP', 'Physical-Systolic_BP', 'Physical-HeartRate',
            'PreInt_EduHx-computerinternet_hoursday', 'SDS-SDS_Total_T', 'PAQ_A-PAQ_A_Total',
            'PAQ_C-PAQ_C_Total', 'Fitness_Endurance-Max_Stage', 'Fitness_Endurance-Time_Mins','Fitness_Endurance-Time_Sec',
            'FGC-FGC_CU', 'FGC-FGC_GSND','FGC-FGC_GSD','FGC-FGC_PU','FGC-FGC_SRL','FGC-FGC_SRR','FGC-FGC_TL','BIA-BIA_Activity_Level_num', 
            'BIA-BIA_BMC', 'BIA-BIA_BMI', 'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 'BIA-BIA_FFM',
            'BIA-BIA_FFMI','BIA-BIA_FMI', 'BIA-BIA_Fat','BIA-BIA_Frame_num','BIA-BIA_ICW','BIA-BIA_LDM','BIA-BIA_LST',
            'BIA-BIA_SMM','BIA-BIA_TBW']].corr()

        sii_corr = corr_matrix['PCIAT-PCIAT_Total'].drop('PCIAT-PCIAT_Total')
        filtered_corr = sii_corr[(sii_corr > 0.1) | (sii_corr < -0.1)]
        plt.figure(figsize=(8, 6))
        filtered_corr.sort_values().plot(kind='barh', color='coral')
        plt.title('Features with Correlation > 0.1 or < -0.1 with PCIAT-PCIAT_Total')
        plt.xlabel('Correlation coefficient')
        plt.ylabel('Features')
        plt.savefig("visualizations/correlation_matrix.png", bbox_inches="tight")

    # feature engineering
    def _feature_engineering(self, df):
        df['BMI_Age'] = df['Physical-BMI'] * df['Basic_Demos-Age']
        df['Internet_Hours_Age'] = df['PreInt_EduHx-computerinternet_hoursday'] * df['Basic_Demos-Age']
        df['BMI_Internet_Hours'] = df['Physical-BMI'] * df['PreInt_EduHx-computerinternet_hoursday']
        df['BFP_BMI'] = df['BIA-BIA_Fat'] / df['BIA-BIA_BMI']
        df['FFMI_BFP'] = df['BIA-BIA_FFMI'] / df['BIA-BIA_Fat']
        df['FMI_BFP'] = df['BIA-BIA_FMI'] / df['BIA-BIA_Fat']
        df['LST_TBW'] = df['BIA-BIA_LST'] / df['BIA-BIA_TBW']
        df['BFP_BMR'] = df['BIA-BIA_Fat'] * df['BIA-BIA_BMR']
        df['BFP_DEE'] = df['BIA-BIA_Fat'] * df['BIA-BIA_DEE']
        df['BMR_Weight'] = df['BIA-BIA_BMR'] / df['Physical-Weight']
        df['DEE_Weight'] = df['BIA-BIA_DEE'] / df['Physical-Weight']
        df['SMM_Height'] = df['BIA-BIA_SMM'] / df['Physical-Height']
        df['Muscle_to_Fat'] = df['BIA-BIA_SMM'] / df['BIA-BIA_FMI']
        df['Hydration_Status'] = df['BIA-BIA_TBW'] / df['Physical-Weight']
        df['ICW_TBW'] = df['BIA-BIA_ICW'] / df['BIA-BIA_TBW']

        return df

    # all data processing (writes new data to new_data folder)
    def process(self)->None:

        # *train processing*

        # drop PCIAT (not used for training) and Season (useless)
        self.train.drop([col for col in self.train.columns if col.startswith("PCIAT")], axis=1, inplace=True)
        self.train.drop([col for col in self.train.columns if 'Season' in col], axis=1, inplace=True)

        # engineer
        self.train = self._feature_engineering(self.train)
        self.train.replace([np.inf, -np.inf], np.nan, inplace=True)

        threshold = 0.4 # must have 40% of columns
        self.train.dropna(thresh=int(threshold * self.train.shape[1]), inplace=True)

        # track and remove id from train (will be added back after proccessing)
        id_column_train = self.train.pop('id')

        # imputer
        imputer_train = KNNImputer(n_neighbors=5)
        imputed_samples = imputer_train.fit_transform(self.train)
        
        # reconstruct train df and finalize
        imputed_train = pd.DataFrame(imputed_samples, columns=self.train.columns, index=self.train.index)
        imputed_train.insert(0, 'id', id_column_train)
        imputed_train.insert(1, 'sii', imputed_train.pop('sii'))
        imputed_train.to_csv('new_data/new_train.csv', index=False)

        # *test processing*

        # drop season columns
        self.test.drop([col for col in self.test.columns if 'Season' in col], axis=1, inplace=True)

        # engineer
        self.test = self._feature_engineering(self.test)
        self.test.replace([np.inf, -np.inf], np.nan, inplace=True)

        # track and remove id from test (will be added back after proccessing)
        id_column_test = self.test.pop('id')

        # imputer
        imputer_test = KNNImputer(n_neighbors=2)
        imputed_samples = imputer_test.fit_transform(self.test)

        # reconstruct test df and finalize
        imputed_test = pd.DataFrame(imputed_samples, columns=self.test.columns, index=self.test.index)
        imputed_test.insert(0, 'id', id_column_test)
        imputed_test.to_csv('new_data/new_test.csv', index=False)