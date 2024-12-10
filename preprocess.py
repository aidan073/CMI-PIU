import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from matplotlib.ticker import PercentFormatter

class Preprocessor:
    def __init__(self, test_path, train_path):
        self.test = pd.read_csv(test_path)
        self.train = pd.read_csv(train_path)

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

    def feature_engineering(self):
        pass
    def fill_nulls(self):
        imputer = KNNImputer(n_neighbors=5)
        transformed_samples = imputer.fit_transform(self.train)
        
        return