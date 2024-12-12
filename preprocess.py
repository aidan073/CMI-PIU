import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from matplotlib.ticker import PercentFormatter
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt

def visualize_class_distribution(target_before, target_after, save_path="visualizations/smote_comparison.png"):
    """
    Visualize class distributions before and after SMOTE.
    Args:
        target_before: Target labels before applying SMOTE.
        target_after: Target labels after applying SMOTE.
        save_path: Path to save the visualization.
    """
    def plot_class_distribution(target, title, ax):
        class_counts = Counter(target)
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        ax.bar(classes, counts, color='coral', alpha=0.7, edgecolor='black')
        ax.set_xticks(classes)
        ax.set_xlabel('Class Labels')
        ax.set_ylabel('Count')
        ax.set_title(title)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    plot_class_distribution(target_before, "Before SMOTE", axes[0])
    plot_class_distribution(target_after, "After SMOTE", axes[1])

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


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
    
    class autoencoder(nn.Module):
        def __init__(self, input_dim, mid_dim, encoding_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, mid_dim),
                nn.ReLU(),
                nn.Linear(mid_dim, encoding_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, mid_dim),
                nn.ReLU(),
                nn.Linear(mid_dim, input_dim),
                nn.Tanh()
            )
        
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    # all data processing (writes new data to new_data folder)
    def process(self, with_autoencoder=False, remove_set_dupes=True)->None:
        scaler = MinMaxScaler(feature_range=(-1, 1))

        """
        Train Set Processing
        """
        # remove train samples that are in test set
        if remove_set_dupes:
            self.train = self.train[~self.train['id'].isin(self.test['id'])]

        # drop PCIAT (not used for training) and Season (useless)
        self.train.drop([col for col in self.train.columns if col.startswith("PCIAT")], axis=1, inplace=True)
        self.train.drop([col for col in self.train.columns if 'Season' in col], axis=1, inplace=True)

        # engineer
        self.train = self._feature_engineering(self.train)
        self.train.replace([np.inf, -np.inf], np.nan, inplace=True)

        threshold = 0.4  # must have 40% of columns
        self.train.dropna(thresh=int(threshold * self.train.shape[1]), inplace=True)

        # track and remove id from train (will be added back after processing)
        id_column_train = self.train.pop('id')

        # track and remove sii column (target variable)
        sii_loc = self.train.columns.get_loc("sii")
        target_column_train = self.train.pop('sii')

        # Check and handle missing values in target
        if target_column_train.isna().any():
            print("Warning: NaN values detected in target variable. Dropping rows with NaN targets.")
            valid_indices = ~target_column_train.isna()  # Identify rows without NaN in the target
            self.train = self.train[valid_indices]
            target_column_train = target_column_train[valid_indices]

        # impute
        imputer = KNNImputer(n_neighbors=5)
        imputed_samples_train = imputer.fit_transform(self.train)

        # scale
        scaled_samples_train = scaler.fit_transform(imputed_samples_train)
        target_column_train = target_column_train.astype(int).to_numpy()

        target_column_train_before_smote = target_column_train.copy()

        # apply SMOTE
        smote = SMOTE(random_state=42)
        scaled_samples_train, target_column_train = smote.fit_resample(scaled_samples_train, target_column_train)

        visualize_class_distribution(target_column_train_before_smote, target_column_train)

        # convert to torch tensors
        final_samples_train = torch.from_numpy(scaled_samples_train).float()
        target_column_train = torch.from_numpy(target_column_train).long()

        """
        Test Set Processing
        """
        # drop season columns
        self.test.drop([col for col in self.test.columns if 'Season' in col], axis=1, inplace=True)

        # engineer
        self.test = self._feature_engineering(self.test)
        self.test.replace([np.inf, -np.inf], np.nan, inplace=True)

        # track and remove id from test (will be added back after processing)
        id_column_test = self.test.pop('id')

        # impute and scale
        self.test.insert(sii_loc, column='sii', value=None)
        test_data = self.test.drop(columns=['sii'])
        imputed_samples_test = imputer.transform(test_data)
        final_samples_test = torch.from_numpy(scaler.transform(imputed_samples_test)).float()


        """
        Autoencoder training for dimensionality reduction
        """
        if with_autoencoder:
            model = self.autoencoder(int(final_samples_test.shape[1]), int(final_samples_test.shape[1]/2), 10)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.005)
            epochs = 100
            batch_size = 100

            # store the loss values for plotting
            train_losses = []
            val_losses = []

            for epoch in range(epochs):
                model.train()
                temp_train_losses = []
                for i in range(0, len(final_samples_train), batch_size):
                    batch = final_samples_train[i:i+batch_size]
                    optimizer.zero_grad()
                    reconstructed = model(batch)
                    train_loss = criterion(reconstructed, batch)
                    train_loss.backward()
                    optimizer.step()
                    temp_train_losses.append(train_loss.item())

                model.eval()
                with torch.no_grad():
                    final_samples_test_reconstructed = model(final_samples_test)
                    val_loss = criterion(final_samples_test_reconstructed, final_samples_test).item()

                train_losses.append(sum(temp_train_losses) / len(temp_train_losses))
                val_losses.append(val_loss)

                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {sum(temp_train_losses) / len(temp_train_losses):.4f}, Val Loss: {val_loss:.4f}")

            plt.figure(figsize=(10, 6))
            plt.plot(range(epochs), train_losses, label='Train Loss', color='blue')
            plt.plot(range(epochs), val_losses, label='Validation Loss', color='red')
            plt.title('Training and Validation Loss Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig("visualizations/autoencoder_training.png", bbox_inches="tight")

            # encode
            with torch.no_grad():
                final_samples_test = model.encoder(final_samples_test).numpy()
                final_samples_train = model.encoder(final_samples_train).numpy()

            test_path = 'new_data/new_test_encoded.csv'
            train_path = 'new_data/new_train_encoded.csv'

        if not with_autoencoder:
            final_samples_test = final_samples_test.numpy()
            final_samples_train = final_samples_train.numpy()
            test_path = 'new_data/new_test.csv'
            train_path = 'new_data/new_train.csv'

        # reconstruct test df and finalize
        final_test = pd.DataFrame(final_samples_test)
        final_test.insert(0, 'id', id_column_test)
        final_test.to_csv(test_path, index=False)

        # reconstruct train df and finalize
        final_train = pd.DataFrame(final_samples_train)
        final_train.insert(0, 'id', id_column_train.reset_index(drop=True))
        final_train.insert(1, 'sii', target_column_train.numpy())
        final_train.to_csv(train_path, index=False)