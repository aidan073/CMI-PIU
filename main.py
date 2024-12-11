import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from preprocess import Preprocessor
from sklearn.ensemble import RandomForestRegressor

processor = Preprocessor("data/test.csv", "data/train.csv")
#processor.visualization()
processor.process()

def RF_model(X_train, y_train, X_val, y_val, threshold):
    params = {'n_estimators': 48, 'max_depth': 4, 'min_samples_split': 25, 'min_samples_leaf': 5, 'max_features': 'sqrt', 'criterion': 'poisson'}

    RF_Clfr = RandomForestRegressor(**params)
    RF_Clfr.fit(X_train, y_train)
    
    predictions = RF_Clfr.predict(X_val)
    path = RF_Clfr.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = RandomForestRegressor(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)
    print(f"Number of nodes in the last tree is: {clfs[-1].tree_.node_count} with ccp_alpha: { ccp_alphas[-1]}")

    return score

def main():
  train_df = pd.read_csv('new_data/new_train.csv')
  test_df = pd.read_csv('new_data/new_test.csv')
  y = train_df.pop('sii')

  X_train,X_val,y_train,y_val=train_test_split(train_df,y,test_size=0.1)

