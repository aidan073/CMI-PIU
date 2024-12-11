import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from preprocess import Preprocessor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt



def RF_model_optim(X_train, y_train, X_val, y_val):
  params = {'n_estimators': 48, 'max_depth': 4, 'min_samples_split': 25, 'min_samples_leaf': 5, 'max_features': 'sqrt', 'criterion': 'poisson', 'n_jobs': -1}
  RF_Clfr = RandomForestRegressor(**params)
  
  path = RF_Clfr.cost_complexity_pruning_path(X_train, y_train)
  ccp_alphas, impurities = path.ccp_alphas, path.impurities

  clfs = []
  for ccp_alpha in ccp_alphas:
    clf = RandomForestRegressor(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
    
  print(f"Number of nodes in the last tree is: {clfs[-1].tree_.node_count} with ccp_alpha: { ccp_alphas[-1]}")

  clfs = clfs[:-1]
  ccp_alphas = ccp_alphas[:-1]

  node_counts = [clf.tree_.node_count for clf in clfs]
  depth = [clf.tree_.max_depth for clf in clfs]
  fig, ax = plt.subplots(2, 1)
  ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
  ax[0].set_xlabel("alpha")
  ax[0].set_ylabel("number of nodes")
  ax[0].set_title("Number of nodes vs alpha")
  ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
  ax[1].set_xlabel("alpha")
  ax[1].set_ylabel("depth of tree")
  ax[1].set_title("Depth vs alpha")
  fig.tight_layout()
  plt.savefig("./ccp_alphas.png")


  train_scores = [clf.score(X_train, y_train) for clf in clfs]
  test_scores = [clf.score(X_val, y_val) for clf in clfs]

  fig, ax = plt.subplots()
  ax.set_xlabel("alpha")
  ax.set_ylabel("accuracy")
  ax.set_title("Accuracy vs alpha for training and testing sets")
  ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
  ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
  ax.legend()
  plt.savefig("./accuracy.png")

  # RF_Clfr.fit(X_train, y_train)
  # predictions = RF_Clfr.predict(X_val)
  # return score

def main():
  # processor = Preprocessor("data/test.csv", "data/train.csv")
  # rocessor.visualization()
  # processor.process()

  train_df = pd.read_csv('new_data/new_train.csv')
  test_df = pd.read_csv('new_data/new_test.csv')
  y = train_df.pop('sii')

  X_train,X_val,y_train,y_val=train_test_split(train_df,y,test_size=0.1)

  RF_model_optim(X_train, y_train, X_val, y_val)

if __name__ == "__main__":
  main()