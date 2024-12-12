import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_predict, cross_validate

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

import numpy as np

def train(X_train, y_train, X_val, y_val):
  regr = AdaBoostClassifier(estimator=RandomForestClassifier(max_depth=6, n_estimators=250, criterion='gini', n_jobs=-1, random_state=333), algorithm='SAMME')
  regr.classes_ = [0,1,2,3]
  regr.fit(X_train, y_train)

  pred = regr.predict(X_val)

  cm = metrics.confusion_matrix(y_val, pred, labels=regr.classes_)
  disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=regr.classes_)
  disp.plot()
  plt.savefig("con_matrix.png")

  print("MSE:",metrics.root_mean_squared_error(y_val, pred))

  return regr


def main():
  # processor = Preprocessor("data/test.csv", "data/train.csv")
  # rocessor.visualization()
  # processor.process()

  train_df = pd.read_csv('new_data/new_train_encoded.csv')
  train_df = train_df.drop(columns=['id'])
  test_df = pd.read_csv('new_data/new_test_encoded.csv')
  X = train_df.drop(columns=['sii'])
  y = train_df.pop('sii')

  X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.1)
  model = train(X_train, y_train, X_val, y_val)
  id = test_df.pop('id')
  submission = pd.DataFrame()
  submission['id'] = id
  results = model.predict(test_df)
  submission['sii'] = results.astype(int)
  submission.to_csv('./submission.csv', index=False)

if __name__ == "__main__":
  main()