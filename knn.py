import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

train_data = pd.read_csv("new_data/new_train_encoded.csv")
test_data = pd.read_csv("new_data/new_test_encoded.csv")

X_train = train_data.drop(columns=['id', 'sii'])
y_train = train_data['sii']  # Target variable is the 'sii' column

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_test = test_data.drop(columns=['id'])
X_test_scaled = scaler.transform(X_test)

n_neighbors = 5
knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
knn_model.fit(X_train_scaled, y_train)
predicted = knn_model.predict(X_test_scaled)

submission = pd.DataFrame({
    'id': test_data['id'],
    'sii': predicted
})

submission.to_csv("knn_prediction.csv", index=False)

print("Predictions saved to knn_prediction.csv")
