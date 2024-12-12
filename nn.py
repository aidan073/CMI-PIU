import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Step 1: Load the encoded training and test data
train_data = pd.read_csv("new_data/new_train_encoded.csv")
test_data = pd.read_csv("new_data/new_test_encoded.csv")

# Step 2: Prepare the features and target variable
X_train = train_data.drop(columns=['id', 'sii'])  # Drop 'id' and 'sii' columns from features
y_train = train_data['sii']  # Target variable is the 'sii' column

# Standardize features (important for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

# Prepare the test data
X_test = test_data.drop(columns=['id'])
X_test_scaled = scaler.transform(X_test)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# Step 3: Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)  # First hidden layer with 128 neurons
        self.layer2 = nn.Linear(128, 64)         # Second hidden layer with 64 neurons
        self.layer3 = nn.Linear(64, 4)           # Output layer with 4 classes (since 'sii' ranges from 0-3)

    def forward(self, x):
        x = torch.relu(self.layer1(x))  # Apply ReLU activation function after layer1
        x = torch.relu(self.layer2(x))  # Apply ReLU activation function after layer2
        x = self.layer3(x)              # Output layer (no activation, for raw logits)
        return x

# Step 4: Initialize the model, define loss function, and optimizer
input_dim = X_train.shape[1]  # Number of features
model = NeuralNetwork(input_dim)
criterion = nn.CrossEntropyLoss()  # For classification (multi-class)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Train the model
epochs = 100
batch_size = 64

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train_tensor.size(0))  # Randomize the training data
    running_loss = 0.0
    for i in range(0, X_train_tensor.size(0), batch_size):
        optimizer.zero_grad()

        # Get the current batch
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    if epoch % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(X_train_tensor):.4f}")

# Step 6: Make predictions on the test set
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)  # Get the class with the highest logit value

# Step 7: Save predictions to submission.csv
submission = pd.DataFrame({
    'id': test_data['id'],
    'sii': predicted.numpy()  # Convert predictions to numpy and save
})

submission.to_csv("submission.csv", index=False)

print("Predictions saved to submission.csv")
