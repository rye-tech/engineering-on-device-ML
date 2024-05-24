import numpy as np
import torch
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import torch.nn.functional as F

from torchinfo import summary

# -----------------------Begin Data Prep ------------------------------------

# Define data directories
train_data_dir = "Training Data"
test_data_dir = "Testing Data"

def load_trial(file_path, expected_num_columns=None):
    data = pd.read_csv(file_path, dtype=float)
    label = os.path.splitext(os.path.basename(file_path))[0]  # Remove extension to get label
    data_array = data.to_numpy()  # Extract numerical features
    
    # Ensure all rows have the same number of columns
    if expected_num_columns is not None and data_array.shape[1] != expected_num_columns:
        print(f"Inconsistent number of columns in file {file_path}: {data_array.shape[1]} (expected {expected_num_columns})")
        return None, None
    
    # Convert label to integer or other format as needed
    try:
        label = int(label)
    except ValueError:
        print(f"Invalid label {label} in file {file_path}")
        return None, None
    
    # Map labels to 5 classes
    class_index = (label - 11) // 10  # Adjusted to match the class ranges correctly
    
    print(f"Label: {label}, Class Index: {class_index}")

    data_tensor = torch.tensor(data_array, dtype=torch.float32)
    label_tensor = torch.tensor(class_index, dtype=torch.long)
    return data_tensor, label_tensor

# Create training and testing datasets
class MovementDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = []
        self.expected_num_columns = None
        self.labels_set = set()
        
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    data, label = load_trial(file_path, self.expected_num_columns)
                    if data is not None and label is not None:
                        if self.expected_num_columns is None:
                            self.expected_num_columns = data.shape[1]
                        self.samples.append((data, label))
                        self.labels_set.add(label.item())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data, label = self.samples[idx]
        data = data.t()
        return data, label

# Debugging function to print tensor shapes
def print_tensor_shapes(dataset):
    for i in range(len(dataset)):
        data, label = dataset[i]
        #print(f"Sample {i}: Data shape = {data.shape}, Label = {label}")

# Create training and testing datasets
train_data = MovementDataset(train_data_dir)
test_data = MovementDataset(test_data_dir)

# Print shapes for debugging
print("Training data shapes:")
print_tensor_shapes(train_data)

print("Testing data shapes:")
print_tensor_shapes(test_data)

# Determine the number of classes
num_classes = max(train_data.labels_set) + 1
print(f"Number of classes: {num_classes}")

# Print unique labels for debugging
print("Unique labels in training data:", train_data.labels_set)

# Create DataLoaders for training and testing
train_loader = DataLoader(train_data, batch_size=500, shuffle=True)
test_loader = DataLoader(test_data, batch_size=500, shuffle=False)

# Iterate through the DataLoader to trigger printing within __getitem__
for data, label in train_loader:
    pass  # No need to modify data (printing happens in __getitem__)

for data, label in test_loader:
    pass  # No need to modify data (printing happens in __getitem__)

# ----------------------- End Data Prep --------------------------------

# -------------------------- Begin Model Development ----------------------------------------------------

class SimpleCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size of the flattened tensor
        self.flattened_size = self._calculate_flattened_size(input_channels)
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def _calculate_flattened_size(self, input_channels):
        # Create a dummy input tensor with a sequence length of 100 (adjust as needed)
        x = torch.zeros(1, input_channels, 749)
        x = self._forward_features(x)
        flattened_size = x.view(-1).shape[0]
        #print(f'Flattened size: {flattened_size}')
        return flattened_size

    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        #print(f'After conv1: {x.shape}')
        x = self.pool(x)
        #print(f'After pool1: {x.shape}')
        x = F.relu(self.conv2(x))
        #print(f'After conv2: {x.shape}')
        x = self.pool(x)
        #print(f'After pool2: {x.shape}')
        x = F.relu(self.conv3(x))
        #print(f'After conv3: {x.shape}')
        x = self.pool(x)
        #print(f'After pool3: {x.shape}')
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        #print(f'After flattening: {x.shape}')  # This is mat1
        x = F.relu(self.fc1(x))
        #print(f'After fc1 (mat2): {self.fc1.weight.shape}')  # Print shape of fc1 weights (mat2)
        x = self.fc2(x)
        return x

# Define the input dimensions and number of classes
input_channels = train_data[0][0].shape[0]  # Number of channels in the input
#print(f"Input channels: {input_channels}")

# Create an instance of SimpleCNN
model = SimpleCNN(input_channels, num_classes)
#print(model)

# -------------------------- End Model Development ----------------------------------------------------

# -------------------------- Begin Model Training ----------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)  # Outputs should be of shape (batch_size, num_classes)
        #print(f'Train outputs shape: {outputs.shape}')
        #print(f'Train labels shape: {labels.shape}')
        
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(f'Epoch {epoch}, Loss: {running_loss / (i + 1):.3f}')

print('Finished Training')
# -------------------------- End Model Training ----------------------------------------------------

# -------------------------- Begin Model Testing ----------------------------------------------------
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')
# -------------------------- End Model Testing ----------------------------------------------------

# Calculate confusion matrix

all_labels = []
all_predictions = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.numpy())
        all_predictions.extend(predicted.numpy())

# Create confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

batch_size = 500

summary(model, input_size = (batch_size, 10, 749) ) # args: model, input_size = (batch_size, dim, dim, dim)