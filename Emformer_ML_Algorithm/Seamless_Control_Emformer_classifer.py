"""
This is a deployment of the Emformer Waveform Classifier using 
surface electromyography (sEMG) and inertial measurement unit
data waveforms to recognize patient movements.

Ideally, this could be updated to become a predictor in real-time
and command an exoskeleton to help a patient with rehabilitation.

Authored by. Ryan J. Hartnett and Jarren Bachiller Berdal

Date: 18th May 2024
Version: 1.0
"""

# to make Data Preparation work
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# to make Model Development work
import torchaudio
import torch.nn as nn

# to make Model Training, Testing & Validation work
import torch.optim as optim

# to plot assessment plots
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------- Test Changes -----------------------------------

# change these values to try different hyperparameters
train_batch = 10
test_batch = 10
desired_epochs = 20
lrate = 0.01



# -------------------- Data Preparation --------------------------------

class WaveformDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        self.labels = []
        self.load_data(root_dir)

    def load_data(self, root_dir):
        for label, category in enumerate(os.listdir(root_dir)):
            category_path = os.path.join(root_dir, category)
            if os.path.isdir(category_path):
                for file in os.listdir(category_path):
                    file_path = os.path.join(category_path, file)
                    if file_path.endswith('.csv'):
                        data = pd.read_csv(file_path).values
                        print(f'Loaded {file_path} with shape {data.shape}')  # Debug: Print shape of each file
                        self.data.append(data)
                        self.labels.append(label)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        waveform = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(waveform, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    # Determine max shape
    max_length = max(item[0].shape[0] for item in batch)
    num_features = max(item[0].shape[1] for item in batch)
    
    # Pad sequences
    padded_waveforms = []
    lengths = []
    labels = []
    for waveform, label in batch:
        lengths.append(waveform.shape[0])
        padded_waveform = torch.zeros((max_length, num_features), dtype=torch.float32)
        padded_waveform[:waveform.shape[0], :waveform.shape[1]] = waveform
        padded_waveforms.append(padded_waveform)
        labels.append(label)
    
    return torch.stack(padded_waveforms), torch.tensor(lengths, dtype=torch.long), torch.tensor(labels)

# Root directory containing the data
root_dir = 'data/'
dataset = WaveformDataset(root_dir)
dataloader = DataLoader(dataset, batch_size=train_batch, shuffle=True, collate_fn=collate_fn)

# -------------------------- End Data Preparation ----------------------------------------------------

# -------------------------- Begin Model Development ----------------------------------------------------

class EmformerWaveformClassifier(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_classes, ffn_dim):
        super(EmformerWaveformClassifier, self).__init__()
        # Verify if the Emformer model exists
        if hasattr(torchaudio.models, 'Emformer'):
            self.emformer = torchaudio.models.Emformer(
                input_dim=input_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                ffn_dim=ffn_dim,
                right_context_length=0,
                left_context_length=0,
                segment_length=20,
                dropout=0.1
            )
        else:
            raise ImportError("Emformer model is not available in the current torchaudio version.")
        self.fc = nn.Linear(ffn_dim, num_classes)

    def forward(self, x, lengths):
        print(f'Input to Emformer: {x.shape}')  # Debug: Print shape before Emformer
        # x = x.permute(1, 0, 2)  # (batch, time, channels) -> (time, batch, channels)
        print(f'Input to Emformer after permute: {x.shape}')  # Debug: Print shape after permute
        x, _ = self.emformer(x, lengths=lengths)
        print(f'Output from Emformer: {x.shape}')  # Debug: Print shape after Emformer
        x = x.mean(dim=0)  # Mean over the time dimension
        print(f'Output after mean: {x.shape}')  # Debug: Print shape after mean
        x = self.fc(x)
        print(f'Output after FC: {x.shape}')  # Debug: Print shape after FC
        return x

# -------------------------- End Model Development ----------------------------------------------------

# -------------------------- Begin Model Training ----------------------------------------------------

# Determine input_dim based on standardized data
input_dim = dataset.data[0].shape[1]
print(f'Input dim:{input_dim}')

num_heads = 10
num_layers = 10
num_classes = 5
ffn_dim = 10  # Correct the typo from ffns to ffn_dim

model = EmformerWaveformClassifier(input_dim, num_heads, num_layers, num_classes, ffn_dim=ffn_dim)

# Training loop 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= lrate)

num_epochs = desired_epochs
epoch_batch =  int(len(dataset)%train_batch + (len(dataset)/train_batch)) * int(num_epochs)

epochList = range(epoch_batch)
lossArr = []

for epoch in range(num_epochs):
    train_loss, train_acc, total = 0, 0, 0
    correct,total = 0,0
    
    
    for i, (inputs, lengths, labels) in enumerate(dataloader):
        print(f'Train inputs shape:{inputs.shape}')
        optimizer.zero_grad()
        
        outputs = model(inputs, lengths)  # Outputs should be of shape (batch_size, num_classes)
        # outputs = outputs.permute(1, 0, 2)  # Permute to match the expected shape for CrossEntropyLoss
        outputs = outputs.reshape(-1, num_classes)  # Reshape to (batch_size * time, num_classes)
        print(f'Train outputs shape:{outputs.shape}')
        labels = labels.repeat(inputs.shape[1])  # Repeat labels to match the shape of outputs
        print(f'Train labels shape:{labels.shape}')
        
        # Truncate outputs and labels to match the length of the longest sequence
        max_length = torch.max(lengths)
        outputs = outputs[:max_length]
        labels = labels[:max_length]

        

        loss = criterion(outputs, labels)  # Compute the loss
        lossArr.append(loss.item())   # collate lost into array
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        train_loss += loss.item()
        train_acc = 100 * correct / total
        
        print(f"Epoch: {epoch} | Train Loss: {train_loss / (i + 1):.3f} | Train Acc: {train_acc:.2f}")

plt.plot(epochList, lossArr)
plt.title("Training Loss v Epoch")
plt.ylim([0, 3])
plt.savefig("plots/loss/trainingLoss.png")
   

print('Finished Training')


# -------------------------- End Model Training ----------------------------------------------------

# -------------------------- Begin Model Testing ----------------------------------------------------

# Test loop

test_dir = 'test/'
test_dataset = WaveformDataset(test_dir)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch, shuffle=False, collate_fn=collate_fn)



model.eval()
test_loss, test_acc, total = 0, 0, 0
correct,total = 0,0

with torch.inference_mode():
    for inputs, lengths, labels in test_dataloader:

        print(f'test inputs shape:{inputs.shape}')
        outputs = model(inputs, lengths)  # Outputs should be of shape (batch_size, num_classes)
        # outputs = outputs.permute(1, 0, 2)  # Permute to match the expected shape for CrossEntropyLoss
        outputs = outputs.reshape(-1, num_classes)  # Reshape to (batch_size * time, num_classes)
        print(f'test outputs shape:{outputs.shape}')
        labels = labels.repeat(inputs.shape[1])  # Repeat labels to match the shape of outputs
        print(f'test labels shape:{labels.shape}')
        
        # Truncate outputs and labels to match the length of the longest sequence
        max_length = torch.max(lengths)
        outputs = outputs[:max_length]
        labels = labels[:max_length]
     
        loss = criterion(outputs, labels)  # Compute the loss

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        test_loss += loss.item()
        test_acc = 100 * correct / total
        

print(f'Test total:{total}')
print(f'Test correct:{correct}')
print(f"Test Loss: {test_loss / len(test_dataloader):.3f} | Test Acc: {test_acc:.2f}")

print('Finished Testing')

# print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')



# -------------------------- End Model Testing ----------------------------------------------------


# --------------------------- Calculate confusion matrix ------------------------------------------

all_labels = []
all_predictions = []

with torch.no_grad():
    for inputs, lenghths, labels  in test_dataloader:
        outputs = model(inputs, lenghths)

        outputs = outputs.reshape(-1, num_classes)  # Reshape to (batch_size * time, num_classes)
        print(f'test outputs shape:{outputs.shape}')
        labels = labels.repeat(inputs.shape[1])  # Repeat labels to match the shape of outputs
        print(f'test labels shape:{labels.shape}')
        
        # Truncate outputs and labels to match the length of the longest sequence
        max_length = torch.max(lengths)
        outputs = outputs[:max_length]
        labels = labels[:max_length]


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
plt.savefig("plots/cnfsn/confusionMatrix.png")
plt.show()






"""
        print(f'Test inputs shape:{inputs.shape}')
        outputs = model(inputs, lengths)
        outputs = outputs.reshape(-1, num_classes)  # Reshape to (batch_size * time, num_classes)
        print(f'Test outputs shape:{outputs.shape}')
        # _, predicted = torch.max(outputs.data, 1)
        # total += labels.size(0)
        print(f'Test total:{total}')
        # correct += (predicted == labels).sum().item()
        print(f'Test correct:{correct}')

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # correct += predicted.eq(labels).sum().item()
        test_loss += loss.item()
        test_acc = 100 * correct / total


"""
