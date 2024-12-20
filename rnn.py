import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import time

device_compute = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Computation device: {device_compute}")

def load_dataset(base_dir, data_split='LS'):
    dir_path = os.path.join(base_dir, data_split)
    data_matrix = np.zeros((TOTAL_SAMPLES, NUM_FEATURES, SEQ_LEN))
    labels_output = None

    for idx, feature in enumerate(FEATURE_LIST):
        file_name = f'{data_split}_sensor_{feature}.txt'
        full_path = os.path.join(dir_path, file_name)
        raw = np.loadtxt(full_path)
        raw[raw == -999999.99] = np.nan
        data_matrix[:, idx, :] = raw

    nan_present = np.isnan(data_matrix)
    feature_means = np.nanmean(data_matrix, axis=(0, 2))
    for feature_idx in range(data_matrix.shape[1]):
        data_matrix[:, feature_idx, :][nan_present[:, feature_idx, :]] = feature_means[feature_idx]

    if data_split == 'LS':
        labels_output = np.loadtxt(os.path.join(dir_path, 'activity_Id.txt'))

    return data_matrix, labels_output

def normalize_dataset(dataset):
    for feature_idx in range(dataset.shape[1]):
        mean_val = np.mean(dataset[:, feature_idx, :])
        std_val = np.std(dataset[:, feature_idx, :])
        dataset[:, feature_idx, :] = (dataset[:, feature_idx, :] - mean_val) / (std_val + 1e-8)
    return dataset

class HAR_Dataset(Dataset):
    def __init__(self, features, targets=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets - 1, dtype=torch.long) if targets is not None else None

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        if self.targets is not None:
            return self.features[index], self.targets[index]
        return self.features[index]

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    loss_total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        data = data.permute(0, 2, 1)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
    return loss_total / len(loader)

def evaluate(model, loader, loss_fn, device):
    model.eval()
    loss_sum = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            data = data.permute(0, 2, 1)
            output = model(data)
            loss = loss_fn(output, target)
            loss_sum += loss.item()
            _, preds = torch.max(output, 1)
            correct += (preds == target).sum().item()
    return loss_sum / len(loader), correct / len(loader.dataset)

def make_predictions(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            data = data.permute(0, 2, 1)
            output = model(data)
            _, predicted = torch.max(output, 1)
            preds.extend((predicted.cpu().numpy() + 1).tolist())
    return preds

BASE_DIR = './'
FEATURE_LIST = range(2, 33)
TOTAL_SAMPLES = 3500
SEQ_LEN = 512
NUM_FEATURES = 31

X_train, y_train = load_dataset(BASE_DIR, 'LS')
X_test, _ = load_dataset(BASE_DIR, 'TS')
X_train = normalize_dataset(X_train)
X_test = normalize_dataset(X_test)

full_ds = HAR_Dataset(X_train, y_train)
train_count = int(0.8 * len(full_ds))
val_count = len(full_ds) - train_count
train_ds, val_ds = random_split(full_ds, [train_count, val_count])

BATCH_SIZE = 64
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_ds = HAR_Dataset(X_test)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

input_dim = 31
hidden_dim = 128
output_dim = 14
num_layers = 2
model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers).to(device_compute)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device_compute)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device_compute)
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

start_time = time.time()
test_preds = make_predictions(model, test_loader, device_compute)
end_time = time.time()
prediction_time = end_time - start_time

print(f"Predictions completed in {prediction_time:.2f} seconds.")

submission = pd.DataFrame({
    'Id': np.arange(1, len(test_preds) + 1),
    'Prediction': test_preds
})

submission.to_csv('rnn_submission.csv', index=False)
print("Predictions saved to submission.csv")
