## DL TRAINING

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.transforms import v2
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
### Helper functions

def evaluate(model, dataloader, loss_fn):
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            # Run predictions
            output = model(features)

            # Compute loss
            loss = loss_fn(output, labels)

            # Save metrics
            predicted_labels = output.argmax(dim=1)
            correct_predictions += (predicted_labels == labels).sum().item()
            losses.append(loss.item())

    mean_loss = np.array(losses).mean()
    accuracy = 100.0 * correct_predictions / len(dataloader.dataset)
    
    # Return mean loss and accuracy
    return mean_loss, accuracy



def plot(train_losses, val_losses, train_accuracies, val_accuracies, title):
    plt.figure()
    plt.plot(np.arange(len(train_losses)), train_losses)
    plt.plot(np.arange(len(val_losses)), val_losses)
    plt.legend(['train_loss', 'val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.title('{}: Train/val loss'.format(title));

    plt.figure()
    plt.plot(np.arange(len(train_accuracies)), train_accuracies)
    plt.plot(np.arange(len(val_accuracies)), val_accuracies)
    plt.legend(['train_acc', 'val_acc'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('{}: Train/val accuracy'.format(title));
### Load ICA features and metadata
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# -----------------------
# Load data
# -----------------------
data = np.load("all_patients.npz")
X = data["X"]   # (N, 2, 3000)
y = data["y"]   # (N,)

print("original shapes:", X.shape, y.shape)

# Flatten for MLP
X = X.reshape(X.shape[0], -1)   # (N, 6000)
print("flattened shape:", X.shape)

# -----------------------
# Split: 80/15/5 overall
# -----------------------
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

val_fraction_of_trainval = 0.05 / 0.85

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=val_fraction_of_trainval,
    random_state=42,
    stratify=y_trainval
)

print("split successful")
print("train:", X_train.shape, y_train.shape)
print("val:  ", X_val.shape, y_val.shape)
print("test: ", X_test.shape, y_test.shape)

# -----------------------
# Normalisation
# -----------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("normalisation complete")

# -----------------------
# Dataset / DataLoader
# -----------------------
class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = FeatureDataset(X_train, y_train)
val_ds   = FeatureDataset(X_val, y_val)
test_ds  = FeatureDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)

print("dataloaders set up")

# -----------------------
# MLP model
# -----------------------
class SleepMLP(nn.Module):
    def __init__(self, n_features, n_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SleepMLP(n_features=X_train.shape[1], n_classes=len(np.unique(y))).to(device)

print("model instantiated")

# -----------------------
# Loss / optimizer
# -----------------------
class_counts = np.bincount(y_train, minlength=len(np.unique(y)))
class_weights = 1.0 / np.maximum(class_counts, 1)
class_weights = class_weights / class_weights.sum() * len(class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

print("parameters set")

# -----------------------
# Training / evaluation
# -----------------------
def run_epoch(loader, train=True):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0

    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)

        if train:
            optimizer.zero_grad()

        logits = model(Xb)
        loss = criterion(logits, yb)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(yb)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += len(yb)

    return total_loss / total, correct / total


# -----------------------
# Train
# -----------------------
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(1, 31):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    va_loss, va_acc = run_epoch(val_loader, train=False)

    train_losses.append(tr_loss)
    val_losses.append(va_loss)
    train_accuracies.append(tr_acc)
    val_accuracies.append(va_acc)
    

    print(f"Epoch {epoch:02d} | "
          f"Train: loss={tr_loss:.4f}, acc={tr_acc:.3f} | "
          f"Val: loss={va_loss:.4f}, acc={va_acc:.3f}")
    

plot(train_losses, val_losses, train_accuracies, val_accuracies, title=f"MLP on all patients")




def evaluate(model, dataloader, loss_fn):
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            # Run predictions
            output = model(features)

            # Compute loss
            loss = loss_fn(output, labels)

            # Save metrics
            predicted_labels = output.argmax(dim=1)
            correct_predictions += (predicted_labels == labels).sum().item()
            losses.append(loss.item())

    mean_loss = np.array(losses).mean()
    accuracy = correct_predictions / len(dataloader.dataset)
    
    # Return mean loss and accuracy
    return mean_loss, accuracy

def plot(train_losses, val_losses, train_accuracies, val_accuracies, title):
    plt.figure()
    plt.plot(np.arange(len(train_losses)), train_losses)
    plt.plot(np.arange(len(val_losses)), val_losses)
    plt.legend(['train_loss', 'val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.title(f'{title}: Train/val loss')

    plt.figure()
    plt.plot(np.arange(len(train_accuracies)), train_accuracies)
    plt.plot(np.arange(len(val_accuracies)), val_accuracies)
    plt.legend(['train_acc', 'val_acc'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title(f'{title}: Train/val accuracy')

    plt.show()
# -----------------------
# Test
# -----------------------
va_loss, va_ac = evaluate(model, val_loader, criterion)
print(f'MLP all patients\nValidation Loss: {va_loss:.4f}\nValidation Accuracy: {va_acc:.4f}')
test_loss, test_acc = run_epoch(test_loader, train=False)
print(f"Test: loss={test_loss:.4f}, acc={test_acc:.3f}")



plot(train_losses, val_losses, train_accuracies, val_accuracies, title=f"MLP on all patients")