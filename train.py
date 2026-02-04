import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.model import CNNLSTMNet  # ✅ use correct class name

# ========================
# Paths
# ========================
DATA_DIR = "data/processed"
MODEL_PATH = "models/cnn_lstm_FD001.pth"

# ========================
# Parameters
# ========================
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001

# ========================
# Step 1: Load Preprocessed Data
# ========================
print("📂 Loading processed FD001 data...")
X_train = np.load(os.path.join(DATA_DIR, "FD001_X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "FD001_y_train.npy"))
X_val = np.load(os.path.join(DATA_DIR, "FD001_X_val.npy"))
y_val = np.load(os.path.join(DATA_DIR, "FD001_y_val.npy"))

print(f"✅ Train shape: {X_train.shape}, {y_train.shape}")
print(f"✅ Val shape: {X_val.shape}, {y_val.shape}")

# ========================
# Step 2: DataLoader
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ========================
# Step 3: Model, Loss, Optimizer
# ========================
n_features = X_train.shape[2]
n_classes = len(np.unique(y_train))

model = CNNLSTMNet(n_features=n_features, n_classes=n_classes).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ========================
# Step 4: Training Loop
# ========================
print("🚀 Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_x, val_y = val_x.to(device), val_y.to(device)
            preds = model(val_x).argmax(dim=1)
            correct += (preds == val_y).sum().item()
            total += val_y.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")

# ========================
# Step 5: Save Model
# ========================
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"💾 Model saved to {MODEL_PATH}")
