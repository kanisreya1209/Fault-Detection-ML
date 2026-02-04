import os
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from src.model import CNNLSTMNet

# ========================
# Paths
# ========================
DATA_DIR = "data/processed"
MODEL_PATH = "models/cnn_lstm_FD001.pth"

# ========================
# Step 1: Load Test Data
# ========================
print("📂 Loading test data...")
X_val = np.load(os.path.join(DATA_DIR, "FD001_X_val.npy"))
y_val = np.load(os.path.join(DATA_DIR, "FD001_y_val.npy"))

print(f"✅ Test shape: {X_val.shape}, {y_val.shape}")

# ========================
# Step 2: Model Setup
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_features = X_val.shape[2]
n_classes = len(np.unique(y_val))

model = CNNLSTMNet(n_features=n_features, n_classes=n_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ========================
# Step 3: Predictions
# ========================
print("🔍 Evaluating model...")
with torch.no_grad():
    X_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    preds = model(X_tensor).argmax(dim=1).cpu().numpy()

# ========================
# Step 4: Metrics
# ========================
print("\n📊 Classification Report:")
print(classification_report(y_val, preds))

print("📉 Confusion Matrix:")
print(confusion_matrix(y_val, preds))
