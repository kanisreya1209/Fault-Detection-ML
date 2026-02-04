import os
import numpy as np
import torch
from src.model import CNNLSTMNet

# ========================
# Paths
# ========================
MODEL_PATH = "models/cnn_lstm_FD001.pth"

# ========================
# Load Model
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example dimensions – update based on your dataset
n_features = 24   # number of sensors/features
n_classes = 2     # normal vs faulty

model = CNNLSTMNet(n_features=n_features, n_classes=n_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ========================
# Example New Input
# ========================
# Shape = (batch, seq_len, features)
# Here, we simulate one engine run with 30 timesteps & 24 features
new_engine_data = np.random.rand(1, 30, n_features)

# Convert to tensor
X_new = torch.tensor(new_engine_data, dtype=torch.float32).to(device)

# ========================
# Prediction
# ========================
with torch.no_grad():
    output = model(X_new)
    predicted_class = output.argmax(dim=1).item()

print(f"🔮 Predicted Class: {predicted_class} (0=Normal, 1=Faulty)")
