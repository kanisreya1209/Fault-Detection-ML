import streamlit as st
import numpy as np
import torch
from src.model import CNNLSTMNet 
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import time

st.set_page_config(page_title="Jet Engine Fault Detection Dashboard", layout="wide")

MODEL_PATH = "models/cnn_lstm_FD001.pth"

dataset_map = {
    "train_FD001": ("FD001_X_train.npy", "FD001_y_train.npy", "FD001_scaler.pkl"),
    "val_FD001":   ("FD001_X_val.npy",   "FD001_y_val.npy",   "FD001_scaler.pkl"),
    "test_FD001":  ("FD001_X_test.npy",  "FD001_y_test.npy",  "FD001_scaler.pkl"),
}

selected_dataset = st.sidebar.selectbox("Select Dataset", list(dataset_map.keys()))
X_file, y_file, scaler_file = dataset_map[selected_dataset]
@st.cache_data
def load_npy(file_path):
    return np.load(file_path)

try:
    X_data = load_npy(f"data/processed/{X_file}")
    y_data = load_npy(f"data/processed/{y_file}")
    st.success(f"Loaded {selected_dataset} successfully! Shape: {X_data.shape}")
except FileNotFoundError:
    st.error(f"File not found: data/processed/{X_file} or data/processed/{y_file}")
    st.stop()

scaler = joblib.load(f"data/processed/{scaler_file}")
X_data_scaled = scaler.transform(X_data.reshape(-1, X_data.shape[-1])).reshape(X_data.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_features = X_data_scaled.shape[2]
n_classes = len(np.unique(y_data))

@st.cache_resource
def load_model(path, n_features, n_classes):
    model = CNNLSTMNet(n_features=n_features, n_classes=n_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model(MODEL_PATH, n_features, n_classes)

def predict_sample(sample):
    sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(sample_tensor)
        prob = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_label = np.argmax(prob)
    return pred_label, prob[pred_label]

st.subheader("Sample Prediction & Fault Alert")
sample_index = st.slider("Select Sample Index", 0, len(X_data_scaled)-1, 0)
sample = X_data_scaled[sample_index]
pred_label, confidence = predict_sample(sample)

st.write(f"**Predicted Label:** {pred_label} (Confidence: {confidence:.2f})")
st.write(f"**Actual Label:** {y_data[sample_index]}")

if pred_label != y_data[sample_index]:
    st.error("⚠ Fault detected! Prediction does not match actual label.")
else:
    st.success("✅ Prediction matches actual label.")

st.subheader("Live Continuous Sensor Graphs with Fault Alerts")
stream_samples = st.slider("Number of samples to stream", 20, 200, 50)
update_speed = st.slider("Update speed (ms per sample)", 50, 1000, 200)

live_plot_container = st.empty()

start_idx = st.number_input("Starting Sample Index for Live Stream", 0, len(X_data_scaled)-1, 0)
for i in range(stream_samples):
    idx = start_idx + i
    if idx >= len(X_data_scaled):
        break

    sample = X_data_scaled[idx]
    pred_label, _ = predict_sample(sample)

    fig, ax = plt.subplots(figsize=(18,4))
    for f in range(n_features):
        ax.plot(sample[:,f], label=f"Sensor {f+1}", alpha=0.7)

    if pred_label != y_data[idx]:
        ax.set_facecolor("#ffcccc") 
        alert_text = f"⚠ Fault! Sample {idx}, Predicted: {pred_label}, Actual: {y_data[idx]}"
    else:
        alert_text = f"✅ Sample {idx}, Predicted: {pred_label}, Actual: {y_data[idx]}"

    ax.set_title(alert_text)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Scaled Sensor Value")
    ax.legend(loc='upper right', fontsize='small')

    live_plot_container.pyplot(fig)
    time.sleep(update_speed / 1000)  

st.subheader("Batch Predictions & Fault Alerts")
batch_size = st.slider("Number of samples to preview", 5, 50, 10)

preds, confs = [], []
for i in range(batch_size):
    pl, conf = predict_sample(X_data_scaled[i])
    preds.append(pl)
    confs.append(conf)

df = pd.DataFrame({
    "Sample Index": list(range(batch_size)),
    "Predicted": preds,
    "Actual": y_data[:batch_size],
    "Confidence": confs
})

def highlight_fault(row):
    color = 'background-color: red' if row.Predicted != row.Actual else ''
    return [color]*len(row)

st.dataframe(df.style.apply(highlight_fault, axis=1))

st.subheader("Dataset Metrics")
all_preds = []
for i in range(len(X_data_scaled)):
    pl, _ = predict_sample(X_data_scaled[i])
    all_preds.append(pl)

acc = accuracy_score(y_data, all_preds)
st.write(f"**Accuracy on {selected_dataset}:** {acc*100:.2f}%")

cm = confusion_matrix(y_data, all_preds)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
ax_cm.set_title("Confusion Matrix")
st.pyplot(fig_cm)
num_faults = sum(np.array(all_preds) != y_data)
st.subheader("Fault Summary")
st.write(f"Total samples: {len(X_data_scaled)}")
st.write(f"Total mismatches (fault alerts): {num_faults}")
st.write(f"Fault rate: {num_faults/len(X_data_scaled)*100:.2f}%")
