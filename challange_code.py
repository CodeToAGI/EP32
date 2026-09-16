"""
CodeToAGI — Deep Learning EP32 Challenge
Forecast Daily BTC-USD Price with LSTM
Window=30, Horizon=7 | Beat the naïve model on MAPE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime

# ────────────────────────────────────────────────
# 1. Download data
# ────────────────────────────────────────────────
print("Downloading BTC-USD daily prices...")
df = yf.download("BTC-USD", start="2018-01-01", end=datetime.now().strftime("%Y-%m-%d"), progress=False)
series = df["Close"].values.astype(np.float32).flatten()
print(f"Total days: {len(series)}")

# ────────────────────────────────────────────────
# 2. Chronological split (80 / 10 / 10)
# ────────────────────────────────────────────────
n = len(series)
train_end = int(n * 0.80)
val_end   = int(n * 0.90)

train_raw = series[:train_end]
val_raw   = series[train_end:val_end]
test_raw  = series[val_end:]

# ────────────────────────────────────────────────
# 3. Scale (fit on TRAIN only)
# ────────────────────────────────────────────────
scaler = StandardScaler()
train = scaler.fit_transform(train_raw.reshape(-1, 1)).flatten()
val   = scaler.transform(val_raw.reshape(-1, 1)).flatten()
test  = scaler.transform(test_raw.reshape(-1, 1)).flatten()

# ────────────────────────────────────────────────
# 4. Sliding window dataset
# ────────────────────────────────────────────────
WINDOW  = 30
HORIZON = 7

class TimeSeriesDataset(Dataset):
    def __init__(self, data, window=WINDOW, horizon=HORIZON):
        self.data = data
        self.W = window
        self.H = horizon

    def __len__(self):
        return len(self.data) - self.W - self.H + 1

    def __getitem__(self, i):
        x = self.data[i : i + self.W]
        y = self.data[i + self.W : i + self.W + self.H]
        return (
            torch.tensor(x, dtype=torch.float32).unsqueeze(-1),  # (W, 1)
            torch.tensor(y, dtype=torch.float32),                # (H,)
        )

train_ds = TimeSeriesDataset(train)
val_ds   = TimeSeriesDataset(val)
test_ds  = TimeSeriesDataset(test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

# ────────────────────────────────────────────────
# 5. Model
# ────────────────────────────────────────────────
class LSTMForecaster(nn.Module):
    def __init__(self, horizon=HORIZON, hidden=128):
        super().__init__()
        self.lstm1 = nn.LSTM(1, hidden, batch_first=True)
        self.drop  = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True)
        self.fc    = nn.Linear(hidden, horizon)

    def forward(self, x):
        out, _      = self.lstm1(x)
        out, (h, _) = self.lstm2(self.drop(out))
        return self.fc(h[-1])          # (B, horizon)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = LSTMForecaster().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# ────────────────────────────────────────────────
# 6. Train
# ────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    total_mae = 0.0
    n = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_mae += F.l1_loss(pred, y, reduction="sum").item()
            n += y.numel()
    return total_mae / n

best_val = float("inf")
best_state = None

print("\nTraining...")
for epoch in range(1, 51):
    model.train()
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = F.mse_loss(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    val_mae = evaluate(model, val_loader)
    scheduler.step(val_mae)

    if val_mae < best_val:
        best_val = val_mae
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d} | Val MAE (scaled): {val_mae:.4f}")

model.load_state_dict(best_state)
print("Best model loaded.\n")

# ────────────────────────────────────────────────
# 7. Evaluate on test set (original scale)
# ────────────────────────────────────────────────
model.eval()
all_preds, all_targets = [], []

with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        pred = model(X).cpu().numpy()
        all_preds.append(pred)
        all_targets.append(y.numpy())

preds   = np.concatenate(all_preds, axis=0)      # (N, 7)
targets = np.concatenate(all_targets, axis=0)

# Inverse transform
preds_inv   = scaler.inverse_transform(preds)
targets_inv = scaler.inverse_transform(targets)

# Metrics
mae  = np.mean(np.abs(preds_inv - targets_inv))
rmse = np.sqrt(np.mean((preds_inv - targets_inv) ** 2))
mape = np.mean(np.abs(preds_inv - targets_inv) / np.abs(targets_inv)) * 100

print("=" * 50)
print(f"LSTM Results (7-day horizon)")
print(f"MAE  : ${mae:,.2f}")
print(f"RMSE : ${rmse:,.2f}")
print(f"MAPE : {mape:.2f}%")
print("=" * 50)

# ────────────────────────────────────────────────
# 8. Naïve baseline (predict tomorrow = today)
# ────────────────────────────────────────────────
# For horizon=7 the naïve forecast is simply repeating the last known value 7 times
naive_preds = np.repeat(targets_inv[:, 0:1], HORIZON, axis=1)  # last observed → 7 days
naive_mape  = np.mean(np.abs(naive_preds - targets_inv) / np.abs(targets_inv)) * 100

print(f"Naïve MAPE : {naive_mape:.2f}%")
print(f"Improvement: {naive_mape - mape:.2f} percentage points")
print("=" * 50)

# ────────────────────────────────────────────────
# 9. Plot last forecast
# ────────────────────────────────────────────────
last_true = targets_inv[-1]
last_pred = preds_inv[-1]

plt.figure(figsize=(10, 5))
plt.plot(range(1, 8), last_true, "o-", label="Actual", color="#8be9fd", linewidth=2)
plt.plot(range(1, 8), last_pred, "s--", label="LSTM Forecast", color="#ffb86c", linewidth=2)
plt.title("BTC-USD — 7-Day Forecast (last test sample)", fontsize=14)
plt.xlabel("Days ahead")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("btc_7day_forecast.png", dpi=150)
plt.show()
print("Plot saved → btc_7day_forecast.png")
