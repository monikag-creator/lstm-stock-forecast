"""
LSTM Model for Stock Price Forecasting
=======================================
Defines and trains the LSTM neural network architecture
for multi-step time series forecasting.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMForecaster(nn.Module):
    """
    Multi-layer LSTM model for stock price prediction.

    Architecture:
        - Input layer (feature_size)
        - Stacked LSTM layers with dropout
        - Fully connected output layer (1 → predicted price)
    """

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2, output_size: int = 1):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Stacked LSTM with dropout between layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)

        # Fully connected output head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM → FC.

        Args:
            x: Tensor of shape (batch, seq_len, input_size)

        Returns:
            Tensor of shape (batch, output_size)
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))       # (batch, seq_len, hidden)
        out = self.dropout(out[:, -1, :])       # take last time-step
        out = self.fc(out)                       # (batch, output_size)
        return out


def train_model(model: nn.Module,
                X_train: np.ndarray,
                y_train: np.ndarray,
                X_val: np.ndarray,
                y_val: np.ndarray,
                epochs: int = 100,
                batch_size: int = 32,
                lr: float = 1e-3,
                patience: int = 15) -> dict:
    """
    Train the LSTM model with early stopping.

    Args:
        model       : LSTMForecaster instance
        X_train     : Training sequences  (N, seq_len, features)
        y_train     : Training targets    (N,)
        X_val       : Validation sequences
        y_val       : Validation targets
        epochs      : Max training epochs
        batch_size  : Mini-batch size
        lr          : Adam learning rate
        patience    : Early-stopping patience (epochs without improvement)

    Returns:
        history dict with train_loss and val_loss lists
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Convert numpy → tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_v  = torch.tensor(X_val,   dtype=torch.float32).to(device)
    y_v  = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1).to(device)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=7, factor=0.5
    )

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_weights = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = np.mean(batch_losses)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v), y_v).item()

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

    # Restore best weights
    if best_weights:
        model.load_state_dict(best_weights)

    return history
