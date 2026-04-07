"""
Main Script — LSTM Stock Price Forecasting
==========================================
End-to-end pipeline:
    1. Download / load data
    2. Preprocess & engineer features
    3. Train LSTM model
    4. Evaluate & visualize results
    5. Save model checkpoint

Usage:
    python main.py --ticker AAPL --start 2018-01-01 --end 2024-01-01
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch

# ── local imports ──────────────────────────────────────────────────────────────
from utils.preprocessing import preprocess_pipeline
from utils.evaluation    import (compute_metrics, plot_predictions,
                                  plot_training_history, plot_error_distribution,
                                  plot_full_overview)
from models.lstm_model   import LSTMForecaster, train_model


# ── CLI args ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LSTM Stock Price Forecaster")
    p.add_argument("--ticker",      default="AAPL",       help="Yahoo Finance ticker symbol")
    p.add_argument("--start",       default="2018-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end",         default="2024-01-01", help="End date   (YYYY-MM-DD)")
    p.add_argument("--seq_len",     type=int, default=60,  help="LSTM look-back window")
    p.add_argument("--hidden",      type=int, default=128, help="LSTM hidden units")
    p.add_argument("--layers",      type=int, default=2,   help="Number of LSTM layers")
    p.add_argument("--dropout",     type=float, default=0.2)
    p.add_argument("--epochs",      type=int, default=100)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--patience",    type=int, default=15,  help="Early-stopping patience")
    p.add_argument("--data_path",   default=None,           help="Optional CSV path (skip download)")
    p.add_argument("--output_dir",  default="results",      help="Directory for outputs")
    return p.parse_args()


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_data(args) -> pd.DataFrame:
    """
    Load OHLCV data either from a local CSV or via yfinance.
    Expected columns: Open, High, Low, Close, Volume  (DatetimeIndex).
    """
    if args.data_path:
        print(f"Loading data from {args.data_path} …")
        df = pd.read_csv(args.data_path, index_col=0, parse_dates=True)
    else:
        try:
            import yfinance as yf
            print(f"Downloading {args.ticker} data ({args.start} → {args.end}) …")
            df = yf.download(args.ticker, start=args.start, end=args.end, progress=False)
            if df.empty:
                raise ValueError(f"No data returned for ticker '{args.ticker}'.")
        except ImportError:
            raise ImportError(
                "yfinance not installed. Run:  pip install yfinance\n"
                "Or supply --data_path to a local CSV file."
            )

    # Normalise column names
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing  = required - set(df.columns)
    if missing:
         df = df.sort_index()
    print(f"  Loaded {len(df)} trading days  ({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ── Inference Helpers ──────────────────────────────────────────────────────────

def predict(model: torch.nn.Module,
            X: np.ndarray,
            device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(X, dtype=torch.float32).to(device)
        preds  = model(tensor).cpu().numpy().flatten()
    return preds


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load data
    df = load_data(args)

    # 2. Preprocess
    print("\n── Preprocessing ─────────────────────────────────")
    data = preprocess_pipeline(df, seq_len=args.seq_len)

    # 3. Build model
    input_size = data["X_train"].shape[2]          # number of features
    print(f"\n── Model Architecture ─────────────────────────────")
    print(f"  Input size  : {input_size}")
    print(f"  Hidden size : {args.hidden}")
    print(f"  LSTM layers : {args.layers}")
    print(f"  Dropout     : {args.dropout}")

    model = LSTMForecaster(
        input_size  = input_size,
        hidden_size = args.hidden,
        num_layers  = args.layers,
        dropout     = args.dropout,
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {total_params:,}")

    # 4. Train
    print(f"\n── Training (max {args.epochs} epochs) ──────────────")
    history = train_model(
        model,
        data["X_train"], data["y_train"],
        data["X_val"],   data["y_val"],
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        patience   = args.patience,
    )

    # 5. Evaluate on test set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_pred = predict(model, data["X_test"], device)

    print("\n── Evaluation ────────────────────────────────────")
    metrics = compute_metrics(
        data["y_test"], y_pred,
        target_scaler=data["target_scaler"]
    )

    # Save metrics JSON
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({k: round(v, 6) for k, v in metrics.items()}, f, indent=2)
    print(f"  Metrics saved → {metrics_path}")

    # 6. Plots
    print("\n── Generating plots ──────────────────────────────")
    # Test date index
    n_test = len(data["y_test"])
    test_dates = data["clean_df"].index[-n_test:]

    plot_predictions(
        data["y_test"], y_pred,
        target_scaler = data["target_scaler"],
        dates         = test_dates,
        title         = f"{args.ticker} — LSTM Forecast (Test Set)",
        save_path     = os.path.join(args.output_dir, "predictions.png"),
    )

    plot_training_history(
        history,
        save_path = os.path.join(args.output_dir, "training_loss.png"),
    )

    plot_error_distribution(
        data["y_test"], y_pred,
        target_scaler = data["target_scaler"],
        save_path     = os.path.join(args.output_dir, "error_distribution.png"),
    )

    plot_full_overview(
        df_clean      = data["clean_df"],
        y_test_true   = data["y_test"],
        y_test_pred   = y_pred,
        target_scaler = data["target_scaler"],
        seq_len       = args.seq_len,
        save_path     = os.path.join(args.output_dir, "full_overview.png"),
    )

    # 7. Save model
    ckpt_path = os.path.join(args.output_dir, "lstm_model.pth")
    torch.save({
        "model_state":  model.state_dict(),
        "args":         vars(args),
        "metrics":      metrics,
        "feature_cols": data["feature_cols"],
    }, ckpt_path)
    print(f"\n  Model checkpoint saved → {ckpt_path}")

    print("\n✓  Pipeline complete. All outputs in:", args.output_dir)


if __name__ == "__main__":
    main()
