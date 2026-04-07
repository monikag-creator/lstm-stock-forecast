"""
Data Preprocessing Utilities
==============================
Handles data loading, cleaning, feature engineering,
scaling, and sequence generation for the LSTM model.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List


# ─── Feature Engineering ────────────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute common technical indicators and append as new columns.

    Indicators added:
        MA_5, MA_20     – Simple Moving Averages (5-day, 20-day)
        EMA_12, EMA_26  – Exponential Moving Averages
        RSI_14          – Relative Strength Index (14-day)
        BB_Upper/Lower  – Bollinger Bands (20-day, 2σ)
        MACD            – MACD line (EMA_12 − EMA_26)
        Daily_Return    – Percentage change in Close
        Volume_MA_5     – 5-day moving average of Volume
    """
    df = df.copy()

    # Moving Averages
    df["MA_5"]  = df["Close"].rolling(window=5).mean()
    df["MA_20"] = df["Close"].rolling(window=20).mean()

    # Exponential Moving Averages
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]

    # Relative Strength Index (14-day)
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    std20 = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["MA_20"] + 2 * std20
    df["BB_Lower"] = df["MA_20"] - 2 * std20

    # Daily return & volume trend
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volume_MA_5"]  = df["Volume"].rolling(5).mean()

    return df


# ─── Missing Value Handling ──────────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy:
        1. Forward-fill to propagate last valid observation.
        2. Back-fill to handle leading NaNs (from rolling windows).
        3. Drop any remaining NaN rows.
    """
    df = df.ffill().bfill()
    initial_len = len(df)
    df = df.dropna()
    dropped = initial_len - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with unresolvable NaNs.")
    return df


# ─── Scaling ─────────────────────────────────────────────────────────────────

def scale_features(df: pd.DataFrame,
                   feature_cols: List[str],
                   target_col: str = "Close"
                   ) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """
    Independently scale features and target with MinMaxScaler [0, 1].

    Returns:
        X_scaled       : scaled feature array
        y_scaled       : scaled target array
        feature_scaler : fitted scaler for features
        target_scaler  : fitted scaler for target (needed for inverse transform)
    """
    feature_scaler = MinMaxScaler()
    target_scaler  = MinMaxScaler()

    X_scaled = feature_scaler.fit_transform(df[feature_cols].values)
    y_scaled = target_scaler.fit_transform(df[[target_col]].values).flatten()

    return X_scaled, y_scaled, feature_scaler, target_scaler


# ─── Sequence Generation ─────────────────────────────────────────────────────

def create_sequences(X: np.ndarray,
                     y: np.ndarray,
                     seq_len: int = 60
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a window of length `seq_len` across the data.

    Args:
        X       : Feature array  (N, features)
        y       : Target array   (N,)
        seq_len : Look-back window size

    Returns:
        X_seq : (N - seq_len, seq_len, features)
        y_seq : (N - seq_len,)
    """
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


# ─── Train / Val / Test Split ─────────────────────────────────────────────────

def temporal_split(X: np.ndarray,
                   y: np.ndarray,
                   train_ratio: float = 0.70,
                   val_ratio: float   = 0.15
                   ) -> Tuple[np.ndarray, ...]:
    """
    Chronological (non-shuffled) split into train / validation / test sets.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    n = len(X)
    t = int(n * train_ratio)
    v = int(n * (train_ratio + val_ratio))

    return (X[:t], X[t:v], X[v:],
            y[:t], y[t:v], y[v:])


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def preprocess_pipeline(df: pd.DataFrame,
                        seq_len: int = 60,
                        train_ratio: float = 0.70,
                        val_ratio: float   = 0.15
                        ) -> dict:
    """
    Full preprocessing pipeline:
        raw DataFrame → cleaned → features → scaled → sequences → split

    Args:
        df          : Raw OHLCV DataFrame with DatetimeIndex
        seq_len     : LSTM look-back window
        train_ratio : Fraction for training
        val_ratio   : Fraction for validation

    Returns:
        Dictionary containing split arrays, scalers, and feature column names.
    """
    print("→ Adding technical indicators …")
    df = add_technical_indicators(df)

    print("→ Handling missing values …")
    df = handle_missing_values(df)

    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "MA_5", "MA_20", "EMA_12", "EMA_26", "MACD",
        "RSI_14", "BB_Upper", "BB_Lower",
        "Daily_Return", "Volume_MA_5"
    ]

    print("→ Scaling features and target …")
    X_scaled, y_scaled, feat_scaler, tgt_scaler = scale_features(
        df, feature_cols, target_col="Close"
    )

    print(f"→ Creating sequences (look-back = {seq_len}) …")
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_len)

    print("→ Splitting into train / val / test …")
    X_train, X_val, X_test, y_train, y_val, y_test = temporal_split(
        X_seq, y_seq, train_ratio, val_ratio
    )

    print(f"   Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val":  y_val, "y_test":  y_test,
        "feature_scaler": feat_scaler,
        "target_scaler":  tgt_scaler,
        "feature_cols":   feature_cols,
        "clean_df":       df,
    }
