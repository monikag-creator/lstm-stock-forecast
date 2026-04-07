"""
Evaluation & Visualization Utilities
=====================================
Computes regression metrics and generates publication-quality plots
for the LSTM stock price forecasting project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from typing import Optional


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    target_scaler: Optional[MinMaxScaler] = None) -> dict:
    """
    Compute MAE, RMSE, MSE, and MAPE.

    If `target_scaler` is supplied the arrays are inverse-transformed
    to original price scale before metric computation.

    Returns:
        dict with keys: MAE, RMSE, MSE, MAPE
    """
    if target_scaler is not None:
        y_true = target_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100

    metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE (%)": mape}

    print("\n── Model Evaluation Metrics ──────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    print("──────────────────────────────────────────────────\n")

    return metrics


# ─── Plot Helpers ─────────────────────────────────────────────────────────────

STYLE = {
    "bg":     "#0D1117",
    "panel":  "#161B22",
    "grid":   "#21262D",
    "accent": "#58A6FF",
    "actual": "#3FB950",
    "pred":   "#FF7B72",
    "text":   "#E6EDF3",
    "muted":  "#8B949E",
}


def _apply_dark_theme(fig, axes):
    fig.patch.set_facecolor(STYLE["bg"])
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(STYLE["panel"])
        ax.tick_params(colors=STYLE["text"])
        ax.xaxis.label.set_color(STYLE["text"])
        ax.yaxis.label.set_color(STYLE["text"])
        ax.title.set_color(STYLE["text"])
        ax.spines[:].set_color(STYLE["grid"])
        ax.yaxis.set_tick_params(which='both', colors=STYLE["muted"])
        ax.xaxis.set_tick_params(which='both', colors=STYLE["muted"])
        ax.grid(color=STYLE["grid"], linestyle="--", linewidth=0.6, alpha=0.7)


# ─── Individual Plots ─────────────────────────────────────────────────────────

def plot_predictions(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     target_scaler: Optional[MinMaxScaler],
                     dates: Optional[pd.DatetimeIndex] = None,
                     title: str = "LSTM Stock Price Forecast",
                     save_path: str = "results/predictions.png"):
    """Actual vs Predicted prices on the test set."""

    if target_scaler is not None:
        y_true = target_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    x = dates if dates is not None else np.arange(len(y_true))

    fig, ax = plt.subplots(figsize=(14, 5))
    _apply_dark_theme(fig, ax)

    ax.plot(x, y_true, color=STYLE["actual"], lw=1.8, label="Actual Price", alpha=0.9)
    ax.plot(x, y_pred, color=STYLE["pred"],   lw=1.8, label="Predicted Price",
            linestyle="--", alpha=0.9)

    ax.fill_between(x, y_true, y_pred,
                    where=(y_pred > y_true), alpha=0.08, color=STYLE["pred"])
    ax.fill_between(x, y_true, y_pred,
                    where=(y_pred <= y_true), alpha=0.08, color=STYLE["actual"])

    if dates is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=30)

    ax.set_title(title, fontsize=15, pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    legend = ax.legend(frameon=True, fontsize=10)
    legend.get_frame().set_facecolor(STYLE["panel"])
    legend.get_frame().set_edgecolor(STYLE["grid"])
    for text in legend.get_texts():
        text.set_color(STYLE["text"])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_training_history(history: dict,
                          save_path: str = "results/training_loss.png"):
    """Training vs validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 4))
    _apply_dark_theme(fig, ax)

    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], color=STYLE["accent"],  lw=2, label="Train Loss")
    ax.plot(epochs, history["val_loss"],   color=STYLE["pred"],    lw=2,
            linestyle="--", label="Validation Loss")

    ax.set_title("Training History (MSE Loss)", fontsize=14, pad=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    legend = ax.legend(frameon=True)
    legend.get_frame().set_facecolor(STYLE["panel"])
    legend.get_frame().set_edgecolor(STYLE["grid"])
    for t in legend.get_texts():
        t.set_color(STYLE["text"])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_error_distribution(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            target_scaler: Optional[MinMaxScaler],
                            save_path: str = "results/error_distribution.png"):
    """Histogram of residuals (actual − predicted)."""
    if target_scaler is not None:
        y_true = target_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    errors = y_true - y_pred

    fig, ax = plt.subplots(figsize=(9, 4))
    _apply_dark_theme(fig, ax)

    ax.hist(errors, bins=40, color=STYLE["accent"], edgecolor=STYLE["bg"], alpha=0.8)
    ax.axvline(0, color=STYLE["actual"], lw=1.5, linestyle="--", label="Zero Error")
    ax.axvline(errors.mean(), color=STYLE["pred"],  lw=1.5, linestyle="-",
               label=f"Mean Error: {errors.mean():.2f}")

    ax.set_title("Residual Distribution", fontsize=14, pad=10)
    ax.set_xlabel("Error (Actual − Predicted)")
    ax.set_ylabel("Frequency")
    legend = ax.legend(frameon=True)
    legend.get_frame().set_facecolor(STYLE["panel"])
    legend.get_frame().set_edgecolor(STYLE["grid"])
    for t in legend.get_texts():
        t.set_color(STYLE["text"])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_full_overview(df_clean: pd.DataFrame,
                       y_test_true: np.ndarray,
                       y_test_pred: np.ndarray,
                       target_scaler: MinMaxScaler,
                       seq_len: int,
                       save_path: str = "results/full_overview.png"):
    """
    4-panel overview:
        1. Full price history  2. Test predictions
        3. Residuals over time 4. RSI indicator
    """
    if target_scaler is not None:
        y_true = target_scaler.inverse_transform(y_test_true.reshape(-1, 1)).flatten()
        y_pred = target_scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
    else:
        y_true, y_pred = y_test_true, y_test_pred

    dates_all  = df_clean.index
    dates_test = dates_all[-(len(y_true)):]

    fig, axes = plt.subplots(4, 1, figsize=(15, 16),
                             gridspec_kw={"height_ratios": [3, 3, 2, 2]})
    _apply_dark_theme(fig, axes)
    fig.suptitle("LSTM Stock Price Forecast — Full Overview",
                 color=STYLE["text"], fontsize=16, y=1.01)

    # Panel 1: Full history
    axes[0].plot(dates_all, df_clean["Close"],
                 color=STYLE["accent"], lw=1.2, label="Close Price")
    axes[0].axvspan(dates_test[0], dates_test[-1],
                    alpha=0.12, color=STYLE["pred"], label="Test Period")
    axes[0].set_title("Full Price History", fontsize=12)
    axes[0].set_ylabel("Price (USD)")
    _legend(axes[0])

    # Panel 2: Test predictions
    axes[1].plot(dates_test, y_true, color=STYLE["actual"], lw=1.8, label="Actual")
    axes[1].plot(dates_test, y_pred, color=STYLE["pred"],   lw=1.8,
                 linestyle="--", label="Predicted")
    axes[1].set_title("Test Set: Actual vs Predicted", fontsize=12)
    axes[1].set_ylabel("Price (USD)")
    _legend(axes[1])

    # Panel 3: Residuals
    residuals = y_true - y_pred
    axes[2].bar(dates_test, residuals,
                color=[STYLE["actual"] if r >= 0 else STYLE["pred"] for r in residuals],
                alpha=0.7, width=1.5)
    axes[2].axhline(0, color=STYLE["muted"], lw=1)
    axes[2].set_title("Residuals (Actual − Predicted)", fontsize=12)
    axes[2].set_ylabel("USD")

    # Panel 4: RSI
    if "RSI_14" in df_clean.columns:
        axes[3].plot(dates_all, df_clean["RSI_14"],
                     color=STYLE["accent"], lw=1.2, label="RSI (14)")
        axes[3].axhline(70, color=STYLE["pred"],   lw=1, linestyle="--", alpha=0.7)
        axes[3].axhline(30, color=STYLE["actual"], lw=1, linestyle="--", alpha=0.7)
        axes[3].fill_between(dates_all, 70, df_clean["RSI_14"].clip(upper=100),
                             where=df_clean["RSI_14"] > 70,
                             alpha=0.1, color=STYLE["pred"])
        axes[3].fill_between(dates_all, df_clean["RSI_14"].clip(lower=0), 30,
                             where=df_clean["RSI_14"] < 30,
                             alpha=0.1, color=STYLE["actual"])
        axes[3].set_title("RSI (14-day) — Overbought / Oversold", fontsize=12)
        axes[3].set_ylabel("RSI")
        axes[3].set_ylim(0, 100)
        _legend(axes[3])

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def _legend(ax):
    leg = ax.legend(frameon=True, fontsize=9)
    leg.get_frame().set_facecolor(STYLE["panel"])
    leg.get_frame().set_edgecolor(STYLE["grid"])
    for t in leg.get_texts():
        t.set_color(STYLE["text"])
