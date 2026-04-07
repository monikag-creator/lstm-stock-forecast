# lstm-stock-forecast
LSTM Stock Price Forecasting using PyTorch

LSTM Stock Price Forecasting
A deep learning project that predicts stock prices using an LSTM neural network built with PyTorch.
Dataset
Source: Yahoo Finance (via yfinance library)
Stock: AAPL (Apple Inc.)
Period: 2018-01-01 to 2024-01-01
Total: 1,509 trading days

Features Used:
Open, High, Low, Close, Volume
Moving Averages (MA5, MA20)
EMA, MACD, RSI, Bollinger Bands

Model:
2-layer LSTM network
Hidden size: 128
Dropout: 0.2
Optimizer: Adam
Loss: MSE

Results:
MetricValueMAE3.37RMSE4.03MSE16.27MAPE1.91%

How to Run:
bashpip install -r requirements.txt
python main.py --ticker AAPL --start 2018-01-01 --end 2024-01-01

Output:
All results are saved in the results/ folder:
predictions.png — Actual vs Predicted prices
training_loss.png — Training curve
full_overview.png — Complete analysis
metrics.json — Performance metrics
