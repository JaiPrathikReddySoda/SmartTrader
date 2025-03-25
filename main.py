import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Define StockLSTM Model
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]
        output = self.fc(last_hidden_state)
        return output

# Model Hyperparameters
input_size = 3  # Number of features (High, Low, Close)
hidden_size = 64
num_layers = 2
output_size = 3  # Predict High, Low, and Avg Close
sequence_length = 10  # Sequence length for training

# Create Model
def createModel():
    model = StockLSTM(input_size, hidden_size, num_layers, output_size)
    return model

# Denormalization Function
def denormalize(scaler, data):
    return scaler.inverse_transform(data)

# Train Model Function
def trainModel(model, dataloader):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 15
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

# Prepare DataLoader
def createDataLoader(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length, :])
        y.append(data[i+sequence_length, :])
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Prediction Function
def predictStockPrices(date_input, ticker):
    selected_date = datetime.strptime(date_input, "%Y-%m-%d").date()

    # Fetch data for training
    start_date = selected_date - timedelta(days=365)
    end_date = selected_date
    df = yf.download(ticker, start=start_date, end=end_date)

    # Preprocess data
    df = df[['High', 'Low', 'Close']].dropna().values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Prepare model and data loader
    model = createModel()
    dataloader = createDataLoader(scaled_data, sequence_length)
    trainModel(model, dataloader)

    # Predict next 6 days (including today)
    predictions = []
    input_sequence = scaled_data[-sequence_length:, :]  # Last sequence for prediction
    model.eval()
    with torch.no_grad():
        for _ in range(6):  # Predict today and 5 consecutive days
            input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0)
            next_day_prediction = model(input_tensor).numpy()[0]
            predictions.append(next_day_prediction)
            input_sequence = np.vstack((input_sequence[1:], next_day_prediction))  # Update sequence

    # Denormalize predictions
    predictions = np.array(predictions)
    denorm_predictions = denormalize(scaler, predictions)

    return denorm_predictions

# Trading Strategy Function
def simulate_trading(date_input):
    nvda_predictions = predictStockPrices(date_input, 'NVDA')

    actions = []
    for i in range(1, 6):  # Start from day 1 (tomorrow)
        today_close = nvda_predictions[i - 1, 2]  # Previous day's close
        tomorrow_close = nvda_predictions[i, 2]  # Current day's close

        if tomorrow_close > today_close:
            actions.append("BULLISH")
        elif tomorrow_close < today_close:
            actions.append("BEARISH")
        else:
            actions.append("IDLE")

    # Calculate highest, lowest, and average closing prices for the next 5 days
    highest_price = np.max(nvda_predictions[1:6, 0])
    lowest_price = np.min(nvda_predictions[1:6, 1])
    average_close = np.mean(nvda_predictions[1:6, 2])

    return actions, nvda_predictions[1:6], highest_price, lowest_price, average_close

# Streamlit App
st.title("SmartTrader Console")

# Date picker input
max_date = datetime.now().date() + timedelta(days=10)
date_input = st.date_input("Assume today's date is:", value=datetime.now().date(), max_value=max_date)

if st.button("Predict"):
    try:
        # Convert selected date to string format
        date_input_str = date_input.strftime("%Y-%m-%d")

        # Simulate trading
        actions, nvda_predictions, highest_price, lowest_price, average_close = simulate_trading(date_input_str)

        # Prepare DataFrame for highest, lowest, and average prices
        df_summary = pd.DataFrame(
            {
                "Metric": ["Highest Price", "Lowest Price", "Average Closing Price"],
                "Value": [f"{highest_price:.2f}", f"{lowest_price:.2f}", f"{average_close:.2f}"],
            }
        )

        # Prepare DataFrame for trading strategy
        df_trading = pd.DataFrame(
            {
                "Day": [f"Day {i+1}" for i in range(5)],
                "Action": actions
            }
        )

        # Display summary table
        st.write("### Predicted prices for the next five business days (in USD) are")
        st.table(df_summary)

        # Display trading strategy table
        st.write("### Recommended Trading Strategy")
        st.table(df_trading)

    except Exception as e:
        st.error(f"An error occurred: {e}")
