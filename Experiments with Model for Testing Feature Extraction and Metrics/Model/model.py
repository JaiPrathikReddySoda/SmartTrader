import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# LSTM Model Definition
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        Initializes the LSTM model for stock price prediction.
        Args:
            input_size (int): Number of input features (e.g., Closing price).
            hidden_size (int): Number of hidden units in the LSTM layers.
            num_layers (int): Number of stacked LSTM layers.
            output_size (int): Number of outputs (e.g., Closing prices for the next days).
            dropout (float): Dropout rate for regularization.
        """
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, dropout=dropout)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the LSTM model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
        Returns:
            torch.Tensor: Output predictions of shape (batch_size, output_size).
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # Output shape: (batch_size, sequence_length, hidden_size)
        
        # Select the last time step's output for prediction
        lstm_out_last = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Fully connected layer
        output = self.fc(lstm_out_last)  # Shape: (batch_size, output_size)
        return output
    
def create_model():
    input_size = 4  # Only the closing price
    hidden_size = 64
    num_layers = 2
    output_size = 5  # Predict 5 future closing prices

    model_lstm = StockLSTM(input_size, hidden_size, num_layers, output_size)

    return model_lstm