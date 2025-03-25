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

# Function to create sequences from historical closing prices
def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        labels.append(data[i + sequence_length])
    return np.array(sequences), np.array(labels)

# Example Usage
if __name__ == "__main__":
    # Sample data: Closing prices (replace with actual data)
    data = yf.download('NVDA', start='2024-12-14', end='2024-12-06') # Assume the column "Close" is present
    closing_prices = data['Close'].values

    # Normalize data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(closing_prices.reshape(-1, 1))

    # Prepare sequences for training (using the last 10 days to predict the next 1)
    sequence_length = 10
    X, y = create_sequences(scaled_data, sequence_length)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32)

    # Initialize the model
    input_size = 1  # Only the closing price
    hidden_size = 64
    num_layers = 2
    output_size = 5  # Predict 5 future closing prices
    model = StockLSTM(input_size, hidden_size, num_layers, output_size)

    # Define optimizer and loss function
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Prediction for the next 5 days
    model.eval()
    with torch.no_grad():
        # Use the last sequence for prediction
        last_sequence = torch.tensor(scaled_data[-sequence_length:], dtype=torch.float32).unsqueeze(0)
        predicted_scaled_prices = model(last_sequence)

        # Denormalize the predicted values
        predicted_prices = scaler.inverse_transform(predicted_scaled_prices.numpy().reshape(-1, 1))

    # Output the predicted prices for the next 5 days
    print(f"Predicted Closing Prices for the Next 5 Days: {predicted_prices.flatten()}")