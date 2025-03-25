import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import torch

def create_sequences(data, sequence_length, output_size=5):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length - output_size):
        sequences.append(data[i:i + sequence_length, :])  # Input: all features except the last one (closing price)
        labels.append(data[i + sequence_length:i + sequence_length + output_size, 0])  # Target: next 5 closing prices
    return np.array(sequences), np.array(labels)

def download_data():
    
    data = yf.download('NVDA', start='2024-01-01', end='2024-12-06') # Assume the column "Close" is present
    closing_prices = data['Close'].values

    return data, closing_prices

def calculate_rsi(data, window=14):

    delta = data.diff()  # Difference between consecutive prices
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):

    short_ema = data.ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = data.ewm(span=long_window, min_periods=1, adjust=False).mean()

    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, min_periods=1, adjust=False).mean()
    
    return macd, signal_line


def feature_extracting_pipeline():

    data, close = download_data()

    data['RSI'] = calculate_rsi(data['Close'], window=14)
    data['MACD'], data['MACD_Signal'] = calculate_macd(data['Close'], short_window=12, long_window=26, signal_window=9)

    # Drop rows with NaN values (due to rolling calculations)
    data.dropna(inplace=True)

    features = data[['Close', 'RSI', 'MACD', 'MACD_Signal']].values

    return features



def train_test_splitttt(features):

    split_ratio = 0.9
    split_index = int(len(features) * split_ratio)
    train_data = features[:split_index]
    test_data = features[split_index:]

    return train_data, test_data


def scaler_train_test():
    
    scaler_dict = {}
    features = feature_extracting_pipeline()
    train_data, test_data = train_test_splitttt(features)

    scaler_train = MinMaxScaler(feature_range=(0, 1))
    scaler_test = MinMaxScaler(feature_range=(0, 1))
    scaler_dict[0] = scaler_train
    scaler_dict[1] = scaler_test
    train_scaled = scaler_train.fit_transform(train_data)
    test_scaled = scaler_test.fit_transform(test_data)

    return train_scaled, test_scaled, scaler_dict



def create_tensor_test_train(X_train,y_train,X_test,y_test):

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, y_train, X_test, y_test


def create_date_for_training():

    features = feature_extracting_pipeline()
    train_data, test_data = train_test_splitttt(features)
    train_scaled, test_scaled, scaler_dict = scaler_train_test()
    sequence_length = 10
    X_train, y_train = create_sequences(train_scaled, sequence_length)
    X_test, y_test = create_sequences(test_scaled, sequence_length)
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = create_tensor_test_train(X_train,y_train,X_test,y_test)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler_dict












