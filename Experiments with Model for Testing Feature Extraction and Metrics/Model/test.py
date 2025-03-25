import torch
from Model.model import StockLSTM
from Data.create_data import create_date_for_training
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

input_size = 4  # Only the closing price
hidden_size = 64
num_layers = 2
output_size = 5  # Predict 5 future closing prices

X_train, y_train, X_test, y_test, scaler_dict = create_date_for_training()

model = StockLSTM(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load('saved_model/stock_price_model.pth'))

model.eval()
with torch.no_grad():
    test_predictions = model(X_test).detach().numpy()

    print("Predicted Closing Prices for the Next 5 Days:", test_predictions[-1])

    # Optionally, evaluate the model's performance on the test set
    #print("Test Predictions :", test_predictions[:2])
    #print(y_test.shape, test_predictions.shape)
    #print("Actual Test Values :", y_test.numpy().reshape(-1,1))
    print("RMSE :", root_mean_squared_error(y_test.numpy()[:], test_predictions[:]))
    print("RMSE :", mean_absolute_error(y_test.numpy()[:], test_predictions[:]))
    print("RMSE :", mean_squared_error(y_test.numpy()[:], test_predictions[:]))

def plot():

    plt.plot(y_test.numpy()[:].flatten(), label="Actual Prices", color="blue")
    plt.plot(test_predictions[:].flatten(), label="Predicted Prices", color="red", linestyle="--")
    plt.title("Stock Price Prediction (Actual vs Predicted)")
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()


def plotPredErrordist():
    errors = y_test.numpy()[:].flatten() - test_predictions[:].flatten()
    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=30, color="purple", alpha=0.7)
    plt.title("Prediction Error Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.show()

def plotRollingRMSE():
    errors = y_test.numpy()[:].flatten()- test_predictions[:].flatten()
    window_size = 20
    rolling_rmse = np.sqrt(pd.Series(errors**2).rolling(window=window_size).mean())
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_rmse, label="Rolling RMSE", color="green")
    plt.title("Rolling RMSE Over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()

def plotResidual():
    errors = y_test.numpy()[:].flatten()- test_predictions[:].flatten()
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test.numpy()[:].flatten(), errors, alpha=0.6, color="orange")
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residual Plot")
    plt.xlabel("Actual Prices")
    plt.ylabel("Residuals (Error)")
    plt.show()

def draw_plots():
    plot()
    plotPredErrordist()
    plotRollingRMSE()
    plotResidual()

draw_plots()
    