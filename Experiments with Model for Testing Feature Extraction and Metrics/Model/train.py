import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from Model.model import create_model
from Data.create_data import create_date_for_training
import sys
import os

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

model = create_model()
X_train, y_train, X_test, y_test, scaler_dict = create_date_for_training()


def train(num_epochs):

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    num_epochs = num_epochs
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()


        outputs = model(X_train)
        loss = criterion(outputs, y_train)


        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


train(1000)

torch.save(model.state_dict(), 'saved_model/stock_price_model.pth')