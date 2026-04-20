import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

def predict_for_restaurant(data):
    predictions = {}
    items = data['food_item'].unique()

    data['date'] = pd.to_datetime(data['date'])
    data['day_num'] = (data['date'] - data['date'].min()).dt.days

    for item in items:
        item_data = data[data['food_item'] == item]
        X = item_data[['day_num']]
        y = item_data['quantity_sold']

        model = LinearRegression()
        model.fit(X, y)

        last_day = item_data['day_num'].max()
        future_days = np.array([last_day + i for i in range(1, 8)]).reshape(-1, 1)
        future_dates = [(item_data['date'].max() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]
        preds = model.predict(future_days).round().astype(int)
        preds = [max(0, p) for p in preds]

        predictions[item] = dict(zip(future_dates, preds))

    return predictions
