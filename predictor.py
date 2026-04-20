import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from datetime import timedelta


def prepare_features(df):
    df = df.copy()
    # 🔥 DATA CLEANING
    df = df.dropna(subset=['date', 'food_item', 'quantity_sold'])
    df = df.sort_values('date')

    # Convert date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Feature Engineering
    df['day_num'] = (df['date'] - df['date'].min()).dt.days
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Lag features
    df['lag_1'] = df['quantity_sold'].shift(1)
    df['rolling_avg'] = df['quantity_sold'].rolling(3).mean()

    # Handle missing values
    df = df.bfill().ffill()
    return df


def predict_for_restaurant(data):
    predictions = {}
    model_info = {}
    items = data['food_item'].unique()

    for item in items:
        item_data = data[data['food_item'] == item]
        item_data = prepare_features(item_data)

        features = ['day_num', 'day_of_week', 'is_weekend', 'lag_1', 'rolling_avg']

        X = item_data[features]
        y = item_data['quantity_sold']

        # Train models
        lr = LinearRegression()
        rf = RandomForestRegressor(n_estimators=100, random_state=42)

        lr.fit(X, y)
        rf.fit(X, y)

        # Evaluate
        lr_pred = lr.predict(X)
        rf_pred = rf.predict(X)

        lr_error = mean_absolute_error(y, lr_pred)
        rf_error = mean_absolute_error(y, rf_pred)

        # Select best model
        best_model = lr if lr_error < rf_error else rf

        # 🔥 STORE MODEL INFO
        if lr_error < rf_error:
            model_info[item] = f"Linear Regression (MAE: {lr_error:.2f})"
        else:
            model_info[item] = f"Random Forest (MAE: {rf_error:.2f})"

        # Future prediction
        last_row = item_data.iloc[-1]
        future_preds = []
        future_dates = []

        current_lag = last_row['quantity_sold']

        for i in range(1, 8):
            future_date = last_row['date'] + timedelta(days=i)

            future_row = {
                'day_num': last_row['day_num'] + i,
                'day_of_week': future_date.dayofweek,
                'is_weekend': int(future_date.dayofweek in [5, 6]),
                'lag_1': current_lag,
                'rolling_avg': np.mean([current_lag] * 3)
            }

            pred = best_model.predict(pd.DataFrame([future_row]))[0]
            pred = max(0, int(round(pred)))

            future_preds.append(pred)
            future_dates.append(future_date.strftime('%Y-%m-%d'))

            current_lag = pred  # update lag

        predictions[item] = dict(zip(future_dates, future_preds))

    return predictions, model_info