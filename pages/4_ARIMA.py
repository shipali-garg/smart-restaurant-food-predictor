import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Function for loading and preparing data
def prepare_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df


# ARIMA model for forecasting food sales
def forecast_with_arima(df, food_item, forecast_days=7):
    food_data = df[df['food_item'] == food_item]
    daily_sales = food_data.groupby('date')['quantity_sold'].sum()
    model = ARIMA(daily_sales, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_days)
    forecast_dates = pd.date_range(start=daily_sales.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'predicted_sales': forecast
    })
    return forecast_df

st.title("🍽️ Restaurant Food Usage Predictor with ARIMA")

uploaded_file = st.file_uploader("Upload your sales CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df = prepare_data(df)

        st.subheader("📋 Preview of Uploaded Data")
        st.dataframe(df.sample(10))

        # Select food item for ARIMA forecasting
        food_items = df['food_item'].unique()
        selected_food_item = st.selectbox("Select Food Item for ARIMA Forecast", food_items)

        if selected_food_item:
            forecast_days = st.slider("Number of Days to Forecast", 1, 30, 7)
            forecast_df = forecast_with_arima(df, selected_food_item, forecast_days)

            st.subheader(f"📊 Predicted Sales for Next {forecast_days} Days for {selected_food_item}")
            st.dataframe(forecast_df)

            st.subheader(f"📈 Sales Forecast for {selected_food_item}")

            # Group data for the selected food item
            historical_data = (
                df[df['food_item'] == selected_food_item]
                .groupby('date')['quantity_sold']
                .sum()
            )

            # Plotting
            fig, ax = plt.subplots()
            ax.plot(historical_data.index, historical_data.values, label='Historical Sales')
            ax.plot(forecast_df['date'], forecast_df['predicted_sales'], label='Forecasted Sales', color='red')

            ax.set_title(f"Sales Forecast for {selected_food_item}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Quantity Sold")
            ax.legend()
            st.pyplot(fig)

            # Allow users to download the forecasted data
            csv_data = forecast_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecasted Sales as CSV",
                data=csv_data,
                file_name=f"{selected_food_item}_forecast.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("Please upload a CSV to begin.")
