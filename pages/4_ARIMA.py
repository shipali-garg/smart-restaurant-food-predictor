import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

st.set_page_config(page_title="ARIMA Food Predictor", layout="centered", page_icon="🍽️")

st.title("🍽️ Restaurant Food Usage Predictor (ARIMA)")

# -----------------------------
# 🔧 Helper Functions
# -----------------------------

def prepare_time_series(df, food_item):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    item_df = df[df['food_item'] == food_item]
    ts = item_df.groupby('date')['quantity_sold'].sum()

    return ts


def check_stationarity(ts):
    result = adfuller(ts.dropna())
    return result[1] < 0.05  # p-value < 0.05 → stationary


def forecast_with_arima(df, food_item, steps=7):
    ts = prepare_time_series(df, food_item)

    # Handle stationarity
    if not check_stationarity(ts):
        ts = ts.diff().dropna()

    # ARIMA parameters
    p, d, q = 3, 1, 1

    model = ARIMA(ts, order=(p, d, q))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=steps)

    # Future dates
    last_date = ts.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps)

    forecast_df = pd.DataFrame({
        'date': future_dates,
        'predicted_sales': forecast.values
    })

    forecast_df['predicted_sales'] = forecast_df['predicted_sales'].clip(lower=0)

    return forecast_df


# -----------------------------
# 📂 File Upload
# -----------------------------

uploaded_file = st.file_uploader("Upload your sales CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # 🔥 Validation
        required_cols = {'date', 'food_item', 'quantity_sold'}
        if not required_cols.issubset(df.columns):
            st.error(f"CSV must contain: {', '.join(required_cols)}")
            st.stop()

        st.subheader("📋 Preview of Uploaded Data")
        st.dataframe(df.sample(10))

        # Select item
        food_items = df['food_item'].unique()
        selected_food_item = st.selectbox("Select Food Item", food_items)

        forecast_days = st.slider("Forecast Days", 1, 30, 7)

        if st.button("🔮 Run ARIMA Forecast"):

            forecast_df = forecast_with_arima(df, selected_food_item, forecast_days)

            st.success("✅ Forecast completed!")

            # -----------------------------
            # 🤖 Model Info
            # -----------------------------
            st.subheader("🤖 Model Info")
            st.write("Model: ARIMA")
            st.write("Configuration: (p=3, d=1, q=1)")
            st.write("Stationarity handled using differencing")

            # -----------------------------
            # 📊 Output Table
            # -----------------------------
            st.subheader(f"📊 Forecast for {selected_food_item}")
            st.dataframe(forecast_df)

            # -----------------------------
            # 📈 Trend Insight
            # -----------------------------
            if len(forecast_df) > 1:
                if forecast_df['predicted_sales'].iloc[-1] > forecast_df['predicted_sales'].iloc[0]:
                    st.success("📈 Increasing demand trend predicted")
                else:
                    st.warning("📉 Decreasing demand trend predicted")

            # -----------------------------
            # 📊 Summary Stat
            # -----------------------------
            avg_prediction = int(forecast_df['predicted_sales'].mean())
            st.info(f"📊 Average predicted daily sales: {avg_prediction}")

            # -----------------------------
            # 📈 Visualization
            # -----------------------------
            st.subheader("📈 Forecast Visualization")

            # 🔥 FIXED HISTORICAL DATA
            hist_df = df[df['food_item'] == selected_food_item].copy()

            hist_df['date'] = pd.to_datetime(hist_df['date'])
            hist_df = hist_df.sort_values('date')

            historical_data = hist_df.groupby('date')['quantity_sold'].sum()

            # Safety check
            if historical_data.empty:
                st.error("No historical data available for this item")
                st.stop()

            fig, ax = plt.subplots()

            # Convert forecast date
            forecast_df['date'] = pd.to_datetime(forecast_df['date'])

            # Plot historical
            ax.plot(
                historical_data.index,
                historical_data.values,
                label='Historical Sales'
            )

            # Plot forecast
            ax.plot(
                forecast_df['date'],
                forecast_df['predicted_sales'],
                label='Forecasted Sales',
                linestyle='--'
            )

            ax.set_title(f"Sales Forecast for {selected_food_item}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Quantity Sold")
            ax.legend()

            st.pyplot(fig)

            # -----------------------------
            # 📥 Download
            # -----------------------------
            csv_data = forecast_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast CSV",
                data=csv_data,
                file_name=f"{selected_food_item}_forecast.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Please upload a CSV to begin.")