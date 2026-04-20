import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import streamlit as st

st.set_page_config(page_title="Restaurant Order Predictor", layout="centered", page_icon="🍽️")

st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 2em;
    }
    .stButton>button:hover {
        background-color: #ff7878;
    }
    .stSlider>div>div>div {
        color: #333;
    }
    .stSelectbox>div>div>div {
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

data = {
    'day_of_week': np.random.randint(0, 7, 500),  # 0=Monday, 6=Sunday
    'is_holiday': np.random.randint(0, 2, 500),
    'temperature': np.random.normal(25, 5, 500),
    'rain_mm': np.random.exponential(1, 500),
    'special_event': np.random.randint(0, 2, 500),
    'previous_day_orders': np.random.randint(20, 200, 500),
    'predicted_orders': np.random.randint(30, 250, 500)
}
df = pd.DataFrame(data)

X = df.drop(columns='predicted_orders')
y = df['predicted_orders']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Mean Absolute Percentage Error
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_test, y_pred)
accuracy = 100 - mape

print(f"Accuracy: {accuracy:.2f}%")

joblib.dump(model, 'restaurant_food_predictor.pkl')

st.title("🍽️ Restaurant Food Order Predictor")
st.markdown("### Predict expected number of orders based on weather, holidays, and events.")
st.markdown("---")

day_of_week = st.selectbox("Day of the Week", list(range(7)), format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])
is_holiday = st.selectbox("Is it a holiday?", [0, 1], format_func=lambda x: "Yes" if x else "No")
temperature = st.slider("Temperature (°C)", 10.0, 40.0, 25.0)
rain_mm = st.slider("Rainfall (mm)", 0.0, 20.0, 0.5)
special_event = st.selectbox("Is there a special event?", [0, 1], format_func=lambda x: "Yes" if x else "No")
previous_day_orders = st.slider("Previous Day Orders", 0, 300, 100)

if st.button("🔮 Predict Orders"):
    input_data = pd.DataFrame([{
        'day_of_week': day_of_week,
        'is_holiday': is_holiday,
        'temperature': temperature,
        'rain_mm': rain_mm,
        'special_event': special_event,
        'previous_day_orders': previous_day_orders
    }])

    model = joblib.load('restaurant_food_predictor.pkl')
    prediction = model.predict(input_data)[0]
    st.success(f"🍔 Estimated Orders: **{int(prediction)}**")
    st.balloons()
