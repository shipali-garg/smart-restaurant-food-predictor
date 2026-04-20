import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Weather-Based Predictor", layout="centered", page_icon="🌦️")

st.title("🌦️ Weather & Event-Based Food Order Predictor")

st.markdown("Predict restaurant orders using weather conditions and events.")

# -----------------------------
# 📊 REALISTIC SAMPLE DATA (NOT RANDOM)
# -----------------------------

data = pd.DataFrame({
    "temperature": [20, 22, 25, 30, 35, 38, 28, 26, 24, 32],
    "rain":        [0,  2,  5,  0,  0, 10,  0,  3,  1,  0],
    "is_holiday":  [0,  0,  1,  0,  1,  0,  0,  1,  0,  0],
    "event":       [0,  1,  0,  0,  1,  0,  0,  1,  0,  1],
    "prev_orders": [80, 90, 100, 120, 150, 140, 110, 130, 95, 125],
    "orders":      [85, 105, 130, 140, 190, 180, 125, 170, 110, 160]
})

X = data[["temperature", "rain", "is_holiday", "event", "prev_orders"]]
y = data["orders"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# -----------------------------
# 📥 USER INPUT
# -----------------------------

st.subheader("📥 Enter Conditions")

temperature = st.slider("Temperature (°C)", 10, 45, 30)
rain = st.slider("Rainfall (mm)", 0, 50, 0)
is_holiday = st.selectbox("Is it a holiday?", [0, 1], format_func=lambda x: "Yes" if x else "No")
event = st.selectbox("Special Event?", [0, 1], format_func=lambda x: "Yes" if x else "No")
prev_orders = st.slider("Previous Day Orders", 0, 500, 120)

# -----------------------------
# 🔮 PREDICTION
# -----------------------------

if st.button("🔮 Predict Orders"):

    input_data = pd.DataFrame([{
        "temperature": temperature,
        "rain": rain,
        "is_holiday": is_holiday,
        "event": event,
        "prev_orders": prev_orders
    }])

    prediction = int(model.predict(input_data)[0])

    st.success(f"🍔 Estimated Orders: {prediction}")

    # -----------------------------
    # 📊 INSIGHTS
    # -----------------------------
    if prediction > prev_orders:
        st.info("📈 Expected increase in demand")
    else:
        st.warning("📉 Possible decrease in demand")

    # -----------------------------
    # 🤖 MODEL INFO
    # -----------------------------
    st.subheader("🤖 Model Info")
    st.write("Model: Random Forest Regressor")
    st.write("Features used: temperature, rain, holiday, event, previous orders")

    # Feature importance
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.subheader("📊 Feature Importance")
    st.dataframe(importance)