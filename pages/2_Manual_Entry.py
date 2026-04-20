import streamlit as st
import pandas as pd
import plotly.express as px
from predictor import predict_for_restaurant
from utils import save_predictions
from datetime import datetime

st.set_page_config(page_title="Food Usage Predictor", layout="centered", page_icon="🍽️")
st.title("🍽️ Restaurant Food Usage Predictor")

if 'sales_data' not in st.session_state:
    st.session_state.sales_data = pd.DataFrame(columns=["date", "food_item", "quantity_sold"])

if 'food_items' not in st.session_state:
    st.session_state.food_items = ["Burger", "Pizza", "Fries"]

st.sidebar.subheader("🍔 Manage Food Items")
new_item = st.sidebar.text_input("Add new food item")
if st.sidebar.button("Add Item") and new_item:
    if new_item not in st.session_state.food_items:
        st.session_state.food_items.append(new_item)
        st.sidebar.success(f"'{new_item}' added!")
    else:
        st.sidebar.warning("Item already exists.")

st.subheader("📆 Enter Daily Sales")
with st.form("sales_form"):
    date = st.date_input("Date", datetime.today())
    food_item = st.selectbox("Food Item", st.session_state.food_items)
    quantity = st.number_input("Quantity Sold", min_value=0, step=1)
    submitted = st.form_submit_button("Add Sale")

if submitted:
    new_row = pd.DataFrame([[date, food_item, quantity]], columns=["date", "food_item", "quantity_sold"])
    st.session_state.sales_data = pd.concat([st.session_state.sales_data, new_row], ignore_index=True)
    st.success("Sale added!")

st.subheader("📋 Current Sales Data")
st.dataframe(st.session_state.sales_data.tail(10))

if st.button("🔮 Predict Food Usage"):
    if st.session_state.sales_data.empty:
        st.warning("Please enter some sales data first.")
    else:
        predictions, model_info = predict_for_restaurant(st.session_state.sales_data)
        st.success("✅ Predictions completed!")

        total_demand = {item: sum(days.values()) for item, days in predictions.items()}
        top_item = max(total_demand, key=total_demand.get)
        top_value = total_demand[top_item]

        st.success(f"🔥 Highest Demand Item: {top_item} ({top_value} orders expected)")

        st.subheader("🤖 Model Selection Info")

        for item, info in model_info.items():
            st.write(f"{item}: {info}")

        result_df = pd.DataFrame(predictions).T
        result_df.columns.name = "Date"
        st.subheader("📊 Predicted Quantities for Next 7 Days")
        st.dataframe(result_df)

        melted = pd.DataFrame([
            {"food_item": item, "date": date, "quantity": qty}
            for item, data in predictions.items()
            for date, qty in data.items()
        ])
        melted['date'] = pd.to_datetime(melted['date'])

        selected_items = st.multiselect("Filter by food items", options=melted['food_item'].unique(), default=list(melted['food_item'].unique()))
        filtered = melted[melted['food_item'].isin(selected_items)]

        st.subheader("📈 Prediction Chart")
        fig = px.line(filtered, x="date", y="quantity", color="food_item", markers=True)
        fig.update_layout(title="Food Usage Prediction", xaxis_title="Date", yaxis_title="Predicted Quantity")
        st.plotly_chart(fig, use_container_width=True)

        csv_data = filtered.sort_values(["food_item", "date"]).to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv_data,
            file_name="predictions.csv",
            mime="text/csv"
        )

        if st.button("💾 Save Predictions to JSON"):
            save_predictions(predictions)
            st.success("Predictions saved to predictions.json")
