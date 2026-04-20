import streamlit as st
import pandas as pd
import plotly.express as px
from predictor import predict_for_restaurant
from utils import save_predictions

st.set_page_config(page_title="Food Usage Predictor", layout="centered", page_icon="🍽️")
st.title("🍽️ Restaurant Food Usage Predictor")

restaurant_name = st.text_input("Enter restaurant name (optional)", value="My Restaurant")

uploaded_file = st.file_uploader("Upload your sales CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        required_cols = {'date', 'food_item', 'quantity_sold'}
        if not required_cols.issubset(df.columns):
            st.error(f"CSV must include: {', '.join(required_cols)}")
        else:
            restaurant_data = df.copy()

            st.subheader(f"📋 Preview of Sales Data for {restaurant_name}")
            st.dataframe(restaurant_data.sample(10))

            if st.button("🔮 Predict Food Usage"):
                predictions = predict_for_restaurant(restaurant_data)
                st.success("✅ Predictions completed!")

                result_df = pd.DataFrame(predictions).T
                result_df.columns.name = "Date"

                st.subheader("📊 Predicted Quantities for Next 7 Days")
                st.dataframe(result_df)

                # Prepare for plotting
                melted = pd.DataFrame([
                    {"food_item": item, "date": date, "quantity": qty}
                    for item, data in predictions.items()
                    for date, qty in data.items()
                ])
                melted['date'] = pd.to_datetime(melted['date'])

                # Filter
                food_items = melted['food_item'].unique()
                selected_items = st.multiselect("Filter by food items", options=food_items, default=list(food_items))

                filtered = melted[melted['food_item'].isin(selected_items)]

                # Plot
                st.subheader("📈 Visualize Predictions")
                fig = px.line(filtered, x="date", y="quantity", color="food_item", markers=True)
                fig.update_layout(
                    title=f"Food Usage Prediction for {restaurant_name}",
                    xaxis_title="Date",
                    yaxis_title="Predicted Quantity"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Download
                csv_data = filtered.sort_values(["food_item", "date"]).to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv_data,
                    file_name=f"{restaurant_name}_predictions.csv",
                    mime="text/csv"
                )

                # Save JSON
                if st.button("💾 Save Predictions to JSON"):
                    save_predictions(predictions)
                    st.success("Predictions saved to predictions.json")

    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("Please upload a CSV to begin.")
