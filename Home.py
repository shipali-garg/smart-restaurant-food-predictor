import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Restaurant Predictor Hub", layout="centered", page_icon="🍽️")
st.markdown("""
    <style>
    .title {
        font-size: 2.6rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        font-size: 1.2rem;
        margin-bottom: 1.8rem;
    }
    .emoji {
        font-size: 1.3rem;
        margin-right: 0.5rem;
    }
    .kpi-card {
        background: #fff;
        padding: 1rem;
        border-radius: 1rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        text-align: center;
        margin: 0.5rem;
    }
    .kpi-title {
        font-size: 0.9rem;
        color: #888;
    }
    .kpi-value {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🍽️ Restaurant Prediction Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smarter Orders. Less Waste. More Profit.</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="kpi-card"><div class="kpi-title">Today\'s Temp</div><div class="kpi-value">34°C ☀️</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">Current Time</div><div class="kpi-value">{datetime.now().strftime("%H:%M:%S")}</div></div>', unsafe_allow_html=True)

st.write("")
st.markdown("Welcome! Choose one of the prediction methods below to continue:")

st.page_link("pages/1_Weather_Event_Predictor.py", label="🌦️ Weather & Event-Based Orders")
st.page_link("pages/2_Manual_Entry.py", label="✍️ Manual Sales Entry")
st.page_link("pages/3_Upload_csv.py", label="📁 Upload Sales CSV")
st.page_link("pages/4_ARIMA.py", label="📁 Upload Sales CSV (using ARIMA)")


st.write("")
st.markdown("""
#### 💡 Quick Tip:
Use the **Weather Predictor** if you expect unusual weather or events like festivals.  
Upload past sales via CSV to train more accurate models.
""")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("© 2025 SmartKitchen AI")

