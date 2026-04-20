# Smart Restaurant Food Usage Predictor

## Overview

This project is a **Streamlit-based web application** designed to help restaurants **predict food demand** and reduce waste.
It uses machine learning models and time-series forecasting techniques to estimate future food requirements.

## Features

* **Manual Prediction** – Enter inputs and get instant predictions
* **CSV Upload** – Upload bulk data for batch predictions
* **Weather-Based Prediction** – Incorporates weather conditions
* **ARIMA Forecasting** – Time-series forecasting for future demand
* **Interactive UI** – Built using Streamlit for easy use

## Tech Stack

* Python
* Streamlit
* Pandas
* Scikit-learn
* Plotly
* ARIMA (statsmodels)

## Project Structure

```
Restaurant_prediction/
│
├── pages/                     # Multi-page Streamlit modules
├── Home.py                   # Main entry point
├── predictor.py              # ML prediction logic
├── utils.py                  # Helper functions
├── sample_data.csv           # Sample dataset
├── restaurant_food_predictor.pkl  # Trained ML model
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
```

## How to Run the Project

### 1. Clone the repository

```
git clone https://github.com/your-username/repo-name.git
cd repo-name
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the app

```
streamlit run Home.py
```

## Use Case

This system can help:

* Restaurants optimize inventory
* Reduce food wastage
* Improve operational efficiency
* Make data-driven decisions

## Future Improvements

* Deploy the app online (Streamlit Cloud / AWS)
* Add real-time data integration
* Improve model accuracy with advanced algorithms
* Add user authentication

## Author

* Shipali Garg
Developed as part of a machine learning project to solve real-world restaurant demand prediction problems.



