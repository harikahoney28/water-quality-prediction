import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Water Quality Index Prediction",
    page_icon="💧",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------
st.title("💧 Water Quality Index (WQI) Prediction")
st.write("Linear Regression based Water Quality Prediction")


# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Results_MADE.csv")
    return df


df = load_data()


# -----------------------------
# Prepare Data
# -----------------------------
X = df.drop("WQI", axis=1)
y = df["WQI"]


# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# -----------------------------
# Train Linear Regression
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)


# -----------------------------
# Model Prediction
# -----------------------------
y_pred = model.predict(X_test)


# -----------------------------
# Model Evaluation
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


# -----------------------------
# Dataset Information
# -----------------------------
st.subheader("📊 Dataset Information")

st.write("Number of observations:", df.shape[0])
st.write("Number of features:", X.shape[1])

st.dataframe(df.head())


# -----------------------------
# Model Performance
# -----------------------------
st.subheader("📈 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAE", round(mae, 2))

with col2:
    st.metric("RMSE", round(rmse, 2))

with col3:
    st.metric("R² Score", round(r2, 2))


# -----------------------------
# WQI Prediction
# -----------------------------
st.subheader("🔮 Predict Water Quality Index")


st.write("Enter the water-quality parameters:")


input_values = {}

for column in X.columns:
    input_values[column] = st.number_input(
        column,
        value=float(X[column].mean())
    )


# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict WQI"):

    input_data = pd.DataFrame([input_values])

    prediction = model.predict(input_data)

    predicted_wqi = prediction[0]

    st.success(
        f"Predicted Water Quality Index (WQI): {predicted_wqi:.2f}"
    )


# -----------------------------
# Actual vs Predicted
# -----------------------------
st.subheader("📌 Actual vs Predicted WQI")

comparison = pd.DataFrame({
    "Actual WQI": y_test.values,
    "Predicted WQI": y_pred
})

st.dataframe(comparison.head(10))
