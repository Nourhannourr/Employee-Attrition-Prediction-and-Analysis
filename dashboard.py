import streamlit as st
import joblib
import pandas as pd

model = joblib.load("attrition_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("Employee Attrition Risk Predictor")
st.write("Enter employee details to estimate attrition risk.")

age = st.slider("Age", 18, 60, 30)
monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
overtime = st.selectbox("OverTime", ["Yes", "No"])
total_working_years = st.slider("Total Working Years", 0, 40, 5)
job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)

if st.button("Predict"):
    # Build a single-row input matching the training feature columns.
    # Any column not collected from the UI is filled with a default (0)
    # -- replace with real form fields for a production version.
    input_row = {col: 0 for col in feature_columns}
    input_row["Age"] = age
    input_row["MonthlyIncome"] = monthly_income
    input_row["OverTime"] = 1 if overtime == "Yes" else 0
    input_row["TotalWorkingYears"] = total_working_years
    input_row["JobSatisfaction"] = job_satisfaction

    input_df = pd.DataFrame([input_row])[feature_columns]
    scaled = scaler.transform(input_df)

    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    st.metric("Attrition Risk", f"{prob:.1%}")
    st.write("High Risk" if pred == 1 else "Low Risk")
