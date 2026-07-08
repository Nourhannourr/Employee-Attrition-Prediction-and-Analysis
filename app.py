import streamlit as st
import pandas as pd
import joblib

# 1. Load Assets (Model, Scaler, and Feature Columns)
@st.cache_resource 
def load_assets():
    model = joblib.load("attrition_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns

model, scaler, feature_columns = load_assets()

# 2. Prediction Logic
def predict_attrition(age, monthly_income, overtime, total_working_years, years_at_company, job_satisfaction, env_satisfaction):
    
    # Initialize dictionary with all zeros for the 34 columns
    input_row = {col: 0 for col in feature_columns}
    
    # Update dictionary with user inputs
    input_row["Age"] = age
    input_row["MonthlyIncome"] = monthly_income
    input_row["OverTime"] = 1 if overtime == "Yes" else 0
    input_row["TotalWorkingYears"] = total_working_years
    input_row["YearsAtCompany"] = years_at_company
    input_row["JobSatisfaction"] = job_satisfaction
    input_row["EnvironmentSatisfaction"] = env_satisfaction
    
    # Feature Engineering (Salary Experience Ratio)
    input_row["SalaryExperienceRatio"] = monthly_income / (total_working_years + 1)
    
    # Convert to DataFrame ensuring correct column order
    input_df = pd.DataFrame([input_row])[feature_columns]
    
    # Apply Scaling
    scaled_data = scaler.transform(input_df)
    
    # Prediction
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1] * 100 
    
    if prediction == 1:
        return True, f"⚠️ High Risk: The employee is likely to leave ({probability:.1f}% probability)."
    else:
        return False, f"✅ Low Risk: The employee is likely to stay ({probability:.1f}% probability)."

# 3. User Interface Design
st.set_page_config(page_title="Employee Attrition Predictor", page_icon="💼", layout="centered")

st.title("💼 Employee Attrition Predictor")
st.markdown("Enter the employee's details below to predict their likelihood of leaving the company. This tool uses an advanced XGBoost machine learning model.")

st.markdown("---")

# Two-column layout for a professional look
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=60, value=30, step=1)
    monthly_income = st.number_input("Monthly Income ($)", min_value=0, value=5000, step=100)
    overtime = st.radio("Works OverTime?", options=["Yes", "No"], index=1, horizontal=True)
    total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=5, step=1)

with col2:
    years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=3, step=1)
    job_satisfaction = st.number_input("Job Satisfaction (1: Low - 4: Excellent)", min_value=1, max_value=4, value=3, step=1)
    env_satisfaction = st.number_input("Environment Satisfaction (1: Low - 4: Excellent)", min_value=1, max_value=4, value=3, step=1)

st.markdown("---")

# Prediction Button
if st.button("Predict Attrition", use_container_width=True):
    is_leaving, message = predict_attrition(
        age, monthly_income, overtime, total_working_years, 
        years_at_company, job_satisfaction, env_satisfaction
    )
    
    # Display results
    if is_leaving:
        st.error(message)
    else:
        st.success(message)
