import streamlit as st
import pandas as pd
import joblib


@st.cache_resource
def load_assets():
    model = joblib.load("attrition_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    default_values = joblib.load("default_values.pkl")
    best_threshold = joblib.load("best_threshold.pkl")
    tenure_bins = joblib.load("tenure_bins.pkl")
    tenure_encoder = joblib.load("tenure_encoder.pkl")
    salary_bin_edges = joblib.load("salary_bin_edges.pkl")
    salary_encoder = joblib.load("salary_encoder.pkl")
    return (model, scaler, feature_columns, default_values, best_threshold,
            tenure_bins, tenure_encoder, salary_bin_edges, salary_encoder)


(model, scaler, feature_columns, default_values, best_threshold,
 tenure_bins, tenure_encoder, salary_bin_edges, salary_encoder) = load_assets()


def compute_tenure_group(years_at_company):
    """Rebuild TenureGroup the same way it was built during training."""
    clipped = min(max(years_at_company, tenure_bins[0] + 1e-9), tenure_bins[-1])
    label = pd.cut([clipped], bins=tenure_bins,
                    labels=['New', 'Junior', 'Mid', 'Senior'])[0]
    return tenure_encoder.transform([label])[0]


def compute_salary_band(monthly_income):
    """Rebuild SalaryBand using the quantile edges captured at training time."""
    clipped = min(max(monthly_income, salary_bin_edges[0]), salary_bin_edges[-1])
    label = pd.cut([clipped], bins=salary_bin_edges,
                    labels=['Low', 'Medium', 'High'], include_lowest=True)[0]
    return salary_encoder.transform([label])[0]


def predict_attrition(age, monthly_income, overtime, total_working_years,
                       years_at_company, job_satisfaction, env_satisfaction):

    input_row = default_values.copy()

    input_row["Age"] = age
    input_row["MonthlyIncome"] = monthly_income
    input_row["OverTime"] = 1 if overtime == "Yes" else 0
    input_row["TotalWorkingYears"] = total_working_years
    input_row["YearsAtCompany"] = years_at_company
    input_row["JobSatisfaction"] = job_satisfaction
    input_row["EnvironmentSatisfaction"] = env_satisfaction
    input_row["SalaryExperienceRatio"] = monthly_income / (total_working_years + 1)

    # These two used to silently stay at their training-set default values,
    # contradicting whatever the user entered above. Now they track the
    # actual inputs.
    if "TenureGroup" in feature_columns:
        input_row["TenureGroup"] = compute_tenure_group(years_at_company)
    if "SalaryBand" in feature_columns:
        input_row["SalaryBand"] = compute_salary_band(monthly_income)

    input_df = pd.DataFrame([input_row])[feature_columns]
    scaled_data = scaler.transform(input_df)

    probability = model.predict_proba(scaled_data)[0][1] * 100
    prediction = 1 if (probability / 100) >= best_threshold else 0

    if prediction == 1:
        return True, f"⚠️ High Risk: The employee is likely to leave. (probability: {probability:.1f}%)"
    else:
        return False, f"✅ Low Risk: The employee is likely to stay. (probability: {probability:.1f}%)"


st.set_page_config(page_title="Employee Attrition Predictor", page_icon="💼", layout="centered")
st.title("💼 Employee Attrition Predictor")
st.markdown("Enter the employee's details below to predict their likelihood of leaving the company. This tool uses an advanced XGBoost machine learning model.")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=60, value=30, step=1)
    monthly_income = st.number_input("Monthly Income ($)", min_value=0, max_value=20000, value=5000, step=100)
    overtime = st.radio("Works OverTime?", options=["Yes", "No"], index=1, horizontal=True)
    total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=5, step=1)
with col2:
    years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=3, step=1)
    job_satisfaction = st.number_input("Job Satisfaction (1: Low - 4: Excellent)", min_value=1, max_value=4, value=3, step=1)
    env_satisfaction = st.number_input("Environment Satisfaction (1: Low - 4: Excellent)", min_value=1, max_value=4, value=3, step=1)

st.markdown("---")
if st.button("Predict Attrition", use_container_width=True):
    is_leaving, message = predict_attrition(
        age, monthly_income, overtime, total_working_years,
        years_at_company, job_satisfaction, env_satisfaction
    )

    if is_leaving:
        st.error(message)
    else:
        st.success(message)
