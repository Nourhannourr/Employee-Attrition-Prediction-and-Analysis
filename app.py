from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="Employee Attrition Prediction API")

model = joblib.load("attrition_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")


@app.post("/predict")
def predict(employee_data: dict):
    # employee_data must already be in the same encoded numeric format
    # used during training (same columns, same encoding)
    df = pd.DataFrame([employee_data])
    df = df[feature_columns]  # enforce the exact training column order

    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    return {
        "attrition_prediction": "Yes" if prediction == 1 else "No",
        "attrition_probability": round(float(probability), 3)
    }
