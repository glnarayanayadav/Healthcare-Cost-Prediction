"""Feature engineering utilities."""

import numpy as np

def add_features(df):
    df = df.copy()
    
    if "age" in df.columns and "BMI" in df.columns:
        df["age_adjusted_bmi"] = df["BMI"] * np.sqrt(df["age"])
    
    if "smoker" in df.columns:
        df["smoker_numeric"] = df["smoker"].map({"yes":1, "no":0})
    
    if all(col in df.columns for col in ["BMI", "age", "smoker_numeric"]):
        df["compound_health_risk"] = (
            df["BMI"] * 0.4 +
            df["age"] * 0.3 +
            df["smoker_numeric"] * 0.3
        )
    
    return df
