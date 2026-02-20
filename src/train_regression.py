"""Train regression model."""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from xgboost import XGBRegressor
from preprocessing import build_preprocessor
from feature_engineering import add_features
from config import TRAIN_REG_PATH, RANDOM_STATE, CV_FOLDS

def train_regression():
    df = pd.read_csv(TRAIN_REG_PATH)
    X = df.drop("estimated_treatment_cost", axis=1)
    y = df["estimated_treatment_cost"]

    X = add_features(X)
    preprocessor = build_preprocessor(X)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(random_state=RANDOM_STATE))
    ])

    scores = cross_validate(
        model,
        X,
        y,
        cv=CV_FOLDS,
        scoring=["r2", "neg_root_mean_squared_error"]
    )

    print("Mean R2:", scores["test_r2"].mean())
    print("Mean RMSE:", -scores["test_neg_root_mean_squared_error"].mean())

if __name__ == "__main__":
    train_regression()
