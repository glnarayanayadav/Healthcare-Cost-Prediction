"""Train classification model."""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier
from preprocessing import build_preprocessor
from feature_engineering import add_features
from config import TRAIN_CLS_PATH, RANDOM_STATE, CV_FOLDS

def train_classification():
    df = pd.read_csv(TRAIN_CLS_PATH)
    X = df.drop("estimated_treatment_cost", axis=1)
    y = df["estimated_treatment_cost"]

    X = add_features(X)
    preprocessor = build_preprocessor(X)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric="logloss"))
    ])

    scores = cross_validate(
        model,
        X,
        y,
        cv=CV_FOLDS,
        scoring=["f1_macro", "balanced_accuracy"]
    )

    print("Mean F1 Macro:", scores["test_f1_macro"].mean())
    print("Mean Balanced Accuracy:", scores["test_balanced_accuracy"].mean())

if __name__ == "__main__":
    train_classification()
