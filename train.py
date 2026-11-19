"""Train the billing-amount regression model and save it for the Flask app.

Usage:
    python train.py path/to/healthcare_dataset.csv

The CSV needs the columns: Age, Gender, Blood Type, Medical Condition,
Billing Amount. Running this overwrites transformer.pkl and
decision_tree_regressor.pkl in the project folder.
"""
import argparse
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["Age", "Gender", "Blood Type", "Medical Condition"]
CATEGORICAL = ["Gender", "Blood Type", "Medical Condition"]
TARGET = "Billing Amount"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", help="path to the training data CSV")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, usecols=FEATURES + [TARGET]).dropna()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=args.test_size, random_state=args.random_state
    )

    transformer = ColumnTransformer(
        [("one_hot", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
    )
    model = DecisionTreeRegressor(random_state=args.random_state)
    model.fit(transformer.fit_transform(X_train), y_train)

    y_pred = model.predict(transformer.transform(X_test))
    print(f"Test MAE: {mean_absolute_error(y_test, y_pred):,.2f}")
    print(f"Test R2:  {r2_score(y_test, y_pred):.4f}")

    for name, obj in (("transformer.pkl", transformer), ("decision_tree_regressor.pkl", model)):
        with open(os.path.join(BASE_DIR, name), "wb") as f:
            pickle.dump(obj, f)
    print("Saved transformer.pkl and decision_tree_regressor.pkl")


if __name__ == "__main__":
    main()
