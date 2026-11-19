import os
import pickle

import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "decision_tree_regressor.pkl")
TRANSFORMER_PATH = os.path.join(BASE_DIR, "transformer.pkl")

app = Flask(__name__)

# Load the trained model and the fitted preprocessing transformer
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(TRANSFORMER_PATH, "rb") as f:
    transformer = pickle.load(f)

# The category values the model was trained on, read from the fitted encoder so
# the form options can never drift out of sync with the model.
_encoder = transformer.named_transformers_["one_hot"]
CHOICES = {
    column: [str(value) for value in categories]
    for column, categories in zip(_encoder.feature_names_in_, _encoder.categories_)
}

MIN_AGE = 1
MAX_AGE = 120


def render_page(pred=None, error=None, form=None, status=200):
    return (
        render_template(
            "health_bill.html",
            choices=CHOICES,
            pred=pred,
            error=error,
            form=form or {},
            min_age=MIN_AGE,
            max_age=MAX_AGE,
        ),
        status,
    )


def validate(form):
    """Return (features, error). Exactly one of them is None."""
    age = form.get("Age", type=int)
    if age is None or not MIN_AGE <= age <= MAX_AGE:
        return None, f"Please enter an age between {MIN_AGE} and {MAX_AGE}."

    features = {"Age": age}
    for field, column in (
        ("Gender", "Gender"),
        ("BloodType", "Blood Type"),
        ("MedicalCondition", "Medical Condition"),
    ):
        value = form.get(field)
        if value not in CHOICES[column]:
            return None, f"Please select a valid {column.lower()}."
        features[column] = value

    return features, None


@app.route("/")
def index():
    return render_page()


@app.route("/predict", methods=["POST"])
def predict():
    features, error = validate(request.form)
    if error:
        return render_page(error=error, form=request.form, status=400)

    transformed = transformer.transform(pd.DataFrame([features]))
    prediction = float(model.predict(transformed)[0])

    return render_page(pred=f"{prediction:,.2f}", form=request.form)


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
