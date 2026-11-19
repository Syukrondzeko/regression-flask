# regression-flask

[![Tests](https://github.com/Syukrondzeko/regression-flask/actions/workflows/tests.yml/badge.svg)](https://github.com/Syukrondzeko/regression-flask/actions/workflows/tests.yml)

A Flask web app that serves a regression model. It predicts a patient's hospital
**billing amount** from four inputs:

| Input | Accepted values |
|---|---|
| Age | 1–120 |
| Gender | Female, Male |
| Blood Type | A+, A-, AB+, AB-, B+, B-, O+, O- |
| Medical Condition | Arthritis, Asthma, Cancer, Diabetes, Hypertension, Obesity |

The model is a scikit-learn `DecisionTreeRegressor`; the categorical inputs are
one-hot encoded by a fitted `ColumnTransformer`. The form's dropdown options are
read from that transformer at startup, so the UI always matches what the model
was trained on.

## Dataset

Trained on the [Healthcare Dataset](https://www.kaggle.com/datasets/prasad22/healthcare-dataset)
from Kaggle, which is synthetic, so predictions are for demonstration only.

## Project layout

```
app.py                       Flask app: form page (/) and prediction endpoint (POST /predict)
train.py                     Retrains the model and regenerates the .pkl files
transformer.pkl              Fitted one-hot encoder (ColumnTransformer)
decision_tree_regressor.pkl  Trained model
templates/health_bill.html   Page template (Bootstrap 5)
static/css/style.css         Custom styles
tests/                       pytest tests
Procfile                     Heroku start command (gunicorn)
```

## Run locally

Requires Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                      # http://127.0.0.1:5000
```

Set `FLASK_DEBUG=1` to turn on the Flask debugger during development. Never
enable it on a public deployment.

## Run the tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests also run on GitHub Actions for every push to `main` and every pull request.

## Retrain the model

Download the dataset CSV, then:

```bash
python train.py path/to/healthcare_dataset.csv
```

This prints test-set MAE and R² and overwrites both `.pkl` files.

**Library versions matter.** Pickled scikit-learn models only load reliably with
the version that saved them. The current files were saved with scikit-learn
1.3.2, which is why `requirements.txt` pins exact versions. To upgrade
scikit-learn, update `requirements.txt`, retrain with `train.py`, and commit the
new `.pkl` files together.

## Deploy

The `Procfile` runs the app with gunicorn (`web: gunicorn app:app`), and
`.python-version` sets the Python version, so it deploys to Heroku as is.
