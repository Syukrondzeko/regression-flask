import pytest

from app import CHOICES, app

VALID_FORM = {
    "Age": "40",
    "Gender": "Male",
    "BloodType": "A+",
    "MedicalCondition": "Diabetes",
}


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_model_categories_loaded():
    assert CHOICES["Gender"] == ["Female", "Male"]
    assert CHOICES["Blood Type"] == ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]
    assert len(CHOICES["Medical Condition"]) == 6


def test_index_renders_all_model_categories(client):
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    for options in CHOICES.values():
        for option in options:
            assert f'value="{option}"' in html


def test_stylesheet_is_served(client):
    html = client.get("/").get_data(as_text=True)
    assert "/static/css/style.css" in html
    assert client.get("/static/css/style.css").status_code == 200


def test_valid_prediction(client):
    response = client.post("/predict", data=VALID_FORM)
    assert response.status_code == 200
    assert "Predicted Billing Amount is" in response.get_data(as_text=True)


def test_blood_type_affects_prediction(client):
    """Guards against the form sending values the model ignores."""
    a_pos = client.post("/predict", data=VALID_FORM).get_data(as_text=True)
    a_neg = client.post("/predict", data={**VALID_FORM, "BloodType": "A-"}).get_data(as_text=True)
    assert a_pos != a_neg


@pytest.mark.parametrize(
    "override",
    [
        {"Age": ""},
        {"Age": "0"},
        {"Age": "abc"},
        {"Age": "500"},
        {"Gender": "Other"},
        {"BloodType": "A"},
        {"MedicalCondition": "diabetes"},
        {"MedicalCondition": ""},
    ],
)
def test_invalid_input_is_rejected(client, override):
    response = client.post("/predict", data={**VALID_FORM, **override})
    assert response.status_code == 400
    assert "Predicted Billing Amount" not in response.get_data(as_text=True)


def test_predict_rejects_get(client):
    assert client.get("/predict").status_code == 405
