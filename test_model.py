import joblib

def test_model():
    model = joblib.load("model.pkl")
    assert model is not None
    print("Model loaded successfully!")

test_model()