from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['input']
    pred = model.predict([np.array(data)])
    return jsonify({"prediction": int(pred[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)