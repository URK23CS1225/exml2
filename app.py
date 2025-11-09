from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# ✅ Load both ML models
models = {
    "random_forest": joblib.load("models/random_forest.pkl"),
    "logistic_regression": joblib.load("models/logistic_regression.pkl")
}


@app.route("/")
def home():
    return jsonify({
        "message": "ML APIs are running ✅",
        "usage": {
            "predict with Random Forest": "/predict/random_forest",
            "predict with Logistic Regression": "/predict/logistic_regression",
            "dynamic model select": "/predict/model?model_name=random_forest"
        }
    })


# ✅ Endpoint 1: Predict using Random Forest
@app.route("/predict/random_forest", methods=["POST"])
def predict_rf():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = models["random_forest"].predict(features)
    return jsonify({"model": "Random Forest", "prediction": prediction.tolist()})


# ✅ Endpoint 2: Predict using Logistic Regression
@app.route("/predict/logistic_regression", methods=["POST"])
def predict_lr():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = models["logistic_regression"].predict(features)
    return jsonify({"model": "Logistic Regression", "prediction": prediction.tolist()})


# ✅ Endpoint 3 (Dynamic selection — required by Q3)
# Example:  /predict/model?model_name=random_forest
@app.route("/predict/model", methods=["POST"])
def predict_model():
    model_name = request.args.get("model_name")

    if model_name not in models:
        return jsonify({"error": "Invalid model name", "available_models": list(models.keys())}), 400

    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = models[model_name].predict(features)

    return jsonify({"model": model_name, "prediction": prediction.tolist()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
