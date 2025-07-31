from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load("xgboost_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# Feature order
FEATURES = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
            'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        input_data = np.array([data[feature] for feature in FEATURES]).reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        prob = model.predict_proba(input_scaled)[0][1]
        is_fraud = prob >= 0.5

        return jsonify({
            "is_fraud": bool(is_fraud),
            "confidence": round(float(prob), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)

