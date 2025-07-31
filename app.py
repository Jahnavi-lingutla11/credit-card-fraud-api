from flask import Flask, request, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("xgboost_fraud_model.joblib")
scaler = joblib.load("scaler.joblib")

HTML_TEMPLATE = '''
<!doctype html>
<html>
<head>
    <title>Credit Card Fraud Detection</title>
</head>
<body>
    <h2>💳 Credit Card Fraud Detection</h2>
    <form method="POST" action="/predict">
        {% for i in range(1, 29) %}
            V{{ i }}: <input type="text" name="V{{ i }}" value="{{ values['V' ~ i] }}"><br><br>
        {% endfor %}
        Amount: <input type="text" name="Amount" value="{{ values['Amount'] }}"><br><br>
        <input type="submit" value="Check">
    </form>

    {% if prediction is not none %}
        <h3>🔍 Prediction: {{ 'Fraudulent' if prediction == 1 else 'Not Fraudulent' }}</h3>
    {% endif %}
</body>
</html>
'''

@app.route('/')
def home():
    # Initialize with empty strings for each field
    values = {f'V{i}': '' for i in range(1, 29)}
    values['Amount'] = ''
    return render_template_string(HTML_TEMPLATE, prediction=None, values=values)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values safely
        values = {f'V{i}': request.form.get(f'V{i}', '').strip() for i in range(1, 29)}
        values['Amount'] = request.form.get('Amount', '').strip()

        # Convert to float
        try:
            features = [float(values[f'V{i}']) for i in range(1, 29)]
            features.append(float(values['Amount']))
        except ValueError:
            return render_template_string(HTML_TEMPLATE, prediction=None, values=values) + "<p style='color:red;'>❌ Please enter valid numbers for all fields.</p>"

        # Model prediction
        data = np.array(features).reshape(1, -1)
        data_scaled = scaler.transform(data)
        prediction = int(model.predict(data_scaled)[0])

        return render_template_string(HTML_TEMPLATE, prediction=prediction, values=values)

    except Exception as e:
        return f"<h3>🚨 Error: {str(e)}</h3>"

if __name__ == '__main__':
    app.run(debug=True)





