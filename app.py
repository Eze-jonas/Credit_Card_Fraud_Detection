
import joblib
import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Relative Paths to Models
decision_tree_model_path = 'best_decision_tree_model.joblib'
logistic_regression_model_path = 'best_logistic_regression_model.joblib'
ann_model_path = 'best_ann_model.keras'

# Load Models
try:
    decision_tree_model = joblib.load(decision_tree_model_path)
    print("Decision Tree model loaded successfully.")
except Exception as e:
    print(f"Error loading Decision Tree model: {e}")
    decision_tree_model = None

try:
    logistic_regression_model = joblib.load(logistic_regression_model_path)
    print("Logistic Regression model loaded successfully.")
except Exception as e:
    print(f"Error loading Logistic Regression model: {e}")
    logistic_regression_model = None

try:
    ann_model = tf.keras.models.load_model(ann_model_path)
    print("ANN model loaded successfully.")
except Exception as e:
    print(f"Error loading ANN model: {e}")
    ann_model = None

# Home route to serve the index.html
@app.route('/')
def home():
    return render_template('index.html')  # Ensure index.html is in the templates folder

# Prediction route that accepts model name in the URL and JSON body
@app.route('/prediction/<model_name>', methods=['POST'])
def predict(model_name):
    try:
        # Get the JSON data from the request
        data = request.get_json()  # This will parse the JSON body
        features = data.get('features')

        if not features or len(features) != 23:
            return jsonify({"error": "Please provide exactly 23 features."}), 400

        features = np.array(features).reshape(1, -1)

        # Model Selection and Prediction Logic
        if model_name == 'decision_tree' and decision_tree_model:
            prediction = int(decision_tree_model.predict(features)[0])
        elif model_name == 'logistic_regression' and logistic_regression_model:
            prediction = int(logistic_regression_model.predict(features)[0])
        elif model_name == 'ann' and ann_model:
            prediction = (1 if ann_model.predict(features)[0][0] >= 0.5 else 0)
        else:
            return jsonify({"error": f"Model {model_name} not found or unavailable."}), 404

        return jsonify({
            f'{model_name}_prediction': prediction
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True)