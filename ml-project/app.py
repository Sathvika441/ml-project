from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Fertilizer mapping
fertilizer_dict = {
    0: 'Urea',
    1: 'DAP',
    2: 'Fourteen-Thirty Five-Fourteen',
    3: 'Twenty Eight-Twenty Eight',
    4: 'Seventeen-Seventeen-Seventeen',
    5: 'Twenty-Twenty',
    6: 'Ten-Twenty Six-Twenty Six'
}


# Load Random Forest model
def load_model():
    try:
        rf_model_path = os.path.join(
            BASE_DIR,
            "models",
            "random_forest_model.pkl"
        )

        with open(rf_model_path, "rb") as f:
            rf_model = pickle.load(f)

        return rf_model

    except Exception as e:
        print(f"Error loading model: {e}")
        raise


# Load model once at startup
rf_model = load_model()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    try:
        # Get user input
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])

        # Fertilizer prediction
        fertilizer_features = np.array([[N, P, K]])

        fertilizer_prediction = rf_model.predict(
            fertilizer_features
        )[0]

        fertilizer_name = fertilizer_dict.get(
            fertilizer_prediction,
            "Unknown Fertilizer"
        )

        return render_template(
            'index.html',
            fertilizer_prediction=fertilizer_name
        )

    except Exception as e:
        return render_template(
            'index.html',
            error=f"Error: {str(e)}"
        )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
