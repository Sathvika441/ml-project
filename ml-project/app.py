from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models
def load_models():
    try:
        rf_model_path = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")

        with open(nb_model_path, "rb") as f:
            nb_model = pickle.load(f)

        with open(rf_model_path, "rb") as f:
            rf_model = pickle.load(f)

        return nb_model, rf_model

    except Exception as e:
        print(f"Error loading models: {e}")
        raise


# Crop mapping
crop_dict = {
    1: 'rice',
    2: 'maize',
    3: 'jute',
    4: 'cotton',
    5: 'coconut',
    6: 'papaya',
    7: 'orange',
    8: 'apple',
    9: 'muskmelon',
    10: 'watermelon',
    11: 'grapes',
    12: 'mango',
    13: 'banana',
    14: 'pomegranate',
    15: 'lentil',
    16: 'blackgram',
    17: 'mungbean',
    18: 'mothbeans',
    19: 'pigeonpeas',
    20: 'kidneybeans',
    21: 'chickpea',
    22: 'coffee'
}

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

# Load models once at startup
nb_model, rf_model = load_models()


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
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Crop prediction
        crop_features = np.array(
            [[N, P, K, temperature, humidity, ph, rainfall]]
        )

        crop_prediction_num = nb_model.predict(crop_features)[0]
        crop_name = crop_dict.get(crop_prediction_num, "Unknown Crop")

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
            crop_prediction=crop_name,
            fertilizer_prediction=fertilizer_name
        )

    except Exception as e:
        return render_template(
            'index.html',
            error=f"Error: {str(e)}"
        )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
