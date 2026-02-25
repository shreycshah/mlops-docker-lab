from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import os

# Detect paths — works for both Dockerfile and Docker Compose
# Dockerfile:       main.py is at /app/main.py,     templates at /app/templates/
# Docker Compose:   main.py is at /app/src/main.py,  templates at /app/src/templates/
base_dir = os.path.dirname(os.path.abspath(__file__))
template_path = os.path.join(base_dir, 'templates')
static_path = os.path.join(base_dir, 'statics')

app = Flask(
    __name__,
    template_folder=template_path,
    static_folder=static_path,
    static_url_path='/static'
)

# Load the TensorFlow model
model = tf.keras.models.load_model('breast_cancer_model.keras')

# Load the scaler parameters saved during training
scaler_mean = np.load('scaler_mean.npy')
scaler_scale = np.load('scaler_scale.npy')

# Class labels: 0 = Malignant, 1 = Benign (sklearn convention)
class_labels = ['Malignant', 'Benign']


@app.route('/')
def home():
    return render_template('predict.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.form
            mean_radius = float(data['mean_radius'])
            mean_texture = float(data['mean_texture'])
            mean_smoothness = float(data['mean_smoothness'])
            mean_perimeter = float(data['mean_perimeter'])

            input_data = np.array(
                [mean_radius, mean_texture, mean_smoothness, mean_perimeter]
            )[np.newaxis, :]

            input_scaled = (input_data - scaler_mean) / scaler_scale

            prediction = model.predict(input_scaled)
            probability = float(prediction[0][0])

            predicted_index = 1 if probability > 0.5 else 0
            predicted_class = class_labels[predicted_index]
            confidence = probability if predicted_index == 1 else (1 - probability)

            return jsonify({
                "predicted_class": predicted_class,
                "confidence": round(confidence * 100, 2)
            })

        except Exception as e:
            return jsonify({"error": str(e)})

    elif request.method == 'GET':
        return render_template('predict.html')

    else:
        return "Unsupported HTTP method"


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)