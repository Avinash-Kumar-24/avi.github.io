from flask import Flask, jsonify, request, render_template
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load your .h5 model
model = load_model('"C:\Users\Akash-Avinash\Desktop\LeafDetector.h5"')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Assume your model expects input data in the form of a JSON object
    data = request.get_json()

    # Perform any necessary preprocessing on the data
    # ...

    # Make predictions using your model
    predictions = model.predict(np.array(data['input']))

    # Return the predictions as JSON
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
