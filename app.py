
from flask import Flask, request, jsonify, render_template
import tensorflow as tf

from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("C:\\Users\\Akash-Avinash\\Desktop\\LeafDetector.h5")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data.get('image', '')

    # Preprocess the image
    img = image.load_img(image_data, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    predictions = model.predict(img_array)

    # Interpret the prediction
    prediction_result = 'Diseased Leaf' if predictions[0, 0] > 0.5 else 'Healthy Leaf'

    return jsonify({'prediction': prediction_result})

if __name__ == '__main__':
    app.run(debug=True)
