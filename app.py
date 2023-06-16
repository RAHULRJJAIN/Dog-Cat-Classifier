# Importing Dependencies
import io
import numpy as np
from model import model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

from flask import Flask, render_template, request, send_from_directory

# Flask App
app = Flask(__name__)

# Default Page
@app.route('/')
def main():
    return render_template('index.html')

# Prediction Page
@app.route('/home', methods=['POST'])
def home():
    # Getting Image from HTML Form
    image_file = request.files['image']
    # Read image file as bytes
    image_bytes = image_file.read()
    # Open image using Pillow
    image = Image.open(io.BytesIO(image_bytes))
    # Resize the image to 128x128 pixels
    image = image.resize((128, 128))
    # Convert the image to array
    image = img_to_array(image)
    # Normalize the image
    image = image / 255.0
    # Reshape the image to (1, 128, 128, 3)
    image = image.reshape(1, 128, 128, 3)

    # Perform prediction using the model
    prediction = model.predict(image)

    if prediction[0, 0] > prediction[0, 1]:
        result = 'Cat'
    else:
        result = 'Dog'

    return result

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)