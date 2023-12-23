from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import imghdr 

app = Flask(__name__)

model = load_model('models/my_model_5.h5')

def is_mri_image(file_path):
    image_type = imghdr.what(file_path)
    mri_image_types = ["png"]

    return image_type in mri_image_types

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    if not is_mri_image(image_path):
        return render_template('index.html', error="Not an MRI image")

    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(128, 128), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 

    # Make predictions
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Map class indices to labels
    class_labels = {0: "Moderate Demented", 1: "Non Demented", 2: "Very Mild Demented", 3: "Mild Demented"}
    predicted_label = class_labels.get(predicted_class, "Unknown")

    print("Predicted Class (Flask App):", predicted_class)

    return render_template('index.html', prediction=predicted_label, image_path="./images/" + imagefile.filename)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
