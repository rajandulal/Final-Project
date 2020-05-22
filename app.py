from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from util import base64_to_pil

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, request, render_template, Response, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import  WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/Pneumonia using VGG19.h5'

#Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary to make everything ready to run on the GPU ahead of time
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img, model):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)

    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        # Get the image from post request
        img = base64_to_pil(request.json)

        img.save("uploads\image.jpg")

        img_path = os.path.join(os.path.dirname(__file__), 'uploads\image.jpg')

        os.path.isfile(img_path)

        img = image.load_img(img_path, target_size=(150, 150))

        preds = model_predict(img, model)

        result = preds[0, 0]

        print(result)

        if result > 0.5:
            return jsonify(result="PNEUMONIA")
        else:
            return jsonify(result="NORMAL")

    return None


if __name__ == '__main__':
    app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
