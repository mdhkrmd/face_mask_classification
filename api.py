# =[Modules dan Packages]========================

from flask import Flask,render_template,request,jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
from PIL import Image
import io
import base64
from fungsi import make_model

# =[Variabel Global]=============================
app = Flask(__name__)

model = None
NUM_CLASSES = 3
classPred = ['incorrect_mask', 'with_mask', 'without_mask']

@app.route('/klasifikasi', methods=['POST'])
def predict():
    ambil = request.get_json()
    image_path = ambil['image_path']
    
    if not image_path:
        response = {'message': 'path salah, gambar tidak ditemukan'}
        return jsonify(response)

    # dec = base64.b64decode(image_path)
    # img = Image.open(io.BytesIO(dec))
    img = Image.open(image_path)
    test_image_resized = img.resize((150, 150))
    img_array = np.array(test_image_resized) / 255.0
    img_test = np.expand_dims(img_array, axis=0)

    # predict = model.predict(img)
    predict = model.predict(img_test)
    y_pred_test_classes_single = np.argmax(predict, axis=1)
    hasil_prediksi = classPred[y_pred_test_classes_single[0]]

    response = {'label': str(hasil_prediksi)}
    return jsonify(response)


if __name__ == '__main__':
    model = make_model()
    model.load_weights("no-tl-3.h5")

	# Run Flask 
    app.run(host="localhost", port=5000, debug=True)