# =[Modules dan Packages]========================

from flask import Flask,render_template,request,jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
from PIL import Image
from io import BytesIO
import base64
from fungsi import make_model

# =[Variabel Global]=============================

app = Flask(__name__, static_url_path='/static')

app.config['MAX_CONTENT_LENGTH'] = 2048 * 2048
app.config['UPLOAD_EXTENSIONS']  = ['.jpg','.JPG', '.png', '.PNG', '.jpeg', '.JPEG']
app.config['UPLOAD_PATH']        = './static/images/uploads/'

model = None

NUM_CLASSES = 3
classPred = ['incorrect_mask', 'with_mask', 'without_mask']

# =[Routing]=====================================

# [Routing untuk Halaman Utama atau Home]
@app.route("/")
def beranda():
	return render_template('index.html')

# [Routing untuk API]	
@app.route("/klasifikasi", methods=['POST'])
def apiDeteksi():
	hasil_prediksi  = '(none)'
	gambar_prediksi = '(none)'

	# Get File
	uploaded_file = request.files['file']
	if uploaded_file == "":
		respon = jsonify({
			"message": "Upload File"
		})
		respon.status_code=400
		return respon
	filename      = secure_filename(uploaded_file.filename)
	
	# Periksa
	if filename != '':
	
		# Set/mendapatkan extension dan path dari file yg diupload
		file_ext        = os.path.splitext(filename)[1]
		gambar_prediksi = '/static/images/uploads/' + filename
		
		# extension file sesuai (jpg)
		if file_ext in app.config['UPLOAD_EXTENSIONS']:
			
			# Simpan Gambar
			uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
			
			# Memuat Gambar
			# img = uploaded_file.read()
			# test_image = Image.open(BytesIO(base64.b64decode(img.split(',')[1])))
			test_image         = Image.open('.' + gambar_prediksi)
			
			# Mengubah Ukuran Gambar
			test_image_resized = test_image.resize((150, 150))
			
			# Konversi
			image_array        = np.array(test_image_resized)
			test_image_x       = (image_array / 255) - 0.5
			# test_image_x       = np.expand_dims(test_image_x, axis=0)
			test_image_x       = np.array([image_array])
			
			# Prediksi Gambar
			y_pred_test_single         = model.predict_proba(test_image_x)
			y_pred_test_classes_single = np.argmax(y_pred_test_single, axis=1)
			
			hasil_prediksi = classPred[y_pred_test_classes_single[0]]
			
			# Return  JSON
			respon = jsonify({
				"prediksi": hasil_prediksi,
				"gambar_prediksi" : gambar_prediksi
			})
			respon.status_code=200
			print(respon.json)
			return respon
		else:
			respon = jsonify({
				'message': 'Error'
			})
			respon.status_code=400
			return respon
# =[Main]========================================		

if __name__ == '__main__':	
	
	# Load model
	model = make_model()
	model.load_weights("no-tl-3.h5")

	# Run Flask 
	app.run(host="localhost", port=5000, debug=True)