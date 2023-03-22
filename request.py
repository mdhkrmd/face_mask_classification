import requests

url = "http://localhost:5000/klasifikasi"
files = {'file': open('D:\Synapsis Test\FMD_DATASET_FIX/test\incorrect_mask\mmc/109.jpg', 'rb')}
response = requests.post(url, files=files)

if response.status_code == 200:
    hasil_prediksi = response.json()['prediksi']
    gambar_prediksi = response.json()['gambar_prediksi']
    print(f"Prediksi: {hasil_prediksi}")
    print(f"Gambar prediksi: {gambar_prediksi}")
else:
    print("Terjadi kesalahan saat melakukan request.")