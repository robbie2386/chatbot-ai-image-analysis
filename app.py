
from flask import Flask, request, jsonify
import cv2
import numpy as np
import pytesseract
import requests
from PIL import Image
import io
import os

app = Flask(__name__)

# Fungsi untuk membaca teks dari gambar (OCR)
def extract_text_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip()

# Fungsi untuk mendeteksi wajah dalam gambar
def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces)

# Fungsi untuk mengecek apakah gambar terlalu buram
def is_blurry(image, threshold=100):
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var < threshold

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    image_stream = io.BytesIO(file.read())
    image = np.array(Image.open(image_stream))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    detected_text = extract_text_from_image(image)
    num_faces = detect_faces(image)
    blurry = is_blurry(image)
    
    result = {
        "Teks dalam gambar": detected_text if detected_text else "Tidak ada teks terdeteksi.",
        "Jumlah wajah terdeteksi": num_faces,
        "Foto buram": "Ya" if blurry else "Tidak"
    }
    
    return jsonify(result)

import os

port = os.getenv("PORT")

# Jika variabel PORT tidak ditemukan atau tidak valid, gunakan default 5000
if port is None or not port.isdigit():
    print("⚠️ WARNING: PORT tidak ditemukan atau tidak valid. Menggunakan port default 5000.")
    port = "5000"

port = int(port)  # Konversi ke integer

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=port, debug=True)

