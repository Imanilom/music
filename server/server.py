import librosa
import numpy as np
from keras.models import load_model
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Load model, scaler, and label encoder
model = load_model('modelmusicgenre.h5')  # Ganti dengan path model
df =pd.read_csv('./dataset/datamusic1.csv')
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
scaler = StandardScaler().fit(X)  # Gunakan scaler yang sama dengan saat pelatihan
labelencoder = LabelEncoder().fit(y)  # Gunakan label encoder yang sama dengan saat pelatihan

def extract_features(y, sr):

    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    sc= librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidths = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rates = librosa.feature.zero_crossing_rate(y=y)
    tempo,_ = librosa.beat.beat_track(y=y, sr=sr)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    features = {
    'Chroma': chroma.mean(),
    'Spectral Centroid': sc.mean(),
    'Spectral Bandwidth' :spectral_bandwidths.mean(),
    'Spectral Rolloff' :spectral_rolloff.mean(),
    'Zero-Crossing Rate' :zero_crossing_rates.mean(),
    'Tempo' : tempo.mean()
    }
    # Menghitung rata-rata untuk setiap nilai pada mel spectrogram
    for i in range(1, 129): # Mel spectrogram memiliki 128 band
        features[f'Mel_Spectrogram_{i}'] = spectrogram[i-1].mean()
    # Menghitung rata-rata untuk setiap koefisien MFCC
    for i in range(1, 14): # 13 koefisien MFCC
        features[f'MFCC_{i}'] = mfccs[i-1].mean()

    return features

@app.route('/predict', methods=['POST'])
def predict_genre():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Simpan file sementara (misalnya, di folder 'uploads')
        file_path = "uploads/" + file.filename
        file.save(file_path)

        # Muat audio dan ekstrak fitur
        y, sr = librosa.load(file_path, sr=None)
        features = extract_features(y, sr)

        # Prediksi genre
        features_scaled = scaler.transform(pd.DataFrame([features]))
        prediction = model.predict(features_scaled)
        genre = labelencoder.inverse_transform([np.argmax(prediction)])

        return jsonify({'genre': genre[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

