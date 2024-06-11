from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import pandas as pd
import numpy as np
import pickle
import os
import tempfile

# Path to the trained model
model_path = 'random_forest_model.pkl'
genre_labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Load the trained model
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    features = {
        'Chroma': np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        'RMSE': np.mean(librosa.feature.rms(y=y)),
        'Spectral_Centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'Spectral_Bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        'RollOff': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        'Zero_Crossing_Rate': np.mean(librosa.feature.zero_crossing_rate(y)),
        'Harmony': np.mean(librosa.effects.harmonic(y)),
        'Percussive': np.mean(librosa.effects.percussive(y)),
        'Tempo': librosa.beat.tempo(y=y, sr=sr)[0]
    }
    for i, mfcc in enumerate(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)):
        features[f'MFCC_{i+1}'] = np.mean(mfcc)
    return pd.DataFrame([features])

@app.route('/')
def home():
    return "Welcome to the Music Genre Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if a file is in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save the file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name

        # Extract features from the audio file
        features = extract_features(temp_file_path)
        
        # Make prediction
        prediction = model.predict(features)
        
        # Clean up the temporary file
        os.remove(temp_file_path)
        
        # Create response
        response = {
            'prediction': prediction[0]
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
