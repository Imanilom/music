import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState('');
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      alert('Please upload a file first!');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setPrediction(response.data.genre);
      console.log(response.data)
  
      setError('');  // Clear any previous error
    } catch (error) {
      console.error('Error uploading file:', error);
      setError('Failed to upload file. Please try again.');
    }
  };

  return (
    <div className="App">
      <h1>Music Genre Prediction</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" accept=".mp3,.wav" onChange={handleFileChange} />
        <button type="submit">Upload and Predict</button>
      </form>
      {prediction && <div className="prediction">Predicted Genre: {prediction}</div>}
      {error && <div className="error">{error}</div>}
    </div>
  );
}

export default App;
