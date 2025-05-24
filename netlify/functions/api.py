import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from huggingface_hub import hf_hub_download
from config import HF_TOKEN, HF_REPO_ID

app = Flask(__name__)
CORS(app)

model = None
tokenizer = None

def download_from_hf(repo_id, filename):
    """Download a file from Hugging Face Hub"""
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=HF_TOKEN
        )
        return file_path
    except Exception as e:
        print(f"Error downloading from Hugging Face: {str(e)}")
        return None

def load_resources():
    global model, tokenizer
    try:
        model_path = download_from_hf(HF_REPO_ID, 'model_hate_speech.h5')
        tokenizer_path = download_from_hf(HF_REPO_ID, 'tokenizer.pkl')
        
        if model_path is None or tokenizer_path is None:
            print("Failed to download model or tokenizer")
            return False
        
        model = load_model(model_path, compile=False)
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        
        print("Model and tokenizer loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading resources: {str(e)}")
        return False

def predict_hate_speech(text):
    try:
        if model is None or tokenizer is None:
            print("Model or tokenizer not loaded")
            return None, None
            
        # Preprocess the text
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=100, padding='pre')
        
        # Make prediction
        prediction = model.predict(padded, verbose=0)
        
        # Get the predicted class and probability
        predicted_class = np.argmax(prediction, axis=1)
        probability = np.max(prediction)
        
        # Map class index to label
        class_labels = ['hate_speech', 'offensive_language', 'neither']
        predicted_label = class_labels[predicted_class[0]]
        
        return predicted_label, float(probability)
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None, None

@app.route('/api/health', methods=['GET'])  # Updated to /api/health
def health_check():
    try:
        if model is None or tokenizer is None:
            return jsonify({
                'status': 'error',
                'message': 'Model or tokenizer not loaded'
            }), 503
        return jsonify({
            'status': 'success',
            'message': 'Service is running',
            'model_loaded': model is not None,
            'tokenizer_loaded': tokenizer is not None
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if not request.is_json:
            return jsonify({
                'status': 'error',
                'message': 'Content-Type must be application/json'
            }), 415

        if model is None or tokenizer is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not properly loaded'
            }), 503
        
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No text provided in request body'
            }), 400
        
        text = data['text']
        if not text.strip():
            return jsonify({
                'status': 'error',
                'message': 'Empty text provided'
            }), 400
        
        predicted_label, confidence = predict_hate_speech(text)
        
        if predicted_label is None:
            return jsonify({
                'status': 'error',
                'message': 'Error during prediction'
            }), 500
            
        return jsonify({
            'status': 'success',
            'data': {
                'prediction': predicted_label,
                'confidence': f"{confidence:.2%}",
                'input_text': text
            }
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Initialize resources when the function is loaded
load_resources()

def handler(event, context):
    """Netlify serverless function handler"""
    from serverless_wsgi import handle_request
    return handle_request(app, event, context)
