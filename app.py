from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
from huggingface_hub import hf_hub_download
from config import HF_TOKEN, HF_REPO_ID

# Debug prints for environment
print("Current working directory:", os.getcwd())
print("Environment variables:")
print("HF_TOKEN:", HF_TOKEN)
print("HF_REPO_ID:", HF_REPO_ID)
print("Files in current directory:", os.listdir('.'))

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Global variables for model and tokenizer
model = None
tokenizer = None

def download_from_hf(repo_id, filename):
    """Download a file from Hugging Face Hub"""
    try:
        print(f"Attempting to download {filename} from {repo_id}")
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=HF_TOKEN
        )
        print(f"Successfully downloaded {filename} to {file_path}")
        return file_path
    except Exception as e:
        print(f"Error downloading from Hugging Face: {str(e)}")
        return None

def create_response(data, status_code=200):
    """Helper function to create consistent JSON responses"""
    response = make_response(jsonify(data))
    response.headers['Content-Type'] = 'application/json'
    return response, status_code

def load_resources():
    global model, tokenizer
    try:
        # Use local model and tokenizer files
        model_path = 'model_hate_speech.h5'
        tokenizer_path = 'tokenizer.pkl'
        
        print(f"Loading model from: {model_path}")
        print(f"Loading tokenizer from: {tokenizer_path}")
        
        try:
            # Try loading with different approaches
            # First attempt: Load with custom_objects
            # model = load_model(model_path, compile=False, custom_objects={'tf': tf})
            model = load_model(download_from_hf(HF_REPO_ID, 'model_hate_speech.h5'), compile=False)
            print("Model loaded successfully")
        except Exception as model_error:
            print(f"Error loading model: {str(model_error)}")
            print(f"TensorFlow version: {tf.__version__}")
            raise
        
        try:
            # Load the tokenizer
            tokenizer_path = download_from_hf(HF_REPO_ID, 'tokenizer.pkl')
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            print("Tokenizer loaded successfully")
        except Exception as tokenizer_error:
            print(f"Error loading tokenizer: {str(tokenizer_error)}")
            raise
        
        return True
    except Exception as e:
        print(f"Error loading resources: {str(e)}")
        return False

# Load resources when starting the app
load_resources()

# Define the preprocessing function
def preprocess_text(text):
    try:
        # Convert text to sequence
        sequence = tokenizer.texts_to_sequences([text])
        # Pad sequence
        padded = pad_sequences(sequence, padding='pre', maxlen=20)
        return padded
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return None

# Define the prediction function
def predict_hate_speech(text):
    try:
        if model is None or tokenizer is None:
            print("Model or tokenizer not loaded")
            return None, None
            
        # Preprocess the text
        processed_text = preprocess_text(text)
        if processed_text is None:
            return None, None
        
        # Make prediction
        prediction = model.predict(processed_text, verbose=0)
        
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

@app.route('/')
def home():
    return create_response({
        'status': 'success',
        'message': 'Hate Speech Detection API',
        'endpoints': {
            'health': '/health',
            'predict': '/predict'
        },
        'usage': {
            'predict': {
                'method': 'POST',
                'url': '/predict',
                'headers': {
                    'Content-Type': 'application/json'
                },
                'body': {
                    'text': 'Your text to analyze'
                },
                'example': {
                    'curl': 'curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d \'{"text": "your text to analyze"}\''
                }
            }
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for API monitoring"""
    try:
        if model is None or tokenizer is None:
            return create_response({
                'status': 'error',
                'message': 'Model or tokenizer not loaded'
            }, 503)
        return create_response({
            'status': 'success',
            'message': 'Service is running',
            'model_loaded': model is not None,
            'tokenizer_loaded': tokenizer is not None
        })
    except Exception as e:
        return create_response({
            'status': 'error',
            'message': str(e)
        }, 500)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check content type
        if not request.is_json:
            return create_response({
                'status': 'error',
                'message': 'Content-Type must be application/json',
                'example': {
                    'headers': {
                        'Content-Type': 'application/json'
                    },
                    'body': {
                        'text': 'Your text to analyze'
                    }
                }
            }, 415)

        # Check if model and tokenizer are loaded
        if model is None or tokenizer is None:
            return create_response({
                'status': 'error',
                'message': 'Model not properly loaded. Please check server logs.'
            }, 503)
        
        # Get and validate input
        data = request.get_json()
        if not data or 'text' not in data:
            return create_response({
                'status': 'error',
                'message': 'No text provided in request body',
                'example': {
                    'text': 'Your text to analyze'
                }
            }, 400)
        
        text = data['text']
        if not text.strip():
            return create_response({
                'status': 'error',
                'message': 'Empty text provided'
            }, 400)
        
        # Make prediction
        predicted_label, confidence = predict_hate_speech(text)
        
        print(predicted_label)
        print(confidence)
        
        if predicted_label is None:
            return create_response({
                'status': 'error',
                'message': 'Error during prediction. Please check server logs.'
            }, 500)
        
        return create_response({
            'status': 'success',
            'data': {
                'prediction': predicted_label,
                'confidence': f"{confidence:.2%}",
                'input_text': text
            }
        })
    except Exception as e:
        return create_response({
            'status': 'error',
            'message': str(e)
        }, 500)

if __name__ == '__main__':
    # Check if resources are loaded
    if not load_resources():
        print("Warning: Resources not loaded properly. The app may not work correctly.")
    
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=True) 