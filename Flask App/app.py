from flask import Flask, request, jsonify, make_response, render_template
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
from huggingface_hub import hf_hub_download
from config import (
    API_KEY, RATE_LIMIT, RATE_LIMIT_WINDOW,
    CACHE_SIZE, CACHE_TTL,
    HF_TOKEN, HF_REPO_ID,
    PORT, DEBUG, LOG_LEVEL
)
from dotenv import load_dotenv
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from functools import wraps, lru_cache
import time
from datetime import datetime, timedelta
import logging
from logging.handlers import RotatingFileHandler
import hashlib

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Add file handler for logging
file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(getattr(logging, LOG_LEVEL))
logger.addHandler(file_handler)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Global variables for model and tokenizer
model = None
tokenizer = None

# Store rate limiting data
rate_limit_data = {}

# Initialize cache
prediction_cache = {}

def get_cache_key(text):
    """Generate a cache key for the input text"""
    return hashlib.md5(text.encode()).hexdigest()

def get_cached_prediction(text):
    """Get prediction from cache if available and not expired"""
    cache_key = get_cache_key(text)
    if cache_key in prediction_cache:
        cached_data = prediction_cache[cache_key]
        if time.time() - cached_data['timestamp'] < CACHE_TTL:
            logger.info(f"Cache hit for text: {text[:100]}...")
            return cached_data['prediction']
    return None

def cache_prediction(text, prediction):
    """Cache the prediction result"""
    cache_key = get_cache_key(text)
    prediction_cache[cache_key] = {
        'prediction': prediction,
        'timestamp': time.time()
    }
    
    # Remove oldest entries if cache is full
    if len(prediction_cache) > CACHE_SIZE:
        oldest_key = min(prediction_cache.keys(), 
                        key=lambda k: prediction_cache[k]['timestamp'])
        del prediction_cache[oldest_key]

def check_rate_limit(ip_address):
    """Check if the IP address has exceeded the rate limit"""
    current_time = time.time()
    if ip_address not in rate_limit_data:
        rate_limit_data[ip_address] = {
            'requests': [],
            'limit': RATE_LIMIT
        }
    
    # Clean old requests
    rate_limit_data[ip_address]['requests'] = [
        req_time for req_time in rate_limit_data[ip_address]['requests']
        if current_time - req_time < RATE_LIMIT_WINDOW
    ]
    
    # Check if rate limit exceeded
    if len(rate_limit_data[ip_address]['requests']) >= rate_limit_data[ip_address]['limit']:
        return False
    
    # Add new request
    rate_limit_data[ip_address]['requests'].append(current_time)
    return True

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

def create_response(data, status_code=200):
    """Helper function to create consistent JSON responses"""
    response = make_response(jsonify(data))
    response.headers['Content-Type'] = 'application/json'
    return response, status_code

def load_resources():
    global model, tokenizer
    try:
        
        # Download and load model from Hugging Face
        try:
            model_path = download_from_hf(HF_REPO_ID, 'model_hate_speech.h5')
            if model_path:
                # Create a custom InputLayer class that converts batch_shape to shape
                class CustomInputLayer(tf.keras.layers.InputLayer):
                    def __init__(self, **kwargs):
                        if 'batch_shape' in kwargs:
                            # Convert batch_shape to shape by removing the batch dimension
                            batch_shape = kwargs.pop('batch_shape')
                            kwargs['shape'] = batch_shape[1:]
                        super().__init__(**kwargs)
                
                # Load model with custom objects
                model = load_model(model_path, compile=False, 
                                 custom_objects={'InputLayer': CustomInputLayer})
            else:
                raise Exception("Failed to download model from Hugging Face")
        except Exception as hf_model_error:
            print(f"Error loading model from Hugging Face: {str(hf_model_error)}")
            raise
        
        # Download and load tokenizer from Hugging Face
        try:
            tokenizer_path = download_from_hf(HF_REPO_ID, 'tokenizer.pkl')
            if tokenizer_path:
                with open(tokenizer_path, 'rb') as f:
                    tokenizer = pickle.load(f)
            else:
                raise Exception("Failed to download tokenizer from Hugging Face")
        except Exception as hf_tokenizer_error:
            print(f"Error loading tokenizer from Hugging Face: {str(hf_tokenizer_error)}")
            raise
        
        if model is None or tokenizer is None:
            raise Exception("Model or tokenizer failed to load")
            
        return True
        
    except Exception as e:
        print(f"Critical error loading resources: {str(e)}")
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
            logger.error("Model or tokenizer not loaded")
            return None, None
            
        # Check cache first
        cached_result = get_cached_prediction(text)
        if cached_result:
            return cached_result
            
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
        
        result = (predicted_label, float(probability))
        
        # Cache the result
        cache_prediction(text, result)
        
        return result
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return None, None

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check rate limit
        if not check_rate_limit(request.remote_addr):
            return render_template('index.html', 
                                error='Too many requests. Please try again later.')
        
        text = request.form.get('text', '')
        if not text.strip():
            return render_template('index.html', error='Please enter some text to analyze')
        
        predicted_label, confidence = predict_hate_speech(text)
        
        if predicted_label is None:
            return render_template('index.html', error='Error during prediction. Please try again.')
        
        return render_template('index.html', 
                             prediction=predicted_label,
                             confidence=f"{confidence:.2%}",
                             text=text)
    
    return render_template('index.html')

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
        # Check rate limit
        if not check_rate_limit(request.remote_addr):
            return create_response({
                'status': 'error',
                'message': 'Rate limit exceeded. Please try again later.',
                'retry_after': RATE_LIMIT_WINDOW
            }, 429)

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
            logger.error("Model or tokenizer not loaded")
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
        
        # Log the request
        logger.info(f"Processing request for text: {text[:100]}...")
        
        # Make prediction
        predicted_label, confidence = predict_hate_speech(text)
        
        if predicted_label is None:
            logger.error("Prediction failed")
            return create_response({
                'status': 'error',
                'message': 'Error during prediction. Please check server logs.'
            }, 500)
        
        # Log the prediction
        logger.info(f"Prediction result: {predicted_label} (confidence: {confidence:.2%})")
        
        return create_response({
            'status': 'success',
            'data': {
                'prediction': predicted_label,
                'confidence': f"{confidence:.2%}",
                'input_text': text
            }
        })
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return create_response({
            'status': 'error',
            'message': 'An unexpected error occurred'
        }, 500)

# For local development
if __name__ == '__main__':
    # Check if resources are loaded
    if not load_resources():
        print("Warning: Resources not loaded properly. The app may not work correctly.")
    
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=True) 