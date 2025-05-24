import json
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
from huggingface_hub import hf_hub_download
from config import HF_TOKEN, HF_REPO_ID

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

def load_resources():
    global model, tokenizer
    try:
        model = load_model(download_from_hf(HF_REPO_ID, 'model_hate_speech.h5'), compile=False)
        print("Model loaded successfully")
        
        tokenizer_path = download_from_hf(HF_REPO_ID, 'tokenizer.pkl')
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        print("Tokenizer loaded successfully")
        
        return True
    except Exception as e:
        print(f"Error loading resources: {str(e)}")
        return False

def preprocess_text(text):
    try:
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, padding='pre', maxlen=20)
        return padded
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return None

def predict_hate_speech(text):
    try:
        if model is None or tokenizer is None:
            print("Model or tokenizer not loaded")
            return None, None
            
        processed_text = preprocess_text(text)
        if processed_text is None:
            return None, None
        
        prediction = model.predict(processed_text, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)
        probability = np.max(prediction)
        
        class_labels = ['hate_speech', 'offensive_language', 'neither']
        predicted_label = class_labels[predicted_class[0]]
        
        return predicted_label, float(probability)
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None, None

def handler(event, context):
    # Load resources if not already loaded
    if model is None or tokenizer is None:
        if not load_resources():
            return {
                'statusCode': 503,
                'body': json.dumps({
                    'status': 'error',
                    'message': 'Failed to load model resources'
                })
            }

    # Get the path from the event
    path = event.get('path', '')
    # Remove /api prefix if present
    if path.startswith('/api'):
        path = path[4:]

    if path == '/' or path == '':
        return {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'success',
                'message': 'Service is running',
            })
        }
    
    # Handle different endpoints
    if path == '/health' or path == 'health':
        return {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'success',
                'message': 'Service is running',
                'model_loaded': model is not None,
                'tokenizer_loaded': tokenizer is not None
            })
        }
    
    elif path == '/predict' or path == 'predict':
        if event.get('httpMethod') != 'POST':
            return {
                'statusCode': 405,
                'body': json.dumps({
                    'status': 'error',
                    'message': 'Method not allowed'
                })
            }
        
        try:
            body = json.loads(event.get('body', '{}'))
            text = body.get('text', '')
            
            if not text.strip():
                return {
                    'statusCode': 400,
                    'body': json.dumps({
                        'status': 'error',
                        'message': 'Empty text provided'
                    })
                }
            
            predicted_label, confidence = predict_hate_speech(text)
            
            if predicted_label is None:
                return {
                    'statusCode': 500,
                    'body': json.dumps({
                        'status': 'error',
                        'message': 'Error during prediction'
                    })
                }
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'status': 'success',
                    'data': {
                        'prediction': predicted_label,
                        'confidence': f"{confidence:.2%}",
                        'input_text': text
                    }
                })
            }
        except Exception as e:
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'status': 'error',
                    'message': str(e)
                })
            }
    
    else:
        return {
            'statusCode': 404,
            'body': json.dumps({
                'status': 'error',
                'message': 'Endpoint not found'
            })
        }
