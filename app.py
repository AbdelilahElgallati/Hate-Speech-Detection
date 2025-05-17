from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

app = Flask(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_resources():
    global model, tokenizer
    try:
        # Load the trained model
        model = load_model('model_hate_speech.h5')
        print("Model loaded successfully")
        
        # Load the tokenizer
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
        print("Tokenizer loaded successfully")
        
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
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model and tokenizer are loaded
        if model is None or tokenizer is None:
            return jsonify({'error': 'Model not properly loaded. Please check server logs.'})
        
        text = request.json.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'})
        
        predicted_label, confidence = predict_hate_speech(text)
        
        if predicted_label is None:
            return jsonify({'error': 'Error during prediction. Please check server logs.'})
        
        return jsonify({
            'prediction': predicted_label,
            'confidence': f"{confidence:.2%}"
        })
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Make sure the templates directory exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Check if resources are loaded
    if not load_resources():
        print("Warning: Resources not loaded properly. The app may not work correctly.")
    
    # Run the app
    app.run(debug=True, port=5000) 