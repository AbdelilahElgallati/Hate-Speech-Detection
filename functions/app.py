from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import json
import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the model
model = tf.keras.models.load_model('model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        # Preprocess the text (you'll need to implement your preprocessing logic)
        # This is a placeholder - replace with your actual preprocessing
        processed_text = text.lower()
        
        # Make prediction
        prediction = model.predict([processed_text])
        
        # Convert prediction to list for JSON serialization
        prediction_list = prediction.tolist()
        
        return jsonify({
            'success': True,
            'prediction': prediction_list
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# This is the serverless function handler
def handler(event, context):
    # Parse the event body
    body = json.loads(event['body']) if event.get('body') else {}
    
    # Create a mock request object
    class MockRequest:
        def __init__(self, json_data):
            self.json_data = json_data
        
        def get_json(self):
            return self.json_data
    
    # Create a mock request
    request = MockRequest(body)
    
    # Call the predict function
    response = predict()
    
    # Return the response in Netlify Functions format
    return {
        'statusCode': 200,
        'body': json.dumps(response.get_json())
    } 