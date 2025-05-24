import sys
import os

# Add virtual environment path
venv_path = '/home/AbdelilahElgallati/.virtualenvs/hate-speech-env/lib/python3.9/site-packages'
if venv_path not in sys.path:
    sys.path.insert(0, venv_path)

# Add your project directory to the Python path
path = '/home/AbdelilahElgallati/Hate-Speech-Detection'  # Replace with your PythonAnywhere username
if path not in sys.path:
    sys.path.insert(0, path)

# Set environment variables
os.environ['PYTHONPATH'] = path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Import your Flask app
from app import app as application

# Enable error logging
import logging
logging.basicConfig(stream=sys.stderr) 