import sys
import os

# Print debug information
print("Python version:", sys.version)
print("Python path before modifications:", sys.path)

# Add virtual environment path
venv_path = '/home/AbdelilahElgallati/.virtualenvs/hate-speech-env/lib/python3.9/site-packages'
if venv_path not in sys.path:
    sys.path.insert(0, venv_path)

# Add your project directory to the Python path
path = '/home/AbdelilahElgallati/Hate-Speech-Detection'
if path not in sys.path:
    sys.path.insert(0, path)

# Print modified Python path
print("Python path after modifications:", sys.path)

# Set environment variables
os.environ['PYTHONPATH'] = path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Try importing Flask to check if it's available
try:
    import flask
    print("Flask version:", flask.__version__)
except ImportError as e:
    print("Error importing Flask:", str(e))

# Import your Flask app
from app import app as application

# Enable error logging
import logging
logging.basicConfig(stream=sys.stderr) 