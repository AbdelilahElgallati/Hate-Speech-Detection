import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# PythonAnywhere specific settings
DEBUG = False
ALLOWED_HOSTS = ['yourusername.pythonanywhere.com']  # Replace with your PythonAnywhere domain

# Flask settings
SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Model and tokenizer paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model_hate_speech.h5')
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), 'tokenizer.pkl')

# Hugging Face settings
HF_TOKEN = os.getenv('HF_TOKEN')
HF_REPO_ID = os.getenv('HF_REPO_ID') 