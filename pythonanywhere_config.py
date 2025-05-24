import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# PythonAnywhere specific settings
DEBUG = False
ALLOWED_HOSTS = ['yourusername.pythonanywhere.com']  # Replace with your PythonAnywhere domain

# Flask settings
SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Model and tokenizer paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_hate_speech.h5')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'tokenizer.pkl')

# Hugging Face settings
HF_TOKEN = os.getenv('HF_TOKEN')
HF_REPO_ID = os.getenv('HF_REPO_ID')

# Log configuration
logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Model path: {MODEL_PATH}")
logger.info(f"Tokenizer path: {TOKENIZER_PATH}")
logger.info(f"HF_TOKEN present: {bool(HF_TOKEN)}")
logger.info(f"HF_REPO_ID present: {bool(HF_REPO_ID)}") 