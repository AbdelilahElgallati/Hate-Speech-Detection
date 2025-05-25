import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


# Load environment variables
load_dotenv()

# API Configuration
API_KEY = os.getenv('API_KEY', 'default_key')
RATE_LIMIT = int(os.getenv('RATE_LIMIT', 100))
RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW', 3600))

# Cache Configuration
CACHE_SIZE = int(os.getenv('CACHE_SIZE', 1000))
CACHE_TTL = int(os.getenv('CACHE_TTL', 3600))

# Hugging Face Configuration
HF_TOKEN = os.getenv('HF_TOKEN')
HF_REPO_ID = os.getenv('HF_REPO_ID', 'AbdelilahElgallati/Hate-model-detection')

# Server Configuration
PORT = int(os.getenv('PORT', 8000))
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
