import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_KEY = os.getenv('API_KEY', 'your-api-key-here')

# Rate Limiting
RATE_LIMIT = int(os.getenv('RATE_LIMIT', 100))  # Number of requests allowed
RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW', 3600))  # Time window in seconds

# Cache Configuration
CACHE_SIZE = int(os.getenv('CACHE_SIZE', 1000))  # Maximum number of cached predictions
CACHE_TTL = int(os.getenv('CACHE_TTL', 3600))  # Cache time-to-live in seconds

# Hugging Face Configuration
HF_TOKEN = os.getenv('HF_TOKEN', 'your-hf-token-here')
HF_REPO_ID = os.getenv('HF_REPO_ID', 'your-repo-id-here')

# Server Configuration
PORT = int(os.getenv('PORT', 5000))
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO') 