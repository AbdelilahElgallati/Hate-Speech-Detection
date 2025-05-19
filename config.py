import os
from dotenv import load_dotenv

# Debug prints before loading .env
print("Before loading .env:")
print("HF_TOKEN in env:", os.getenv('HF_TOKEN'))
print("HF_REPO_ID in env:", os.getenv('HF_REPO_ID'))

# Load environment variables from .env file
load_dotenv()

# Debug prints after loading .env
print("\nAfter loading .env:")
print("HF_TOKEN in env:", os.getenv('HF_TOKEN'))
print("HF_REPO_ID in env:", os.getenv('HF_REPO_ID'))

# Hugging Face Configuration
HF_TOKEN = os.getenv('HF_TOKEN')
HF_REPO_ID = os.getenv('HF_REPO_ID', 'AbdelilahElgallati/Hate-detection-model')
