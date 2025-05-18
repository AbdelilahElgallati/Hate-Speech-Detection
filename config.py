import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Hugging Face Configuration
HF_TOKEN = os.getenv('HF_TOKEN')
HF_REPO_ID = os.getenv('HF_REPO_ID', 'your-username/hate-speech-model')

# AWS Configuration
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME') 