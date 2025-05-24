import os

# Get the port from the environment variable, default to 8000 if not set
port = int(os.environ.get("PORT", 8000))

# Gunicorn configuration
bind = f"0.0.0.0:{port}"
workers = 1  # Single worker for CPU-bound tasks
threads = 2  # Multiple threads per worker
timeout = 120  # Increase timeout for model loading 