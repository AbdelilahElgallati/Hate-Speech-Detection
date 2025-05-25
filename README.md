# Hate Speech Detection API

A Flask-based API for detecting hate speech in text using machine learning.

## Project Structure

```
hate-speech-detection/
├── src/
│   ├── app/
│   │   ├── routes/
│   │   │   └── main.py
│   │   ├── services/
│   │   │   └── hate_speech_detector.py
│   │   ├── models/
│   │   ├── utils/
│   │   ├── templates/
│   │   └── static/
│   │       ├── css/
│   │       ├── js/
│   │       └── images/
│   ├── config/
│   │   └── config.py
│   ├── tests/
│   └── wsgi.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup Instructions

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   export FLASK_ENV=development  # On Windows: set FLASK_ENV=development
   ```

4. Run the application:
   ```bash
   python src/wsgi.py
   ```

## API Endpoints

- `POST /predict`: Predict if text contains hate speech
  - Request body: `{"text": "your text here"}`
  - Response: `{"prediction": 0/1, "probability": float, "processed_text": string}`

- `GET /health`: Health check endpoint
  - Response: `{"status": "healthy"}`

## Development

- Run tests: `pytest`
- Format code: `black src/`
- Lint code: `flake8 src/`

## Deployment

The application can be deployed using Gunicorn:

```bash
gunicorn --config gunicorn_config.py src.wsgi:app
```

## License

MIT
