# Hate Speech Detection API

A machine learning-based API for detecting hate speech and offensive language in text. This project uses TensorFlow and Flask to provide a RESTful API service that can classify text into three categories: hate speech, offensive language, or neither.

## Features

- Text classification into three categories:
  - Hate Speech
  - Offensive Language
  - Neither
- Confidence score for predictions
- RESTful API interface
- CORS support for cross-origin requests
- Health check endpoint for monitoring
- Comprehensive error handling

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Jupyter Notebook (for model training)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd hate-speech-detection
```

2. Create and activate a virtual environment (recommended):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Install Jupyter Notebook (for model training):
```bash
pip install jupyter
```

## Project Structure

```
hate-speech-detection/
├── app.py              # Main Flask application
├── model_hate_speech.h5 # Trained TensorFlow model
├── tokenizer.pkl       # Saved tokenizer for text preprocessing
├── requirements.txt    # Python dependencies
├── model_training.ipynb # Jupyter notebook for model training
└── README.md          # Project documentation
```

## Model Training

The model training process is documented in the `model_training.ipynb` Jupyter notebook. This notebook contains:

1. Data preprocessing steps
2. Model architecture definition
3. Training process
4. Model evaluation
5. Model saving instructions

To train the model:

1. Open the Jupyter notebook:
```bash
jupyter notebook model_training.ipynb
```

2. Follow the steps in the notebook to:
   - Load and preprocess the dataset
   - Train the model
   - Evaluate the model's performance
   - Save the model and tokenizer

The training process will generate:
- `model_hate_speech.h5`: The trained model file
- `tokenizer.pkl`: The tokenizer used for text preprocessing

These files are required for running the API.

## API Endpoints

### 1. Root Endpoint
- **URL**: `/`
- **Method**: `GET`
- **Description**: Returns API information and usage documentation
- **Response Example**:
```json
{
    "status": "success",
    "message": "Hate Speech Detection API",
    "endpoints": {
        "health": "/health",
        "predict": "/predict"
    },
    "usage": {
        "predict": {
            "method": "POST",
            "url": "/predict",
            "headers": {
                "Content-Type": "application/json"
            },
            "body": {
                "text": "Your text to analyze"
            }
        }
    }
}
```

### 2. Health Check
- **URL**: `/health`
- **Method**: `GET`
- **Description**: Checks if the service is running and if the model is loaded
- **Response Example**:
```json
{
    "status": "success",
    "message": "Service is running",
    "model_loaded": true,
    "tokenizer_loaded": true
}
```

### 3. Predict
- **URL**: `/predict`
- **Method**: `POST`
- **Headers**: 
  - `Content-Type: application/json`
- **Request Body**:
```json
{
    "text": "Your text to analyze"
}
```
- **Response Example**:
```json
{
    "status": "success",
    "data": {
        "prediction": "hate_speech",
        "confidence": "95.00%",
        "input_text": "Your text to analyze"
    }
}
```

## Usage Examples

### Using curl
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "your text to analyze"}'
```

### Using Python
```python
import requests

url = "http://localhost:5000/predict"
headers = {
    "Content-Type": "application/json"
}
data = {
    "text": "your text to analyze"
}

response = requests.post(url, json=data, headers=headers)
print(response.json())
```

### Using JavaScript
```javascript
fetch('http://localhost:5000/predict', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        text: 'your text to analyze'
    })
})
.then(response => response.json())
.then(data => console.log(data));
```

## Running the Application

1. Make sure you have all the required files:
   - `model_hate_speech.h5`
   - `tokenizer.pkl`

2. Start the Flask application:
```bash
python app.py
```

The server will start on `http://localhost:5000` by default.

## Error Handling

The API provides detailed error messages for various scenarios:

1. Invalid Content-Type:
```json
{
    "status": "error",
    "message": "Content-Type must be application/json",
    "example": {
        "headers": {
            "Content-Type": "application/json"
        },
        "body": {
            "text": "Your text to analyze"
        }
    }
}
```

2. Missing Text:
```json
{
    "status": "error",
    "message": "No text provided in request body",
    "example": {
        "text": "Your text to analyze"
    }
}
```

## Deployment

To deploy the application:

1. Set the `PORT` environment variable if you want to use a different port:
```bash
# Windows
set PORT=8080
python app.py

# Linux/Mac
export PORT=8080
python app.py
```

2. For production deployment, it's recommended to use a WSGI server like Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow for the machine learning framework
- Flask for the web framework
- The dataset used for training the model
