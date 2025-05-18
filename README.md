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
git clone https://github.com/AbdelilahElgallati/Hate-Speech-Detection/
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

The model training process is documented in the `HateSpeechModel.ipynb` Jupyter notebook. This notebook contains:

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


## Acknowledgments

- TensorFlow for the machine learning framework
- Flask for the web framework
- The dataset used for training the model
