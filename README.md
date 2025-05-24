# Hate Speech Detection Web App

A web application that uses machine learning to detect hate speech and offensive language in text.

## Features

- Real-time text analysis
- Three classification categories: Hate Speech, Offensive Language, and Neither
- Confidence score for predictions
- Modern and responsive UI
- REST API endpoints

## Deployment to Vercel

1. Install the Vercel CLI:
```bash
npm install -g vercel
```

2. Login to Vercel:
```bash
vercel login
```

3. Deploy the application:
```bash
vercel
```

4. For production deployment:
```bash
vercel --prod
```

## Environment Variables

Create a `.env` file in the root directory with the following variables:
```
HF_TOKEN=your_huggingface_token
HF_REPO_ID=your_huggingface_repo_id
```

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## API Endpoints

- `GET /`: Web interface
- `POST /`: Submit text for analysis
- `GET /health`: Health check endpoint
- `POST /predict`: API endpoint for predictions

## Technologies Used

- Flask
- TensorFlow
- Hugging Face Hub
- Tailwind CSS
- Vercel
