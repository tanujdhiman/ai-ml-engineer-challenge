from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
import logging
from prometheus_fastapi_instrumentator import Instrumentator
import os
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import re
import emoji
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import uvicorn
import nltk

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model
MODEL_PATH = os.getenv("MODEL_PATH", "models/mbti_model.joblib")

# Define the TextPreprocessor class
class TextPreprocessor:
    """Text preprocessing utility class"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('punkt_tab')
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the input text with multiple cleaning steps.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Replace emojis with text
        text = emoji.demojize(text)
        
        # Expand contractions
        text = contractions.fix(text)
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(tokens)

# Load the model
try:
    logger.info(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise RuntimeError("Failed to load model")

# Initialize the preprocessor
preprocessor = TextPreprocessor()

# Define request/response models
class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    mbti_type: str
    confidence: float

# Initialize FastAPI app
app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict MBTI personality type from input text.
    """
    try:
        # Preprocess the input text
        processed_text = preprocessor.preprocess_text(request.text)
        
        # Get prediction probabilities
        proba = model.predict_proba([processed_text])[0]
        
        # Get the highest probability and its index
        max_proba_idx = np.argmax(proba)
        confidence = proba[max_proba_idx]
        
        # Get the predicted class label
        predicted_type = model.classes_[max_proba_idx]
        
        logger.info(f"Predicted type: {predicted_type}, Confidence: {confidence:.3f}")
        
        return {
            "mbti_type": predicted_type,
            "confidence": float(confidence)
        }
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the app
if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)))