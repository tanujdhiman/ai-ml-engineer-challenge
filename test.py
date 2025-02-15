import joblib
import numpy as np
from typing import Tuple, List
import logging
from pathlib import Path
import re
import emoji
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)


class MBTIPredictor:
    """MBTI Personality Type Predictor"""

    def __init__(self, model_path: str = "models/mbti_model.joblib", 
                 preprocessor_path: str = "models/preprocessor.joblib"):
        """
        Initialize the MBTI predictor.

        Args:
            model_path: Path to the trained model file.
            preprocessor_path: Path to the text preprocessor.
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.preprocessor = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.load_model()

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the input text with multiple cleaning steps.

        Args:
            text: Input text to preprocess.

        Returns:
            Preprocessed text.
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

    def load_model(self):
        """
        Load the trained model from disk.
        """
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
        if not Path(self.preprocessor_path).exists():
            raise FileNotFoundError(f"Preprocessor not found at {self.preprocessor_path}")
        
        logger.info(f"Loading model from {self.model_path}")
        self.model = joblib.load(self.model_path)
        
        logger.info(f"Loading preprocessor from {self.preprocessor_path}")
        try:
            self.preprocessor = joblib.load(self.preprocessor_path)
        except AttributeError:
            self.preprocessor = self.preprocess_text
        
        logger.info("Model and preprocessor loaded successfully")

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict MBTI personality type from input text.

        Args:
            text: Input text to analyze.

        Returns:
            Tuple of (predicted_type, confidence_score).
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model or preprocessor not loaded. Call load_model() first")
        
        # Get prediction probabilities
        try:
            # Preprocess the text first
            processed_text = self.preprocess_text(text)
            
            # Get raw prediction probabilities
            proba = self.model.predict_proba([processed_text])[0]
            
            # Get the highest probability and its index
            max_proba_idx = np.argmax(proba)
            confidence = proba[max_proba_idx]
            
            # Get the predicted class label
            predicted_type = self.model.classes_[max_proba_idx]
            
            logger.debug(f"Predicted type: {predicted_type}, Confidence: {confidence:.3f}")
            
            return predicted_type, float(confidence)
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Predict MBTI personality types for multiple texts.

        Args:
            texts: List of input texts to analyze.

        Returns:
            List of tuples (predicted_type, confidence_score).
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty")
        
        predictions = []
        for text in texts:
            pred_type, confidence = self.predict(text)
            predictions.append((pred_type, confidence))
        
        return predictions


# Example usage
if __name__ == "__main__":
    # Set up logging for the example
    logging.basicConfig(level=logging.INFO)
    
    # Create predictor instance
    predictor = MBTIPredictor()
    
    # Example prediction
    sample_text = "I love deep philosophical conversations and spending time alone reading."
    mbti_type, confidence = predictor.predict(sample_text)
    
    print(f"\nInput text: {sample_text}")
    print(f"Predicted MBTI type: {mbti_type}")
    print(f"Confidence score: {confidence:.3f}")