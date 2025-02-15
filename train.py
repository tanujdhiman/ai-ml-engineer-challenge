import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import logging
import os
from pathlib import Path
import emoji
import contractions
from imblearn.over_sampling import SMOTE

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Text preprocessing utility class"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the input text with multiple cleaning steps
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

def load_and_preprocess_data(data_path: str) -> Tuple[pd.DataFrame, TextPreprocessor]:
    """
    Load and preprocess the MBTI dataset with detailed analysis
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Basic validation
    required_columns = ['type', 'posts']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Dataset must contain columns: {required_columns}")
    
    # Data analysis
    logger.info("\nDataset Overview:")
    logger.info(f"Total samples: {len(df)}")
    logger.info("\nMBTI Type Distribution:")
    type_dist = df['type'].value_counts()
    logger.info(type_dist)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Preprocess posts
    logger.info("Preprocessing text data...")
    df['processed_posts'] = df['posts'].apply(preprocessor.preprocess_text)
    
    # Calculate text statistics
    df['word_count'] = df['processed_posts'].apply(lambda x: len(x.split()))
    
    logger.info("\nText Statistics:")
    logger.info(f"Average words per post: {df['word_count'].mean():.2f}")
    logger.info(f"Median words per post: {df['word_count'].median():.2f}")
    
    return df, preprocessor

def create_model_pipeline(tfidf_params: Dict = None, rf_params: Dict = None) -> Pipeline:
    """
    Create the ML pipeline with custom parameters
    """
    if tfidf_params is None:
        tfidf_params = {
            'max_features': 5000,
            'ngram_range': (1, 2),
            'stop_words': 'english',
            'min_df': 2,
            'max_df': 0.95
        }
    
    if rf_params is None:
        rf_params = {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
    
    return Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('classifier', RandomForestClassifier(**rf_params))
    ])

def train_and_evaluate_model(df: pd.DataFrame) -> Tuple[Pipeline, Dict]:
    """
    Train the model with cross-validation and detailed evaluation
    """
    logger.info("Starting model training...")
    
    # Prepare data
    X = df['processed_posts'].values
    y = df['type'].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Handle class imbalance with SMOTE
    logger.info("Applying SMOTE for class balancing...")
    pipeline = create_model_pipeline()
    tfidf = pipeline.named_steps['tfidf']
    X_train_tfidf = tfidf.fit_transform(X_train)
    
    # smote = SMOTE(random_state=42)
    # X_train_balanced, y_train_balanced = smote.fit_resample(X_train_tfidf, y_train)
    
    # Train model with grid search
    pipeline.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    )
    
    logger.info(f"\nCross-validation scores: {cv_scores}")
    logger.info(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Final evaluation
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Generate evaluation metrics
    evaluation = {
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'cv_scores': cv_scores
    }
    
    logger.info("\nClassification Report:")
    logger.info(evaluation['classification_report'])
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        evaluation['confusion_matrix'],
        annot=True,
        fmt='d',
        cmap='Blues'
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return pipeline, evaluation

def save_model_and_artifacts(
    pipeline: Pipeline,
    preprocessor: TextPreprocessor,
    model_dir: str
):
    """
    Save the trained model and associated artifacts
    """
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    # Save main pipeline
    model_path = os.path.join(model_dir, 'mbti_model.joblib')
    logger.info(f"Saving model to {model_path}")
    joblib.dump(pipeline, model_path)
    
    # Save preprocessor
    preprocessor_path = os.path.join(model_dir, 'preprocessor.joblib')
    logger.info(f"Saving preprocessor to {preprocessor_path}")
    joblib.dump(preprocessor, preprocessor_path)
    
    # Save feature names
    feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
    features_path = os.path.join(model_dir, 'feature_names.txt')
    with open(features_path, 'w') as f:
        f.write('\n'.join(feature_names))

def main():
    # Configuration
    DATA_PATH = "data/MBTI 500.csv"
    MODEL_DIR = "models"
    
    try:
        # Load and preprocess data
        df, preprocessor = load_and_preprocess_data(DATA_PATH)
        
        # Train and evaluate model
        pipeline, evaluation = train_and_evaluate_model(df)
        
        # Save model and artifacts
        save_model_and_artifacts(pipeline, preprocessor, MODEL_DIR)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()