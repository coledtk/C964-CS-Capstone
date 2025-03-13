#!/ usr / bin / env python3
# SMS Spam Detection - Train and Save Model
# Cole Detrick - WGU Capstone Project

# This script loads the SMS Spam dataset, trains the model, and saves it for use in the application

import pandas as pd
import numpy as np
import os
import re
import logging
import pickle
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SMS_Train_Model")


def download_nltk_resources():
    """Download required NLTK resources if not already present"""
    try:
        nltk.data.find('corpora/stopwords')
        logger.info("NLTK resources already downloaded")
    except LookupError:
        logger.info("Downloading NLTK resources...")
        nltk.download('stopwords')
        logger.info("NLTK resources downloaded successfully")


def download_dataset(output_path="spam.csv"):
    """
    Download the SMS Spam Collection dataset if it's not already available.
    """
    try:
        # Check if the file already exists
        if os.path.exists(output_path):
            logger.info(f"Dataset already exists at {output_path}")
            return True

        # URL for the SMS Spam Collection dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

        logger.info(f"Downloading dataset from {url}")

        # Download and extract the dataset
        import requests
        import zipfile
        import io

        # Download the zip file
        response = requests.get(url)

        if response.status_code != 200:
            logger.error(f"Failed to download dataset: HTTP {response.status_code}")
            return False

        # Extract the zip file
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # The dataset is in a file named 'SMSSpamCollection'
            with z.open('SMSSpamCollection') as f:
                content = f.read().decode('utf-8')

        # Convert the dataset to CSV format
        lines = content.strip().split('\n')
        rows = [line.split('\t', 1) for line in lines]

        # Create a DataFrame
        df = pd.DataFrame(rows, columns=['label', 'message'])

        # Save as CSV
        df.to_csv(output_path, index=False)

        logger.info(f"Dataset downloaded and saved to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        return False


def load_dataset(file_path):
    """
    Load the SMS dataset from a CSV file.
    """
    try:
        logger.info(f"Loading dataset from {file_path}")

        # Read the CSV file
        df = pd.read_csv(file_path, encoding='latin-1')

        # Check if the dataset has the expected columns
        if 'v1' in df.columns and 'v2' in df.columns:
            # UCI dataset format
            df = df.rename(columns={'v1': 'label', 'v2': 'message'})
            df = df[['label', 'message']]

        # Convert labels to binary values (spam=1, ham=0)
        if 'label' in df.columns:
            df['label'] = df['label'].map({'spam': 1, 'ham': 0})

        logger.info(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        return df

    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None


def preprocess_text(text):
    """
    Preprocess text by removing punctuation, converting to lowercase,
    removing stopwords, and stemming.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenize
    tokens = text.split()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    return ' '.join(tokens)


def preprocess_dataset(df):
    """
    Preprocess the entire dataset.t
    """
    logger.info("Preprocessing dataset...")

    # Create a copy to preserve the original data
    processed_df = df.copy()

    # Preprocess messages
    processed_df['processed_message'] = processed_df['message'].apply(preprocess_text)

    # Calculate message length
    processed_df['message_length'] = processed_df['message'].apply(len)

    logger.info("Dataset preprocessing completed")
    return processed_df


def train_model(X_train, y_train):
    """
    Train the TF-IDF + Naive Bayes model.
    """
    logger.info("Training the model...")

    # Create a pipeline with TF-IDF and Multinomial Naive Bayes
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, min_df=2, ngram_range=(1, 2))),
        ('classifier', MultinomialNB(alpha=0.1))
    ])

    # Train the model
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()

    logger.info(f"Model training completed in {end_time - start_time:.2f} seconds")

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set.
    """
    logger.info("Evaluating the model...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Create a confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Calculate ROC curve and AUC
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Store metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'auc': roc_auc
    }

    # Log performance metrics
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")

    # Display classification report
    report = classification_report(y_test, y_pred)
    logger.info(f"Classification Report:\n{report}")

    return metrics


def save_model(model, model_path, metadata=None):
    """
    Save the trained model to a file.
    """
    try:
        logger.info(f"Saving model to {model_path}")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)

        # Add metadata if provided
        if metadata:
            model_dict = {
                'model': model,
                'metadata': metadata
            }
            with open(model_path, 'wb') as f:
                pickle.dump(model_dict, f)
        else:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

        logger.info("Model saved successfully")
        return True

    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        return False


def generate_visualizations(df, metrics, output_dir="visualizations"):
    """
    Generate visualizations for the dataset and model performance.
    """
    try:
        logger.info("Generating visualizations...")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # 1. Class Distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(x='label', data=df)
        plt.title('Distribution of Spam vs. Ham Messages')
        plt.xlabel('Label (0=Ham, 1=Spam)')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
        plt.close()

        # 2. Message Length Distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x='message_length', hue='label', bins=50, kde=True)
        plt.title('Message Length Distribution')
        plt.xlabel('Message Length (characters)')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, 'message_length_distribution.png'))
        plt.close()

        # 3. Confusion Matrix
        cm = metrics['confusion_matrix']
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()

        # 4. ROC Curve
        plt.figure(figsize=(8, 6))
        plt.plot(metrics['fpr'], metrics['tpr'], label=f'ROC Curve (AUC = {metrics["auc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()

        # 5. Performance Metrics
        plt.figure(figsize=(10, 6))
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Score': [
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1_score']
            ]
        })
        sns.barplot(x='Metric', y='Score', data=metrics_df)
        plt.title('Performance Metrics')
        plt.ylim(0, 1)
        plt.savefig(os.path.join(output_dir, 'performance_metrics.png'))
        plt.close()

        logger.info(f"Visualizations saved to {output_dir}")

    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")


def main():
    """Main function to train and save the model"""
    # Download NLTK resources
    download_nltk_resources()

    # Define paths
    dataset_path = "spam.csv"
    model_path = "sms_spam_model.pkl"
    vis_dir = "visualizations"

    # Download the dataset if it doesn't exist
    if not os.path.exists(dataset_path):
        if not download_dataset(dataset_path):
            logger.error("Failed to download the dataset. Exiting.")
            return

    # Load the dataset
    df = load_dataset(dataset_path)
    if df is None:
        logger.error("Failed to load the dataset. Exiting.")
        return

    # Preprocess the dataset
    processed_df = preprocess_dataset(df)

    # Split the data
    X = processed_df['processed_message']
    y = processed_df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Data split completed. Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)

    # Generate visualizations
    generate_visualizations(processed_df, metrics, vis_dir)

    # Save the model with metadata
    metadata = {
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'metrics': metrics,
        'feature_count': model.named_steps['tfidf'].max_features
    }

    if save_model(model, model_path, metadata):
        logger.info(f"Model saved successfully to {model_path}")
    else:
        logger.error("Failed to save the model")

    logger.info("Model training and evaluation completed successfully!")


if __name__ == "__main__":
    main()