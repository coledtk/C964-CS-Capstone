# SMS Spam Detection System
# Cole Detrick - WGU Capstone Project
# Completion Date: 04/10/2025

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import pickle
import os
import warnings
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("spam_detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SMS_Spam_Detector")

# Suppress warnings
warnings.filterwarnings('ignore')


class SMSSpamDetector:
    """
    A class for SMS spam detection using Naive Bayes and TF-IDF.
    Includes methods for data preprocessing, exploratory data analysis,
    model training, evaluation, and deployment.
    """

    def __init__(self, data_path=None):
        """
        Initialize the SMSSpamDetector object.

        Parameters:
        -----------
        data_path : str, optional
            Path to the SMS dataset.
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.vectorizer = None
        self.performance_metrics = {}
        self.stemmer = PorterStemmer()

        # Download NLTK resources if not already downloaded
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        logger.info("SMS Spam Detector initialized")

    def load_data(self, data_path=None):
        """
        Load the SMS spam dataset from a CSV file.

        Parameters:
        -----------
        data_path : str, optional
            Path to the dataset. If None, use the path specified during initialization.

        Returns:
        --------
        pandas.DataFrame
            The loaded dataset.
        """
        if data_path:
            self.data_path = data_path

        if not self.data_path:
            raise ValueError("Data path is not specified")

        try:
            # Check if the file exists
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"File not found: {self.data_path}")

            # Load the data
            self.df = pd.read_csv(self.data_path, encoding='latin-1')

            # Rename columns if they have default names (v1, v2, etc.)
            if 'v1' in self.df.columns and 'v2' in self.df.columns:
                self.df = self.df.rename(columns={'v1': 'label', 'v2': 'message'})
                self.df = self.df[['label', 'message']]

            # Convert labels to binary values (spam=1, ham=0)
            if 'label' in self.df.columns:
                self.df['label'] = self.df['label'].map({'spam': 1, 'ham': 0})

            logger.info(f"Data loaded successfully with {self.df.shape[0]} rows and {self.df.shape[1]} columns")
            return self.df

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def preprocess_text(self, text):
        """
        Preprocess text by removing punctuation, converting to lowercase,
        removing stopwords, and stemming.

        Parameters:
        -----------
        text : str
            The text to preprocess.

        Returns:
        --------
        str
            The preprocessed text.
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
        tokens = [self.stemmer.stem(token) for token in tokens]

        return ' '.join(tokens)

    def preprocess_data(self):
        """
        Apply preprocessing to the entire dataset.

        Returns:
        --------
        pandas.DataFrame
            The preprocessed dataset.
        """
        if self.df is None:
            logger.error("No data loaded. Call load_data() first.")
            return None

        logger.info("Starting data preprocessing...")

        # Create a copy to preserve the original data
        processed_df = self.df.copy()

        # Preprocess messages
        processed_df['processed_message'] = processed_df['message'].apply(self.preprocess_text)

        logger.info("Data preprocessing completed")
        return processed_df

    def explore_data(self):
        """
        Perform exploratory data analysis on the dataset.

        Returns:
        --------
        dict
            A dictionary containing various statistics about the dataset.
        """
        if self.df is None:
            logger.error("No data loaded. Call load_data() first.")
            return None

        logger.info("Starting exploratory data analysis...")

        # Calculate basic statistics
        stats = {}
        stats['total_messages'] = len(self.df)
        stats['spam_count'] = sum(self.df['label'] == 1)
        stats['ham_count'] = sum(self.df['label'] == 0)
        stats['spam_percentage'] = (stats['spam_count'] / stats['total_messages']) * 100
        stats['ham_percentage'] = (stats['ham_count'] / stats['total_messages']) * 100

        # Calculate message length statistics
        self.df['message_length'] = self.df['message'].apply(len)
        stats['avg_spam_length'] = self.df[self.df['label'] == 1]['message_length'].mean()
        stats['avg_ham_length'] = self.df[self.df['label'] == 0]['message_length'].mean()

        logger.info("Exploratory data analysis completed")

        # Display some statistics
        logger.info(f"Total messages: {stats['total_messages']}")
        logger.info(f"Spam messages: {stats['spam_count']} ({stats['spam_percentage']:.2f}%)")
        logger.info(f"Ham messages: {stats['ham_count']} ({stats['ham_percentage']:.2f}%)")
        logger.info(f"Average spam message length: {stats['avg_spam_length']:.2f} characters")
        logger.info(f"Average ham message length: {stats['avg_ham_length']:.2f} characters")

        return stats

    def visualize_data(self):
        """
        Create visualizations for the dataset.
        """
        if self.df is None:
            logger.error("No data loaded. Call load_data() first.")
            return

        logger.info("Creating data visualizations...")

        # Create a figure with multiple subplots
        plt.figure(figsize=(15, 10))

        # Plot 1: Class distribution
        plt.subplot(2, 2, 1)
        sns.countplot(x='label', data=self.df)
        plt.title('Distribution of Spam vs. Ham Messages')
        plt.xlabel('Label (0=Ham, 1=Spam)')
        plt.ylabel('Count')

        # Plot 2: Message length distribution
        plt.subplot(2, 2, 2)
        sns.histplot(data=self.df, x='message_length', hue='label', bins=50, kde=True)
        plt.title('Message Length Distribution')
        plt.xlabel('Message Length (characters)')
        plt.ylabel('Frequency')

        # Plot 3: Average message length by class
        plt.subplot(2, 2, 3)
        sns.barplot(x='label', y='message_length', data=self.df)
        plt.title('Average Message Length by Class')
        plt.xlabel('Label (0=Ham, 1=Spam)')
        plt.ylabel('Average Length (characters)')

        # Save the figure
        plt.tight_layout()
        plt.savefig('sms_spam_eda.png')
        logger.info("Visualizations saved as 'sms_spam_eda.png'")

        # Close the figure to free memory
        plt.close()

    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets.

        Parameters:
        -----------
        test_size : float, optional
            The proportion of the dataset to include in the test split.
        random_state : int, optional
            Random seed for reproducibility.

        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        if self.df is None:
            logger.error("No data loaded. Call load_data() first.")
            return None

        logger.info(f"Splitting data with test_size={test_size}")

        # Use the processed message if available, otherwise use the original message
        if 'processed_message' in self.df.columns:
            X = self.df['processed_message']
        else:
            X = self.df['message']

        y = self.df['label']

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        logger.info(
            f"Data split completed. Training set: {len(self.X_train)} samples, Test set: {len(self.X_test)} samples")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def build_model(self, optimize=True):
        """
        Build and train the TF-IDF + Naive Bayes model.

        Parameters:
        -----------
        optimize : bool, optional
            Whether to perform hyperparameter optimization using GridSearchCV.

        Returns:
        --------
        sklearn.pipeline.Pipeline
            The trained model.
        """
        if self.X_train is None or self.y_train is None:
            logger.error("Data not split. Call split_data() first.")
            return None

        logger.info("Building the model...")

        # Create a pipeline with TF-IDF and Multinomial Naive Bayes
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('classifier', MultinomialNB())
        ])

        if optimize:
            logger.info("Performing hyperparameter optimization...")

            # Define hyperparameter grid
            param_grid = {
                'tfidf__max_features': [3000, 5000, 7000],
                'tfidf__min_df': [1, 2, 3],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'classifier__alpha': [0.1, 0.5, 1.0]
            }

            # Perform grid search
            grid_search = GridSearchCV(
                self.model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
            )

            grid_search.fit(self.X_train, self.y_train)

            # Get the best parameters and model
            logger.info(f"Best parameters: {grid_search.best_params_}")
            self.model = grid_search.best_estimator_
        else:
            # Train the model without optimization
            self.model.fit(self.X_train, self.y_train)

        logger.info("Model training completed")

        # Extract the vectorizer for later use
        self.vectorizer = self.model.named_steps['tfidf']

        return self.model

    def evaluate_model(self):
        """
        Evaluate the model on the test set.

        Returns:
        --------
        dict
            A dictionary containing performance metrics.
        """
        if self.model is None:
            logger.error("No model trained. Call build_model() first.")
            return None

        if self.X_test is None or self.y_test is None:
            logger.error("Data not split. Call split_data() first.")
            return None

        logger.info("Evaluating the model...")

        # Make predictions
        y_pred = self.model.predict(self.X_test)

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)

        # Create a confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)

        # Calculate ROC curve and AUC
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        # Store metrics
        self.performance_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'fpr': fpr,
            'tpr': tpr,
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
        report = classification_report(self.y_test, y_pred)
        logger.info(f"Classification Report:\n{report}")

        return self.performance_metrics

    def visualize_evaluation(self):
        """
        Create visualizations for model evaluation.
        """
        if not self.performance_metrics:
            logger.error("No evaluation metrics available. Call evaluate_model() first.")
            return

        logger.info("Creating evaluation visualizations...")

        # Create a figure with multiple subplots
        plt.figure(figsize=(15, 10))

        # Plot 1: Confusion Matrix
        plt.subplot(2, 2, 1)
        cm = self.performance_metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        # Plot 2: ROC Curve
        plt.subplot(2, 2, 2)
        plt.plot(self.performance_metrics['fpr'], self.performance_metrics['tpr'],
                 label=f"AUC = {self.performance_metrics['auc']:.4f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')

        # Plot 3: Performance Metrics
        plt.subplot(2, 2, 3)
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        values = [self.performance_metrics[m] for m in metrics]
        sns.barplot(x=metrics, y=values)
        plt.title('Performance Metrics')
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.ylim(0, 1)

        # Save the figure
        plt.tight_layout()
        plt.savefig('model_evaluation.png')
        logger.info("Evaluation visualizations saved as 'model_evaluation.png'")

        # Close the figure to free memory
        plt.close()

    def save_model(self, model_path='sms_spam_model.pkl'):
        """
        Save the trained model to a file.

        Parameters:
        -----------
        model_path : str, optional
            Path where the model should be saved.

        Returns:
        --------
        bool
            True if the model was saved successfully, False otherwise.
        """
        if self.model is None:
            logger.error("No model trained. Call build_model() first.")
            return False

        logger.info(f"Saving model to {model_path}...")

        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info("Model saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    def load_model(self, model_path='sms_spam_model.pkl'):
        """
        Load a trained model from a file.

        Parameters:
        -----------
        model_path : str, optional
            Path to the saved model.

        Returns:
        --------
        bool
            True if the model was loaded successfully, False otherwise.
        """
        logger.info(f"Loading model from {model_path}...")

        try:
            with open(model_path, 'rb') as f:
                loaded_object = pickle.load(f)

            # Based on inspection, we know it's a dict with 'model' and 'metadata' keys
            if isinstance(loaded_object, dict) and 'model' in loaded_object:
                # Extract the model from the dictionary
                self.model = loaded_object['model']
                self.metadata = loaded_object.get('metadata', {})
            else:
                # Direct model without metadata (fallback)
                self.model = loaded_object
                self.metadata = {}

            # Extract the vectorizer
            if hasattr(self.model, 'named_steps'):
                if 'tfidf' in self.model.named_steps:
                    self.vectorizer = self.model.named_steps['tfidf']
                elif 'count_vec' in self.model.named_steps:
                    self.vectorizer = self.model.named_steps['count_vec']

            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def predict(self, message):
        """
        Predict whether a message is spam or not.

        Parameters:
        -----------
        message : str
            The message to classify.

        Returns:
        --------
        tuple
            (prediction, probability)
        """
        if self.model is None:
            logger.error("No model loaded. Call build_model() or load_model() first.")
            return None

        # Preprocess the message
        processed_message = self.preprocess_text(message)

        # Make prediction
        prediction = self.model.predict([processed_message])[0]
        probability = self.model.predict_proba([processed_message])[0][prediction]

        result = "SPAM" if prediction == 1 else "HAM"
        logger.info(f"Message classified as {result} with probability {probability:.4f}")

        return result, probability

    def batch_predict(self, messages):
        """
        Predict whether multiple messages are spam or not.

        Parameters:
        -----------
        messages : list
            List of messages to classify.

        Returns:
        --------
        pandas.DataFrame
            DataFrame with messages, predictions, and probabilities.
        """
        if self.model is None:
            logger.error("No model loaded. Call build_model() or load_model() first.")
            return None

        # Preprocess messages
        processed_messages = [self.preprocess_text(msg) for msg in messages]

        # Make predictions
        predictions = self.model.predict(processed_messages)
        probabilities = self.model.predict_proba(processed_messages)

        # Create a DataFrame with results
        results = pd.DataFrame({
            'message': messages,
            'prediction': predictions,
            'probability': [prob[pred] for pred, prob in zip(predictions, probabilities)],
            'result': ['SPAM' if pred == 1 else 'HAM' for pred in predictions]
        })

        logger.info(f"Batch prediction completed for {len(messages)} messages")

        return results

    def get_important_features(self, n=20):
        """
        Get the most important features (words) for spam and ham classification.

        Parameters:
        -----------
        n : int, optional
            Number of top features to return for each class.

        Returns:
        --------
        tuple
            (spam_features, ham_features)
        """
        if self.model is None:
            logger.error("No model loaded. Call build_model() or load_model() first.")
            return None

        logger.info(f"Extracting top {n} features for each class...")

        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()

        # Get feature importances
        classifier = self.model.named_steps['classifier']
        feature_importances = classifier.feature_log_prob_

        # Higher values in feature_log_prob_[1] indicate words more likely to be in spam
        # Higher values in feature_log_prob_[0] indicate words more likely to be in ham
        spam_indices = feature_importances[1].argsort()[-n:][::-1]
        ham_indices = feature_importances[0].argsort()[-n:][::-1]

        spam_features = [(feature_names[i], feature_importances[1][i]) for i in spam_indices]
        ham_features = [(feature_names[i], feature_importances[0][i]) for i in ham_indices]

        return spam_features, ham_features

    def visualize_important_features(self, n=20):
        """
        Visualize the most important features for spam and ham classification.

        Parameters:
        -----------
        n : int, optional
            Number of top features to visualize for each class.
        """
        features = self.get_important_features(n)
        if features is None:
            return

        spam_features, ham_features = features

        logger.info("Creating feature importance visualizations...")

        # Create a figure with two subplots
        plt.figure(figsize=(15, 10))

        # Plot 1: Top spam features
        plt.subplot(2, 1, 1)
        words, values = zip(*spam_features)
        plt.barh(words, values)
        plt.title(f'Top {n} Words Most Indicative of Spam')
        plt.xlabel('Log Probability')

        # Plot 2: Top ham features
        plt.subplot(2, 1, 2)
        words, values = zip(*ham_features)
        plt.barh(words, values)
        plt.title(f'Top {n} Words Most Indicative of Ham')
        plt.xlabel('Log Probability')

        # Save the figure
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        logger.info("Feature importance visualizations saved as 'feature_importance.png'")

        # Close the figure to free memory
        plt.close()

    def monitor_performance(self, actual_labels, predicted_labels):
        """
        Monitor the model's performance over time.

        Parameters:
        -----------
        actual_labels : array-like
            The true labels.
        predicted_labels : array-like
            The predicted labels.

        Returns:
        --------
        dict
            Dictionary containing performance metrics.
        """
        # Calculate metrics
        accuracy = accuracy_score(actual_labels, predicted_labels)
        precision = precision_score(actual_labels, predicted_labels)
        recall = recall_score(actual_labels, predicted_labels)
        f1 = f1_score(actual_labels, predicted_labels)

        # Create a timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Log the metrics
        logger.info(f"Performance at {timestamp}:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")

        # Return the metrics
        return {
            'timestamp': timestamp,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def run_pipeline(self, data_path=None, optimize=True, save_model=True, model_path='sms_spam_model.pkl'):
        """
        Run the entire pipeline: load data, preprocess, split, train, evaluate, save model.

        Parameters:
        -----------
        data_path : str, optional
            Path to the dataset.
        optimize : bool, optional
            Whether to perform hyperparameter optimization.
        save_model : bool, optional
            Whether to save the trained model.
        model_path : str, optional
            Path to save the model to.

        Returns:
        --------
        dict
            Dictionary containing performance metrics.
        """
        # Load data
        self.load_data(data_path)

        # Preprocess data
        self.preprocess_data()

        # Explore data
        self.explore_data()

        # Visualize data
        self.visualize_data()

        # Split data
        self.split_data()

        # Build and train model
        self.build_model(optimize)

        # Evaluate model
        metrics = self.evaluate_model()

        # Visualize evaluation
        self.visualize_evaluation()

        # Visualize important features
        self.visualize_important_features()

        # Save model if requested
        if save_model:
            self.save_model(model_path)

        return metrics


# Example usage of the SMS Spam Detector
if __name__ == "__main__":
    # Create an instance of the detector
    detector = SMSSpamDetector(data_path="spam.csv")

    # Run the entire pipeline
    metrics = detector.run_pipeline()

    # Make a prediction
    message = "Congratulations! You've won a free vacation. Call now to claim your prize!"
    result, probability = detector.predict(message)
    print(f"Message: {message}")
    print(f"Prediction: {result} (Probability: {probability:.4f})")