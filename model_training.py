#!/usr/bin/env python3
# SMS Spam Detection - Model Training Script
# Cole Detrick - WGU Capstone Project

import pandas as pd
import numpy as np
import os
import argparse
import logging
import pickle
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import joblib
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SMS_Model_Training")


def load_data(train_path, test_path=None, test_size=0.2, random_state=42):
    """
    Load training and testing data from files or split a single dataset.
    """
    try:
        logger.info(f"Loading data from {train_path}")

        # Load the training data
        train_df = pd.read_csv(train_path)

        # Check if the dataset has the expected columns
        if 'label' not in train_df.columns:
            if 'v1' in train_df.columns and 'v2' in train_df.columns:
                # UCI dataset format
                train_df = train_df.rename(columns={'v1': 'label', 'v2': 'message'})
                train_df = train_df[['label', 'message']]
            else:
                logger.error("Training data does not have 'label' column")
                return None, None, None, None

        if 'message' not in train_df.columns and 'processed_message' not in train_df.columns:
            logger.error("Training data does not have 'message' or 'processed_message' column")
            return None, None, None, None

        # Convert labels to binary values (spam=1, ham=0) if needed
        if train_df['label'].dtype == object:
            train_df['label'] = train_df['label'].map({'spam': 1, 'ham': 0})

        # Use processed message if available
        message_col = 'processed_message' if 'processed_message' in train_df.columns else 'message'

        if test_path:
            # Load the testing data
            logger.info(f"Loading test data from {test_path}")
            test_df = pd.read_csv(test_path)

            # Check if the dataset has the expected columns
            if 'label' not in test_df.columns:
                if 'v1' in test_df.columns and 'v2' in test_df.columns:
                    # UCI dataset format
                    test_df = test_df.rename(columns={'v1': 'label', 'v2': 'message'})
                    test_df = test_df[['label', 'message']]
                else:
                    logger.error("Test data does not have 'label' column")
                    return None, None, None, None

            if message_col not in test_df.columns:
                if 'message' in test_df.columns and message_col == 'processed_message':
                    logger.warning("Test data does not have 'processed_message' column. Using 'message' instead.")
                    message_col = 'message'
                else:
                    logger.error(f"Test data does not have '{message_col}' column")
                    return None, None, None, None

            # Convert labels to binary values (spam=1, ham=0) if needed
            if test_df['label'].dtype == object:
                test_df['label'] = test_df['label'].map({'spam': 1, 'ham': 0})

            X_train = train_df[message_col]
            y_train = train_df['label']
            X_test = test_df[message_col]
            y_test = test_df['label']
        else:
            # Split the data
            logger.info(f"Splitting data with test_size={test_size}")
            X = train_df[message_col]
            y = train_df['label']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

        logger.info(f"Data loaded successfully. Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None, None, None, None


def build_basic_model():
    """
    Build a basic TF-IDF + Naive Bayes model.
    """
    logger.info("Building basic model")
    # Create a pipeline with TF-IDF and Multinomial Naive Bayes
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('classifier', MultinomialNB())
    ])

    return model


def build_count_vector_model():
    """
    Build a CountVectorizer + Naive Bayes model.
    """
    logger.info("Building CountVectorizer model")

    # Create a pipeline with CountVectorizer and Multinomial Naive Bayes
    model = Pipeline([
        ('count_vec', CountVectorizer(max_features=5000)),
        ('classifier', MultinomialNB())
    ])

    return model


def optimize_model(model, X_train, y_train, cv=5):
    """
    Optimize the model using grid search.
    """
    logger.info("Optimizing model with GridSearchCV")

    # Check which vectorizer is in the pipeline
    if 'tfidf' in model.named_steps:
        vectorizer_name = 'tfidf'
    elif 'count_vec' in model.named_steps:
        vectorizer_name = 'count_vec'
    else:
        logger.error("Unknown vectorizer in pipeline")
        return model

    # Define hyperparameter grid
    param_grid = {
        f'{vectorizer_name}__max_features': [3000, 5000, 7000],
        f'{vectorizer_name}__min_df': [1, 2, 3],
        f'{vectorizer_name}__ngram_range': [(1, 1), (1, 2)],
        'classifier__alpha': [0.1, 0.5, 1.0]
    }

    # Perform grid search
    start_time = time.time()
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1
    )

    grid_search.fit(X_train, y_train)
    end_time = time.time()

    # Get the best parameters and model
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    logger.info(f"Grid search completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best F1 score: {best_score:.4f}")

    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test, output_dir="model_evaluation"):
    """
    Evaluate the model on the test set and generate visualizations.
    """
    try:
        logger.info("Evaluating model on test set")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

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
        report = classification_report(y_test, y_pred)
        logger.info(f"Classification Report:\n{report}")

        # Create visualizations

        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()

        # 2. ROC Curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()

        # 3. Performance Metrics
        plt.figure(figsize=(10, 6))
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Score': [accuracy, precision, recall, f1]
        })
        sns.barplot(x='Metric', y='Score', data=metrics_df)
        plt.title('Performance Metrics')
        plt.ylim(0, 1)
        plt.savefig(os.path.join(output_dir, 'performance_metrics.png'))
        plt.close()

        # 4. Feature Importance
        try:
            # Extract the vectorizer
            if 'tfidf' in model.named_steps:
                vectorizer = model.named_steps['tfidf']
            elif 'count_vec' in model.named_steps:
                vectorizer = model.named_steps['count_vec']
            else:
                raise ValueError("Vectorizer not found in pipeline")

            # Get feature names
            feature_names = vectorizer.get_feature_names_out()

            # Get feature importances
            classifier = model.named_steps['classifier']
            feature_importances = classifier.feature_log_prob_

            # Get top spam and ham features
            spam_indices = feature_importances[1].argsort()[-20:][::-1]
            ham_indices = feature_importances[0].argsort()[-20:][::-1]

            spam_features = [(feature_names[i], feature_importances[1][i]) for i in spam_indices]
            ham_features = [(feature_names[i], feature_importances[0][i]) for i in ham_indices]

            # Create DataFrames
            spam_df = pd.DataFrame(spam_features, columns=['Feature', 'Importance'])
            ham_df = pd.DataFrame(ham_features, columns=['Feature', 'Importance'])

            # Plot feature importance
            plt.figure(figsize=(12, 10))

            plt.subplot(2, 1, 1)
            sns.barplot(x='Importance', y='Feature', data=spam_df)
            plt.title('Top 20 Spam Indicators')
            plt.xlabel('Log Probability')

            plt.subplot(2, 1, 2)
            sns.barplot(x='Importance', y='Feature', data=ham_df)
            plt.title('Top 20 Ham Indicators')
            plt.xlabel('Log Probability')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
            plt.close()

        except Exception as e:
            logger.error(f"Error creating feature importance visualization: {str(e)}")

        # Save metrics to a file
        with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"ROC AUC: {roc_auc:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(f"{cm}\n\n")
            f.write("Classification Report:\n")
            f.write(f"{report}\n")

        logger.info(f"Evaluation visualizations saved to {output_dir}")

        return metrics

    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        return None


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


def cross_validate_model(model, X, y, cv=5):
    """Perform cross-validation on the model."""
    logger.info(f"Performing {cv}-fold cross-validation")

    # Handle NaN values in X
    if isinstance(X, pd.Series) and X.isna().any():
        logger.warning(f"Found {X.isna().sum()} NaN values in cross-validation data. Replacing with empty strings.")
        X = X.fillna('')

    try:
        # Calculate cross-validation scores
        accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        precision_scores = cross_val_score(model, X, y, cv=cv, scoring='precision')
        recall_scores = cross_val_score(model, X, y, cv=cv, scoring='recall')
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')

        # Log the scores
        logger.info(f"Cross-Validation Accuracy: {accuracy_scores.mean():.4f} (±{accuracy_scores.std():.4f})")
        logger.info(f"Cross-Validation Precision: {precision_scores.mean():.4f} (±{precision_scores.std():.4f})")
        logger.info(f"Cross-Validation Recall: {recall_scores.mean():.4f} (±{recall_scores.std():.4f})")
        logger.info(f"Cross-Validation F1 Score: {f1_scores.mean():.4f} (±{f1_scores.std():.4f})")

        # Store the scores
        cv_scores = {
            'accuracy': {
                'mean': accuracy_scores.mean(),
                'std': accuracy_scores.std(),
                'scores': accuracy_scores.tolist()
            },
            'precision': {
                'mean': precision_scores.mean(),
                'std': precision_scores.std(),
                'scores': precision_scores.tolist()
            },
            'recall': {
                'mean': recall_scores.mean(),
                'std': recall_scores.std(),
                'scores': recall_scores.tolist()
            },
            'f1': {
                'mean': f1_scores.mean(),
                'std': f1_scores.std(),
                'scores': f1_scores.tolist()
            }
        }

        return cv_scores
    except Exception as e:
        logger.error(f"Error during cross-validation: {str(e)}")
        logger.exception("Exception details:")

        # Return empty scores if cross-validation fails
        cv_scores = {
            'accuracy': {'mean': 0, 'std': 0, 'scores': []},
            'precision': {'mean': 0, 'std': 0, 'scores': []},
            'recall': {'mean': 0, 'std': 0, 'scores': []},
            'f1': {'mean': 0, 'std': 0, 'scores': []}
        }
        return cv_scores


def main():
    """Main function to run the model training script"""
    parser = argparse.ArgumentParser(description="SMS Spam Detection - Model Training Script")

    parser.add_argument("--train", "-t", help="Path to training data file", default="cleaned_spam.csv")
    parser.add_argument("--test", help="Path to testing data file")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Proportion of data to use for testing (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--model-type", choices=['tfidf', 'count_vec'], default='tfidf',
                        help="Type of vectorizer to use (default: tfidf)")
    parser.add_argument("--optimize", action="store_true", help="Perform hyperparameter optimization")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds (default: 5)")
    parser.add_argument("--output", "-o", help="Path to save the trained model", default="sms_spam_model.pkl")
    parser.add_argument("--eval-dir", default="model_evaluation", help="Directory to save evaluation visualizations")

    args = parser.parse_args()

    # Load the data
    X_train, X_test, y_train, y_test = load_data(
        args.train, args.test, args.test_size, args.random_state
    )

    if X_train.isna().any():
        logger.warning(f"Found {X_train.isna().sum()} NaN values in training data. Replacing with empty strings.")
        X_train = X_train.fillna('')

    if X_test is not None and X_test.isna().any():
        logger.warning(f"Found {X_test.isna().sum()} NaN values in test data. Replacing with empty strings.")
        X_test = X_test.fillna('')

    if X_train is None:
        print("Failed to load the data. Exiting.")
        sys.exit(1)

    # Build the model
    if args.model_type == 'tfidf':
        model = build_basic_model()
    else:
        model = build_count_vector_model()

    # Perform cross-validation
    cv_scores = cross_validate_model(model, X_train, y_train, args.cv)

    # Optimize the model if requested
    if args.optimize:
        model = optimize_model(model, X_train, y_train, args.cv)

    # Train the model
    logger.info("Training the model")
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    logger.info(f"Model training completed in {end_time - start_time:.2f} seconds")

    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test, args.eval_dir)

    # Save the model with metadata
    metadata = {
        'model_type': args.model_type,
        'optimized': args.optimize,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_data': args.train,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'metrics': metrics,
        'cross_validation': cv_scores
    }

    save_model(model, args.output, metadata)

    print("Model training completed successfully!")
    print(f"Model saved to {args.output}")
    print(f"Evaluation results saved to {args.eval_dir}")

# Call main
if __name__ == "__main__":
    main()