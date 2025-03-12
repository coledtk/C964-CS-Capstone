#!/usr/bin/env python3
# SMS Spam Detection - Data Preparation Script
# Cole Detrick - WGU Capstone Project
# Completion Date: 04/10/2025

import pandas as pd
import numpy as np
import os
import re
import argparse
import logging
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_preparation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SMS_Data_Preparation")


def download_dataset(output_path="spam.csv"):
    """
    Download the SMS Spam Collection dataset if it's not already available.

    Parameters:
    -----------
    output_path : str, optional
        Path to save the dataset (default: "spam.csv")

    Returns:
    --------
    bool
        True if successful, False otherwise
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

    Parameters:
    -----------
    file_path : str
        Path to the CSV file

    Returns:
    --------
    pandas.DataFrame or None
        The loaded dataset or None if an error occurred
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


def preprocess_text(text, stemming=True):
    """
    Preprocess text by removing punctuation, converting to lowercase,
    removing stopwords, and optionally stemming.

    Parameters:
    -----------
    text : str
        The text to preprocess
    stemming : bool, optional
        Whether to apply stemming (default: True)

    Returns:
    --------
    str
        The preprocessed text
    """
    # Handle NaN values
    if pd.isna(text):
        return ""

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

    # Stemming if requested
    if stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]

    return ' '.join(tokens)


def clean_dataset(df, stemming=True):
    """
    Clean the dataset by preprocessing the text and handling missing values.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to clean
    stemming : bool, optional
        Whether to apply stemming (default: True)

    Returns:
    --------
    pandas.DataFrame
        The cleaned dataset
    """
    try:
        logger.info("Cleaning dataset...")

        # Create a copy to preserve the original data
        cleaned_df = df.copy()

        # Check for missing values
        missing_values = cleaned_df.isnull().sum()

        if missing_values.sum() > 0:
            logger.warning(f"Found missing values: {missing_values}")

            # Fill missing values with empty strings instead of dropping
            if 'message' in cleaned_df.columns and cleaned_df['message'].isnull().any():
                logger.info(
                    f"Filling {cleaned_df['message'].isnull().sum()} NaN values in message column with empty strings")
                cleaned_df['message'] = cleaned_df['message'].fillna('')

        # Preprocess messages
        logger.info("Preprocessing messages...")
        cleaned_df['processed_message'] = cleaned_df['message'].apply(lambda x: preprocess_text(x, stemming))

        # Verify no NaN values in processed messages
        if cleaned_df['processed_message'].isnull().any():
            logger.warning(
                f"Found {cleaned_df['processed_message'].isnull().sum()} NaN values in processed_message after preprocessing")
            logger.info("Filling remaining NaN values with empty strings")
            cleaned_df['processed_message'] = cleaned_df['processed_message'].fillna('')

        # Calculate message length
        cleaned_df['message_length'] = cleaned_df['message'].apply(len)

        logger.info("Dataset cleaning completed")
        return cleaned_df

    except Exception as e:
        logger.error(f"Error cleaning dataset: {str(e)}")
        logger.exception("Exception details:")
        return df


def explore_dataset(df, output_dir="data_exploration"):
    """
    Perform exploratory data analysis on the dataset and generate visualizations.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to explore
    output_dir : str, optional
        Directory to save the visualizations (default: "data_exploration")

    Returns:
    --------
    dict
        A dictionary containing various statistics about the dataset
    """
    try:
        logger.info("Performing exploratory data analysis...")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Calculate basic statistics
        stats = {}
        stats['total_messages'] = len(df)
        stats['spam_count'] = sum(df['label'] == 1)
        stats['ham_count'] = sum(df['label'] == 0)
        stats['spam_percentage'] = (stats['spam_count'] / stats['total_messages']) * 100
        stats['ham_percentage'] = (stats['ham_count'] / stats['total_messages']) * 100

        # Message length statistics
        stats['avg_spam_length'] = df[df['label'] == 1]['message_length'].mean()
        stats['avg_ham_length'] = df[df['label'] == 0]['message_length'].mean()
        stats['max_message_length'] = df['message_length'].max()
        stats['min_message_length'] = df['message_length'].min()

        # Display statistics
        logger.info(f"Total messages: {stats['total_messages']}")
        logger.info(f"Spam messages: {stats['spam_count']} ({stats['spam_percentage']:.2f}%)")
        logger.info(f"Ham messages: {stats['ham_count']} ({stats['ham_percentage']:.2f}%)")
        logger.info(f"Average spam message length: {stats['avg_spam_length']:.2f} characters")
        logger.info(f"Average ham message length: {stats['avg_ham_length']:.2f} characters")

        # Create visualizations

        # 1. Distribution of Spam vs. Ham
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

        # 3. Average Message Length by Class
        plt.figure(figsize=(10, 6))
        sns.barplot(x='label', y='message_length', data=df)
        plt.title('Average Message Length by Class')
        plt.xlabel('Label (0=Ham, 1=Spam)')
        plt.ylabel('Average Length (characters)')
        plt.savefig(os.path.join(output_dir, 'avg_message_length.png'))
        plt.close()

        # 4. Word Clouds

        # Spam Word Cloud
        spam_messages = ' '.join(df[df['label'] == 1]['processed_message'].astype(str))
        spam_wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=100,
            colormap='Reds'
        ).generate(spam_messages)

        plt.figure(figsize=(10, 5))
        plt.imshow(spam_wordcloud, interpolation='bilinear')
        plt.title('Word Cloud for Spam Messages')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'spam_wordcloud.png'))
        plt.close()

        # Ham Word Cloud
        ham_messages = ' '.join(df[df['label'] == 0]['processed_message'].astype(str))
        ham_wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=100,
            colormap='Blues'
        ).generate(ham_messages)

        plt.figure(figsize=(10, 5))
        plt.imshow(ham_wordcloud, interpolation='bilinear')
        plt.title('Word Cloud for Ham Messages')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'ham_wordcloud.png'))
        plt.close()

        logger.info(f"Visualizations saved to {output_dir}")

        # Create frequency distribution of top words
        from collections import Counter
        import nltk

        # For spam messages
        spam_words = ' '.join(df[df['label'] == 1]['processed_message']).split()
        spam_word_count = Counter(spam_words).most_common(20)

        # For ham messages
        ham_words = ' '.join(df[df['label'] == 0]['processed_message']).split()
        ham_word_count = Counter(ham_words).most_common(20)

        # Create DataFrames
        spam_words_df = pd.DataFrame(spam_word_count, columns=['word', 'count'])
        ham_words_df = pd.DataFrame(ham_word_count, columns=['word', 'count'])

        # Plot word frequency
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        sns.barplot(x='count', y='word', data=spam_words_df)
        plt.title('Top 20 Words in Spam Messages')
        plt.xlabel('Count')
        plt.ylabel('Word')

        plt.subplot(2, 1, 2)
        sns.barplot(x='count', y='word', data=ham_words_df)
        plt.title('Top 20 Words in Ham Messages')
        plt.xlabel('Count')
        plt.ylabel('Word')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'word_frequency.png'))
        plt.close()

        return stats

    except Exception as e:
        logger.error(f"Error exploring dataset: {str(e)}")
        return {}


def split_dataset(df, test_size=0.2, random_state=42, stratify=True, output_dir="split_data"):
    """
    Split the dataset into training and testing sets.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to split
    test_size : float, optional
        The proportion of the dataset to include in the test split (default: 0.2)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    stratify : bool, optional
        Whether to stratify the split based on the label (default: True)
    output_dir : str, optional
        Directory to save the split datasets (default: "split_data")

    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    try:
        logger.info(f"Splitting dataset with test_size={test_size}")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Use the processed message if available, otherwise use the original message
        if 'processed_message' in df.columns:
            X = df['processed_message']
        else:
            X = df['message']

        y = df['label']

        # Split the data
        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

        # Create DataFrames for train and test sets
        train_df = pd.DataFrame({'message': X_train, 'label': y_train})
        test_df = pd.DataFrame({'message': X_test, 'label': y_test})

        # Save the splits to CSV
        train_df.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)

        logger.info(f"Data split completed. Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        logger.info(f"Split datasets saved to {output_dir}")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.error(f"Error splitting dataset: {str(e)}")
        return None, None, None, None


def main():
    """Main function to run the data preparation script"""
    parser = argparse.ArgumentParser(description="SMS Spam Detection - Data Preparation Script")

    parser.add_argument("--download", action="store_true", help="Download the SMS Spam Collection dataset")
    parser.add_argument("--input", "-i", help="Path to input CSV file")
    parser.add_argument("--output", "-o", help="Path to save the cleaned dataset", default="cleaned_spam.csv")
    parser.add_argument("--no-stemming", action="store_true", help="Disable stemming during preprocessing")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Proportion of data to use for testing (default: 0.2)")
    parser.add_argument("--no-stratify", action="store_true",
                        help="Disable stratified sampling when splitting the dataset")
    parser.add_argument("--exploration", action="store_true", help="Perform exploratory data analysis")
    parser.add_argument("--exploration-dir", default="data_exploration",
                        help="Directory to save exploration visualizations")
    parser.add_argument("--split", action="store_true", help="Split the dataset into training and testing sets")
    parser.add_argument("--split-dir", default="split_data", help="Directory to save split datasets")

    args = parser.parse_args()

    # Check if NLTK resources are downloaded
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK resources...")
        nltk.download('stopwords')

    # Determine the input file path
    input_path = args.input if args.input else "spam.csv"

    # Download the dataset if requested
    if args.download:
        if not download_dataset(input_path):
            print("Failed to download the dataset. Exiting.")
            sys.exit(1)

    # Load the dataset
    df = load_dataset(input_path)

    if df is None:
        print(f"Failed to load the dataset from {input_path}. Exiting.")
        sys.exit(1)

    # Clean the dataset
    stemming = not args.no_stemming
    cleaned_df = clean_dataset(df, stemming)

    # Save the cleaned dataset
    cleaned_df.to_csv(args.output, index=False)
    print(f"Cleaned dataset saved to {args.output}")

    # Perform exploratory data analysis if requested
    if args.exploration:
        stats = explore_dataset(cleaned_df, args.exploration_dir)

        # Save statistics to a text file
        with open(os.path.join(args.exploration_dir, 'statistics.txt'), 'w') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")

    # Split the dataset if requested
    if args.split:
        stratify = not args.no_stratify
        split_dataset(cleaned_df, args.test_size, 42, stratify, args.split_dir)


if __name__ == "__main__":
    main()