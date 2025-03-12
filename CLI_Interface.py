#!/usr/bin/env python3
# SMS Spam Detection CLI
# Cole Detrick - WGU Capstone Project
# Completion Date: 04/10/2025

import argparse
import pandas as pd
import os
import sys
import time
import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
from rich import print as rprint

# Add parent directory to path to import the SMS Spam Detector class
sys.path.append('.')
from sms_spam_detector import SMSSpamDetector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cli_detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SMS_Spam_CLI")

# Create a rich console for pretty output
console = Console()


def print_header():
    """Print a fancy header for the CLI"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold blue]SMS Spam Detection System[/bold blue]\n"
        "[italic]Cole Detrick - WGU Capstone Project[/italic]",
        border_style="blue"
    ))
    console.print("\n")


def analyze_message(detector, message):
    """Analyze a single message"""
    try:
        # Log the request
        logger.info(f"Analyzing message: {message[:50]}...")

        # Make prediction
        start_time = time.time()
        result, probability = detector.predict(message)
        end_time = time.time()

        # Create a table with the results
        table = Table(show_header=True, header_style="bold")
        table.add_column("Field", style="dim")
        table.add_column("Value")

        # Add rows
        table.add_row("Message", message[:100] + "..." if len(message) > 100 else message)
        table.add_row("Prediction", f"[bold {'red' if result == 'SPAM' else 'green'}]{result}[/bold]")
        table.add_row("Confidence", f"{probability:.2%}")
        table.add_row("Processing Time", f"{(end_time - start_time) * 1000:.2f} ms")

        # Print the table
        console.print(table)

        # Log the result
        logger.info(f"Result: {result} (Confidence: {probability:.2%})")

        return result, probability

    except Exception as e:
        logger.error(f"Error analyzing message: {str(e)}")
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        return None, None


def analyze_file(detector, file_path, message_column):
    """Analyze messages from a CSV file"""
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            console.print(f"[bold red]Error:[/bold red] File not found: {file_path}")
            return

        # Log the request
        logger.info(f"Analyzing file: {file_path}, column: {message_column}")

        # Load the CSV file
        console.print(f"Loading file: {file_path}")
        df = pd.read_csv(file_path)

        # Check if the column exists
        if message_column not in df.columns:
            console.print(f"[bold red]Error:[/bold red] Column '{message_column}' not found in the file.")
            console.print(f"Available columns: {', '.join(df.columns)}")
            return

        # Extract messages
        messages = df[message_column].tolist()
        total_messages = len(messages)

        console.print(f"Found {total_messages} messages to analyze.")

        # Ask for confirmation if there are many messages
        if total_messages > 100:
            response = input(f"Processing {total_messages} messages may take some time. Continue? (y/n): ")
            if response.lower() != 'y':
                console.print("Operation cancelled.")
                return

        # Process the messages
        start_time = time.time()

        with Progress() as progress:
            task = progress.add_task("[cyan]Processing messages...", total=total_messages)

            # Process in batches for better performance
            batch_size = 100
            results = []

            for i in range(0, total_messages, batch_size):
                batch = messages[i:i + batch_size]
                batch_results = detector.batch_predict(batch)
                results.append(batch_results)
                progress.update(task, advance=len(batch))

            # Combine results
            all_results = pd.concat(results, ignore_index=True)

        end_time = time.time()
        total_time = end_time - start_time

        # Calculate statistics
        spam_count = sum(all_results['prediction'] == 1)
        ham_count = sum(all_results['prediction'] == 0)
        spam_percentage = spam_count / total_messages * 100

        # Create a results table
        table = Table(show_header=True, header_style="bold")
        table.add_column("Metric", style="dim")
        table.add_column("Value")

        # Add rows
        table.add_row("Total Messages", str(total_messages))
        table.add_row("Spam Messages", f"{spam_count} ({spam_percentage:.2f}%)")
        table.add_row("Ham Messages", f"{ham_count} ({100 - spam_percentage:.2f}%)")
        table.add_row("Processing Time", f"{total_time:.2f} seconds")
        table.add_row("Average Time per Message", f"{(total_time / total_messages) * 1000:.2f} ms")

        # Print the table
        console.print("\n[bold]Analysis Results:[/bold]")
        console.print(table)

        # Ask if the user wants to save the results
        response = input("Save results to a CSV file? (y/n): ")
        if response.lower() == 'y':
            output_file = input("Enter output file name (default: spam_results.csv): ") or "spam_results.csv"

            # Add results to the original DataFrame
            df['spam_prediction'] = all_results['prediction']
            df['spam_probability'] = all_results['probability']
            df['result'] = all_results['result']

            # Save to CSV
            df.to_csv(output_file, index=False)
            console.print(f"Results saved to [bold]{output_file}[/bold]")

        # Log the completion
        logger.info(f"File analysis completed: {spam_count} spam, {ham_count} ham")

        return all_results

    except Exception as e:
        logger.error(f"Error analyzing file: {str(e)}")
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        return None


def display_model_info(detector):
    """Display information about the loaded model"""
    if not detector.model:
        console.print("[bold red]Error:[/bold red] No model loaded.")
        return

    # Extract model information
    vectorizer = detector.model.named_steps['tfidf']
    classifier = detector.model.named_steps['classifier']

    # Create a table with model information
    table = Table(show_header=True, header_style="bold")
    table.add_column("Property", style="dim")
    table.add_column("Value")

    # Add rows
    table.add_row("Model Type", "Multinomial Naive Bayes with TF-IDF")
    table.add_row("Vectorizer", f"TF-IDF (max_features={vectorizer.max_features})")
    table.add_row("Classifier", "MultinomialNB")
    table.add_row("Alpha", f"{classifier.alpha}")
    table.add_row("Number of Features", f"{len(vectorizer.get_feature_names_out())}")

    # Print the table
    console.print(table)

    # Get top features if available
    try:
        spam_features, ham_features = detector.get_important_features(n=10)

        console.print("\n[bold]Top 10 Spam Indicators:[/bold]")
        for word, score in spam_features:
            console.print(f"- {word}: {score:.4f}")

        console.print("\n[bold]Top 10 Ham Indicators:[/bold]")
        for word, score in ham_features:
            console.print(f"- {word}: {score:.4f}")

    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        console.print("[italic]Feature importance information not available.[/italic]")


def interactive_mode(detector):
    """Run the CLI in interactive mode"""
    print_header()
    console.print("[bold]Interactive Mode[/bold]")
    console.print("Type 'exit' to quit, 'info' to show model information.\n")

    while True:
        message = input("\nEnter a message to analyze (or 'exit'/'info'): ")

        if message.lower() == 'exit':
            break
        elif message.lower() == 'info':
            display_model_info(detector)
        elif message.strip():
            analyze_message(detector, message)
        else:
            console.print("[italic]Please enter a message or command.[/italic]")

    console.print("\n[bold]Goodbye![/bold]")


def main():
    """Main entry point for the CLI"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="SMS Spam Detection CLI")

    # Add arguments
    parser.add_argument("--message", "-m", help="A single message to analyze")
    parser.add_argument("--file", "-f", help="Path to a CSV file containing messages")
    parser.add_argument("--column", "-c", help="Column name in the CSV file containing messages")
    parser.add_argument("--model", help="Path to a saved model file (default: sms_spam_model.pkl)",
                        default="sms_spam_model.pkl")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--info", action="store_true", help="Display information about the loaded model")

    # Parse arguments
    args = parser.parse_args()

    # Print header
    print_header()

    # Load the detector
    console.print("Loading model...")
    detector = SMSSpamDetector()

    if os.path.exists(args.model):
        if detector.load_model(args.model):
            console.print(f"Model loaded successfully from {args.model}")
        else:
            console.print(f"[bold red]Error:[/bold red] Failed to load model from {args.model}")
            return
    else:
        console.print(f"[bold red]Error:[/bold red] Model file not found: {args.model}")
        return

    # Process based on arguments
    if args.info:
        display_model_info(detector)
    elif args.message:
        analyze_message(detector, args.message)
    elif args.file:
        if not args.column:
            console.print("[bold red]Error:[/bold red] Please specify a column name with --column")
            return
        analyze_file(detector, args.file, args.column)
    elif args.interactive:
        interactive_mode(detector)
    else:
# No specific action provided, go to interactive mode