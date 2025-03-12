# SMS Spam Detection Dashboard
# Cole Detrick - WGU Capstone Project
# Completion Date: 04/10/2025

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import time
import sys
import logging

# Add parent directory to path to import the SMS Spam Detector class
sys.path.append('..')
from sms_spam_detector import SMSSpamDetector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dashboard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SMS_Spam_Dashboard")

# Page Configuration
st.set_page_config(
    page_title="SMS Spam Detection Dashboard",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for the dashboard
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .st-emotion-cache-1siy2j7 {
        background-color: #e0e0e0;
    }
    .metric-container {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 0 5px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
    }
    .stTextInput>div>div>input {
        min-height: 100px;
    }
    .wordcloud {
        text-align: center;
    }
    .title {
        text-align: center;
        color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Initialize session state for batch predictions
if 'batch_predictions' not in st.session_state:
    st.session_state.batch_predictions = None


class SpamDashboard:
    """Dashboard for SMS Spam Detection"""

    def __init__(self):
        """Initialize the Dashboard"""
        self.detector = None
        self.model_loaded = False

        # Download NLTK resources if not already downloaded
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            with st.spinner('Downloading NLTK resources...'):
                nltk.download('stopwords')

        # Sidebar Navigation
        st.sidebar.title("SMS Spam Detector")

        # Add navigation options
        self.pages = {
            "Home": self.home_page,
            "Single Message Analysis": self.single_message_page,
            "Batch Processing": self.batch_processing_page,
            "Model Performance": self.model_performance_page,
            "Data Exploration": self.data_exploration_page,
            "System Monitoring": self.system_monitoring_page,
        }

        # Create sidebar navigation
        self.selected_page = st.sidebar.radio("Navigation", list(self.pages.keys()))

        # Load the detector
        self.load_detector()

        # Display the selected page
        self.pages[self.selected_page]()

    def load_detector(self):
        """Load the SMS Spam Detector model"""
        try:
            # Check if model file exists
            if os.path.exists('sms_spam_model.pkl'):
                with st.spinner('Loading model...'):
                    self.detector = SMSSpamDetector()
                    if self.detector.load_model('sms_spam_model.pkl'):
                        self.model_loaded = True
                        st.sidebar.success("✅ Model loaded successfully")
                    else:
                        st.sidebar.error("❌ Failed to load model")
            else:
                st.sidebar.warning("⚠️ Model file not found. Please train the model first.")
                self.detector = SMSSpamDetector()
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {str(e)}")
            logger.error(f"Error loading model: {str(e)}")

    def home_page(self):
        """Home page with information about the spam detector"""
        st.title("SMS Spam Detection System")
        st.markdown("### Welcome to the SMS Spam Detection Dashboard!")

        # About the project
        st.markdown("""
        This dashboard provides an interface to interact with an SMS spam detection model built 
        using machine learning. The model uses a Naive Bayes classifier with TF-IDF vectorization 
        to identify spam text messages.

        ### Features:

        - **Single Message Analysis**: Test individual SMS messages to check if they're spam
        - **Batch Processing**: Analyze multiple messages at once by uploading a CSV file
        - **Model Performance**: View detailed metrics about the model's performance
        - **Data Exploration**: Explore the dataset used to train the model
        - **System Monitoring**: Monitor the system's performance over time

        ### How It Works:

        1. SMS messages are preprocessed by removing punctuation, converting to lowercase, etc.
        2. The model extracts features using TF-IDF (Term Frequency-Inverse Document Frequency)
        3. A Naive Bayes classifier predicts whether the message is spam or not
        4. The prediction is displayed along with a confidence score

        ### Dataset:

        The model was trained on the SMS Spam Collection Dataset, which contains 5,574 SMS 
        messages that are labeled as either "ham" (legitimate) or "spam".
        """)

        # Display model status
        st.markdown("### Model Status")
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Model Status",
                value="Loaded" if self.model_loaded else "Not Loaded"
            )

        with col2:
            if self.model_loaded:
                st.metric(
                    label="Model Type",
                    value="Naive Bayes + TF-IDF"
                )

        # Quick action buttons
        st.markdown("### Quick Actions")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Analyze a Message"):
                st.session_state.selected_page = "Single Message Analysis"
                st.rerun()

        with col2:
            if st.button("Process Multiple Messages"):
                st.session_state.selected_page = "Batch Processing"
                st.rerun()

        with col3:
            if st.button("View Model Performance"):
                st.session_state.selected_page = "Model Performance"
                st.rerun()

    def single_message_page(self):
        """Page for analyzing single SMS messages"""
        st.title("Single Message Analysis")
        st.markdown("Enter an SMS message to check if it's spam or not.")

        # Check if model is loaded
        if not self.model_loaded:
            st.error("❌ Model not loaded. Please check the sidebar for details.")
            return

        # Get user input
        message = st.text_area("Enter SMS message:", height=100)

        col1, col2 = st.columns([1, 3])

        with col1:
            analyze_button = st.button("Analyze", type="primary")

        with col2:
            clear_button = st.button("Clear")

        # Clear the input if requested
        if clear_button:
            message = ""
            st.rerun()

        # Analyze the message if requested
        if analyze_button and message:
            with st.spinner('Analyzing message...'):
                # Make prediction
                start_time = time.time()
                result, probability = self.detector.predict(message)
                end_time = time.time()

                # Log the prediction
                logger.info(f"Message: {message[:50]}...")
                logger.info(f"Prediction: {result} (Probability: {probability:.4f})")

                # Create a dictionary with prediction details
                prediction = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'message': message,
                    'prediction': result,
                    'probability': probability,
                    'processing_time': end_time - start_time
                }

                # Add to history
                st.session_state.history.append(prediction)

            # Display the result
            st.markdown("### Result")

            col1, col2 = st.columns(2)

            with col1:
                if result == "SPAM":
                    st.error(f"📵 SPAM DETECTED (Confidence: {probability:.2%})")
                else:
                    st.success(f"✅ NOT SPAM (Confidence: {probability:.2%})")

            with col2:
                st.info(f"⏱️ Processing Time: {(end_time - start_time) * 1000:.2f} ms")

            # Display features that contributed to the prediction
            st.markdown("### Message Analysis")

            # Preprocess the message to show tokens
            processed_message = self.detector.preprocess_text(message)
            tokens = processed_message.split()

            # Get important spam and ham features
            spam_features, ham_features = self.detector.get_important_features(n=100)
            spam_words = [word for word, _ in spam_features]
            ham_words = [word for word, _ in ham_features]

            # Highlight tokens based on their importance
            highlighted_tokens = []
            for token in tokens:
                if token in spam_words:
                    highlighted_tokens.append(f"<span style='color:red'>{token}</span>")
                elif token in ham_words:
                    highlighted_tokens.append(f"<span style='color:green'>{token}</span>")
                else:
                    highlighted_tokens.append(token)

            highlighted_text = " ".join(highlighted_tokens)

            st.markdown("#### Processed Message with Highlighted Features:")
            st.markdown(f"<p>{highlighted_text}</p>", unsafe_allow_html=True)

            st.markdown("""
            <div style='font-size: 0.8em; margin-top: 10px;'>
                <span style='color:red'>Red</span>: Words associated with spam<br>
                <span style='color:green'>Green</span>: Words associated with legitimate messages
            </div>
            """, unsafe_allow_html=True)

        # Display history
        if st.session_state.history:
            st.markdown("### Recent Analysis History")

            # Convert history to DataFrame
            history_df = pd.DataFrame(st.session_state.history)

            # Show only the last 5 entries
            recent_history = history_df.tail(5).copy()

            # Truncate long messages
            recent_history['message'] = recent_history['message'].apply(
                lambda x: x[:50] + "..." if len(x) > 50 else x
            )

            # Format probability as percentage
            recent_history['probability'] = recent_history['probability'].apply(
                lambda x: f"{x:.2%}"
            )

            # Format processing time in milliseconds
            recent_history['processing_time'] = recent_history['processing_time'].apply(
                lambda x: f"{x * 1000:.2f} ms"
            )

            # Display as table
            st.dataframe(
                recent_history[['timestamp', 'message', 'prediction', 'probability', 'processing_time']],
                use_container_width=True
            )

            # Option to clear history
            if st.button("Clear History"):
                st.session_state.history = []
                st.rerun()

    def batch_processing_page(self):
        """Page for processing multiple SMS messages"""
        st.title("Batch Processing")
        st.markdown("Upload a CSV file with SMS messages to analyze in bulk.")

        # Check if model is loaded
        if not self.model_loaded:
            st.error("❌ Model not loaded. Please check the sidebar for details.")
            return

        # File upload
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)

                # Display the first few rows
                st.markdown("### CSV Preview")
                st.dataframe(df.head(), use_container_width=True)

                # Select column containing messages
                st.markdown("### Select Message Column")
                message_column = st.selectbox(
                    "Select the column containing SMS messages:",
                    df.columns.tolist()
                )

                if st.button("Process Messages", type="primary"):
                    with st.spinner('Processing messages...'):
                        # Get the messages
                        messages = df[message_column].tolist()

                        # Make batch predictions
                        start_time = time.time()
                        batch_results = self.detector.batch_predict(messages)
                        end_time = time.time()

                        # Store the results in session state
                        st.session_state.batch_predictions = batch_results

                        # Log the batch processing
                        logger.info(f"Batch processed {len(messages)} messages")
                        logger.info(f"Processing time: {end_time - start_time:.2f} seconds")

                    st.success(f"✅ Processed {len(messages)} messages in {end_time - start_time:.2f} seconds")

                # Display results if available
                if st.session_state.batch_predictions is not None:
                    st.markdown("### Results")

                    # Get the results
                    results = st.session_state.batch_predictions

                    # Display as dataframe
                    st.dataframe(results, use_container_width=True)

                    # Calculate summary statistics
                    spam_count = sum(results['prediction'] == 1)
                    ham_count = sum(results['prediction'] == 0)
                    spam_percentage = spam_count / len(results) * 100

                    # Display summary
                    st.markdown("### Summary")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            label="Total Messages",
                            value=len(results)
                        )

                    with col2:
                        st.metric(
                            label="Spam Messages",
                            value=spam_count,
                            delta=f"{spam_percentage:.1f}%"
                        )

                    with col3:
                        st.metric(
                            label="Ham Messages",
                            value=ham_count,
                            delta=f"{100 - spam_percentage:.1f}%"
                        )

                    # Create a pie chart
                    fig = px.pie(
                        names=['Ham', 'Spam'],
                        values=[ham_count, spam_count],
                        title="Distribution of Spam vs. Ham",
                        color_discrete_sequence=['#4CAF50', '#F44336'],
                        hole=0.4
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Option to download results
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="spam_detection_results.csv",
                        mime="text/csv",
                    )

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                logger.error(f"Error processing file: {str(e)}")

        # Sample CSV template
        st.markdown("### Sample CSV Template")
        st.markdown("""
        If you don't have a CSV file ready, you can download this template and fill it with your messages:
        """)

        # Create a sample DataFrame
        sample_df = pd.DataFrame({
            'id': [1, 2, 3],
            'message': [
                "Hello, how are you doing today?",
                "URGENT: Your account has been locked. Call now to verify!",
                "Meeting rescheduled to 3pm tomorrow. Please confirm."
            ]
        })

        # Display the sample
        st.dataframe(sample_df, use_container_width=True)

        # Create a downloadable CSV
        csv = sample_df.to_csv(index=False)
        st.download_button(
            label="Download Template",
            data=csv,
            file_name="sms_template.csv",
            mime="text/csv",
        )

    def model_performance_page(self):
        """Page for displaying model performance metrics"""
        st.title("Model Performance")
        st.markdown("View detailed metrics about the model's performance.")

        # Check if model is loaded
        if not self.model_loaded:
            st.error("❌ Model not loaded. Please check the sidebar for details.")
            return

        # Option to run evaluation
        if st.button("Run Model Evaluation", type="primary"):
            with st.spinner('Evaluating model performance...'):
                # Check if test data is available
                if self.detector.X_test is None or self.detector.y_test is None:
                    # Load data and split if not available
                    try:
                        self.detector.load_data("spam.csv")
                        self.detector.preprocess_data()
                        self.detector.split_data()
                    except Exception as e:
                        st.error(f"❌ Error loading test data: {str(e)}")
                        logger.error(f"Error loading test data: {str(e)}")
                        return

                # Evaluate the model
                metrics = self.detector.evaluate_model()

                # Generate visualizations
                self.detector.visualize_evaluation()

                st.success("✅ Model evaluation completed")

        # Display metrics if available
        if hasattr(self.detector, 'performance_metrics') and self.detector.performance_metrics:
            metrics = self.detector.performance_metrics

            # Performance Metrics
            st.markdown("### Performance Metrics")

            # Create metrics in a row
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    label="Accuracy",
                    value=f"{metrics['accuracy']:.4f}",
                    delta=f"{(metrics['accuracy'] - 0.5) * 200:.1f}% above random"
                )

            with col2:
                st.metric(
                    label="Precision",
                    value=f"{metrics['precision']:.4f}"
                )

            with col3:
                st.metric(
                    label="Recall (Sensitivity)",
                    value=f"{metrics['recall']:.4f}"
                )

            with col4:
                st.metric(
                    label="F1 Score",
                    value=f"{metrics['f1_score']:.4f}"
                )

            # ROC Curve
            st.markdown("### ROC Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=metrics['fpr'],
                y=metrics['tpr'],
                mode='lines',
                name=f'ROC Curve (AUC = {metrics["auc"]:.4f})',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                title='Receiver Operating Characteristic (ROC) Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                legend=dict(x=0.01, y=0.99),
                width=700,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # Confusion Matrix
            st.markdown("### Confusion Matrix")
            cm = metrics['confusion_matrix']

            # Calculate derived values
            tn, fp, fn, tp = cm.ravel()
            total = tn + fp + fn + tp

            # Create a heatmap
            fig = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Ham (0)', 'Spam (1)'],
                y=['Ham (0)', 'Spam (1)'],
                color_continuous_scale='Blues'
            )

            fig.update_layout(
                title='Confusion Matrix',
                width=600,
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Explanation of confusion matrix
            st.markdown("""
            ### Confusion Matrix Explained

            - **True Negative (TN)**: {tn} - Correctly classified as Ham
            - **False Positive (FP)**: {fp} - Incorrectly classified as Spam (Type I error)
            - **False Negative (FN)**: {fn} - Incorrectly classified as Ham (Type II error)
            - **True Positive (TP)**: {tp} - Correctly classified as Spam

            - **Ham Detection Rate**: {tn_rate:.2%} of actual Ham messages were correctly classified
            - **Spam Detection Rate**: {tp_rate:.2%} of actual Spam messages were correctly classified
            """.format(
                tn=tn, fp=fp, fn=fn, tp=tp,
                tn_rate=tn / (tn + fp),
                tp_rate=tp / (tp + fn)
            ))

            # Feature Importance
            st.markdown("### Feature Importance")

            spam_features, ham_features = self.detector.get_important_features(n=15)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Top 15 Spam Indicators")

                # Create a DataFrame for Spam features
                spam_df = pd.DataFrame(spam_features, columns=['Feature', 'Importance'])

                # Plot
                fig = px.bar(
                    spam_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Words Most Indicative of Spam',
                    color='Importance',
                    color_continuous_scale='Reds'
                )

                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### Top 15 Ham Indicators")

                # Create a DataFrame for Ham features
                ham_df = pd.DataFrame(ham_features, columns=['Feature', 'Importance'])

                # Plot
                fig = px.bar(
                    ham_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Words Most Indicative of Ham',
                    color='Importance',
                    color_continuous_scale='Greens'
                )

                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

    def data_exploration_page(self):
        """Page for exploring the dataset"""
        st.title("Data Exploration")
        st.markdown("Explore the dataset used to train the model.")

        # Load the dataset if needed
        if self.detector.df is None:
            try:
                with st.spinner('Loading dataset...'):
                    self.detector.load_data("spam.csv")
                    self.detector.explore_data()  # This calculates message lengths
                st.success("✅ Dataset loaded successfully")
            except Exception as e:
                st.error(f"❌ Error loading dataset: {str(e)}")
                logger.error(f"Error loading dataset: {str(e)}")
                return

        # Dataset overview
        st.markdown("### Dataset Overview")

        # Get basic stats
        total_messages = len(self.detector.df)
        spam_count = sum(self.detector.df['label'] == 1)
        ham_count = sum(self.detector.df['label'] == 0)
        spam_percentage = (spam_count / total_messages) * 100
        ham_percentage = 100 - spam_percentage

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Total Messages",
                value=total_messages
            )

        with col2:
            st.metric(
                label="Spam Messages",
                value=spam_count,
                delta=f"{spam_percentage:.1f}% of total"
            )

        with col3:
            st.metric(
                label="Ham Messages",
                value=ham_count,
                delta=f"{ham_percentage:.1f}% of total"
            )

        # Data Distribution
        st.markdown("### Data Distribution")

        # Create pie chart of spam vs ham
        fig = px.pie(
            names=['Ham', 'Spam'],
            values=[ham_count, spam_count],
            title="Distribution of Spam vs. Ham",
            color_discrete_sequence=['#4CAF50', '#F44336'],
            hole=0.4
        )

        st.plotly_chart(fig, use_container_width=True)

        # Message Length Distribution
        st.markdown("### Message Length Distribution")

        # Create histogram of message lengths by class
        fig = px.histogram(
            self.detector.df,
            x='message_length',
            color='label',
            nbins=50,
            marginal="box",
            labels={'message_length': 'Message Length (characters)', 'label': 'Class'},
            title="Distribution of Message Lengths",
            color_discrete_map={0: '#4CAF50', 1: '#F44336'},
            opacity=0.7
        )

        fig.update_layout(
            xaxis_title="Message Length (characters)",
            yaxis_title="Count",
            legend_title="Class",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                itemsizing="constant"
            )
        )

        # Update legend labels
        newnames = {'0': 'Ham', '1': 'Spam'}
        fig.for_each_trace(lambda t: t.update(name=newnames[t.name]))

        st.plotly_chart(fig, use_container_width=True)

        # Message Length Statistics
        avg_spam_length = self.detector.df[self.detector.df['label'] == 1]['message_length'].mean()
        avg_ham_length = self.detector.df[self.detector.df['label'] == 0]['message_length'].mean()

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Average Spam Length",
                value=f"{avg_spam_length:.1f} chars"
            )

        with col2:
            st.metric(
                label="Average Ham Length",
                value=f"{avg_ham_length:.1f} chars"
            )

        # Word Clouds
        st.markdown("### Word Clouds")

        # Create word clouds for spam and ham
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Spam Word Cloud")

            # Generate word cloud for spam
            spam_messages = ' '.join(self.detector.df[self.detector.df['label'] == 1]['message'].astype(str))
            spam_wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                max_words=100,
                colormap='Reds'
            ).generate(spam_messages)

            # Display the word cloud
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(spam_wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

        with col2:
            st.markdown("#### Ham Word Cloud")

            # Generate word cloud for ham
            ham_messages = ' '.join(self.detector.df[self.detector.df['label'] == 0]['message'].astype(str))
            ham_wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                max_words=100,
                colormap='Greens'
            ).generate(ham_messages)

            # Display the word cloud
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(ham_wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

        # Sample Messages
        st.markdown("### Sample Messages")

        # Create tabs for spam and ham examples
        tab1, tab2 = st.tabs(["Spam Examples", "Ham Examples"])

        with tab1:
            spam_samples = self.detector.df[self.detector.df['label'] == 1]['message'].sample(5).tolist()
            for i, message in enumerate(spam_samples):
                st.markdown(f"**Spam Example {i + 1}:**")
                st.markdown(f"> {message}")
                st.markdown("---")

        with tab2:
            ham_samples = self.detector.df[self.detector.df['label'] == 0]['message'].sample(5).tolist()
            for i, message in enumerate(ham_samples):
                st.markdown(f"**Ham Example {i + 1}:**")
                st.markdown(f"> {message}")
                st.markdown("---")

    def system_monitoring_page(self):
        """Page for monitoring system performance"""
        st.title("System Monitoring")
        st.markdown("Monitor the system's performance over time.")

        # Generate simulated performance data if not available
        if 'monitoring_data' not in st.session_state:
            # Create simulated data for the last 30 days
            dates = pd.date_range(end=datetime.now(), periods=30).tolist()

            # Generate random metrics with a trend
            np.random.seed(42)  # For reproducibility
            accuracy = 0.95 + np.random.normal(0, 0.01, 30)
            precision = 0.93 + np.random.normal(0, 0.015, 30)
            recall = 0.89 + np.random.normal(0, 0.02, 30)
            f1 = 0.91 + np.random.normal(0, 0.015, 30)

            # Ensure values are between 0 and 1
            for metric in [accuracy, precision, recall, f1]:
                np.clip(metric, 0, 1, out=metric)

            # Create a DataFrame
            monitoring_data = pd.DataFrame({
                'date': dates,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'requests': np.random.randint(100, 1000, 30),
                'avg_response_time': 50 + 20 * np.random.random(30),
                'spam_ratio': 0.2 + 0.05 * np.random.random(30)
            })

            st.session_state.monitoring_data = monitoring_data

        # Get the data
        monitoring_data = st.session_state.monitoring_data

        # Time range selector
        st.markdown("### Time Range")
        days = st.slider("Select time range (days):", 1, 30, 7)

        # Filter data based on selected range
        filtered_data = monitoring_data.tail(days).copy()

        # Convert date to string for better display
        filtered_data['date_str'] = filtered_data['date'].dt.strftime('%Y-%m-%d')

        # Key Performance Indicators
        st.markdown("### Key Performance Indicators")

        # Calculate current values and changes
        current_accuracy = filtered_data['accuracy'].iloc[-1]
        current_precision = filtered_data['precision'].iloc[-1]
        current_recall = filtered_data['recall'].iloc[-1]
        current_f1 = filtered_data['f1_score'].iloc[-1]

        accuracy_change = current_accuracy - filtered_data['accuracy'].iloc[0]
        precision_change = current_precision - filtered_data['precision'].iloc[0]
        recall_change = current_recall - filtered_data['recall'].iloc[0]
        f1_change = current_f1 - filtered_data['f1_score'].iloc[0]

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Accuracy",
                value=f"{current_accuracy:.4f}",
                delta=f"{accuracy_change:.4f}"
            )

        with col2:
            st.metric(
                label="Precision",
                value=f"{current_precision:.4f}",
                delta=f"{precision_change:.4f}"
            )

        with col3:
            st.metric(
                label="Recall",
                value=f"{current_recall:.4f}",
                delta=f"{recall_change:.4f}"
            )

        with col4:
            st.metric(
                label="F1 Score",
                value=f"{current_f1:.4f}",
                delta=f"{f1_change:.4f}"
            )

        # Performance Metrics Over Time
        st.markdown("### Performance Metrics Over Time")

        # Create line chart for performance metrics
        fig = px.line(
            filtered_data,
            x='date_str',
            y=['accuracy', 'precision', 'recall', 'f1_score'],
            labels={'date_str': 'Date', 'value': 'Score', 'variable': 'Metric'},
            title="Performance Metrics Trend",
            markers=True
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Score",
            legend_title="Metric",
            yaxis_range=[0.8, 1.0],
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

        # System Usage Metrics
        st.markdown("### System Usage")

        col1, col2 = st.columns(2)

        with col1:
            # Requests per day
            fig = px.bar(
                filtered_data,
                x='date_str',
                y='requests',
                labels={'date_str': 'Date', 'requests': 'Number of Requests'},
                title="Daily Requests",
                color='requests',
                color_continuous_scale='Viridis'
            )

            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Requests"
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Response time
            fig = px.line(
                filtered_data,
                x='date_str',
                y='avg_response_time',
                labels={'date_str': 'Date', 'avg_response_time': 'Response Time (ms)'},
                title="Average Response Time",
                markers=True
            )

            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Response Time (ms)"
            )

            st.plotly_chart(fig, use_container_width=True)

        # Spam Ratio Over Time
        st.markdown("### Spam Detection Ratio")

        # Create line chart for spam ratio
        fig = px.line(
            filtered_data,
            x='date_str',
            y='spam_ratio',
            labels={'date_str': 'Date', 'spam_ratio': 'Spam Ratio'},
            title="Spam Ratio Trend",
            markers=True
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Spam Ratio",
            hovermode="x"
        )

        # Add a moving average
        fig.add_trace(
            go.Scatter(
                x=filtered_data['date_str'],
                y=filtered_data['spam_ratio'].rolling(3).mean(),
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='3-Day Moving Average'
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # System Health
        st.markdown("### System Health")

        # Create simulated system health metrics
        np.random.seed(42)
        cpu_usage = 20 + 15 * np.random.random(days)
        memory_usage = 30 + 20 * np.random.random(days)
        disk_usage = 50 + 10 * np.random.random(days)

        system_health = pd.DataFrame({
            'date_str': filtered_data['date_str'],
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'disk_usage': disk_usage
        })

        # Display current health metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="CPU Usage",
                value=f"{cpu_usage[-1]:.1f}%"
            )

        with col2:
            st.metric(
                label="Memory Usage",
                value=f"{memory_usage[-1]:.1f}%"
            )

        with col3:
            st.metric(
                label="Disk Usage",
                value=f"{disk_usage[-1]:.1f}%"
            )

        # Create line chart for system health
        fig = px.line(
            system_health,
            x='date_str',
            y=['cpu_usage', 'memory_usage', 'disk_usage'],
            labels={'date_str': 'Date', 'value': 'Usage (%)', 'variable': 'Resource'},
            title="System Resource Usage",
            markers=True
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Usage (%)",
            legend_title="Resource",
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Maintenance Log
        st.markdown("### Maintenance Log")

        # Create simulated maintenance log
        maintenance_log = [
            {"date": "2025-03-09", "action": "System restart", "details": "Scheduled maintenance"},
            {"date": "2025-03-05", "action": "Model update", "details": "Improved feature extraction"},
            {"date": "2025-02-28", "action": "Database backup", "details": "Regular weekly backup"},
            {"date": "2025-02-25", "action": "Bug fix", "details": "Fixed issue with long messages processing"},
            {"date": "2025-02-20", "action": "Performance tuning", "details": "Optimized TF-IDF vectorization"}
        ]

        # Display maintenance log
        maintenance_df = pd.DataFrame(maintenance_log)
        st.dataframe(maintenance_df, use_container_width=True)


# Main entry point
if __name__ == "__main__":
    dashboard = SpamDashboard()