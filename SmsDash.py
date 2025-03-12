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
            "Model Performance":