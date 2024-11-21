import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model


# Constants
MODEL_PATH = "models/modified_bilstm_attention_model.h5"
LABEL_ENCODER_PATH = './label_classes/le2_classes.npy'
file_path='./sample_input.csv'

# Helper Functions
def load_model(model_path):
    """Load the trained BiLSTM model and the label encoder classes."""
    model = load_model(model_path, compile=False)

    return model


def preprocess_sample_input(file_path_input):
    """
    Load and preprocess the sample input data.

    Parameters:
    - file_path: str, path to the CSV file containing sample input data.

    Returns:
    - X_sample: numpy array, preprocessed input data.
    """
    # Load the sample CSV data
    sample_data = pd.read_csv(file_path_input)
    # Preprocess the data (convert to numpy array and reshape)
    X_sample = sample_data.values  # Convert to numpy array
    X_sample = X_sample.reshape((X_sample.shape[0], 1, X_sample.shape[1]))  # Reshape to (samples, time_steps, features)
    X_sample = np.array(X_sample, dtype='float32')  # Ensure the data type is float32

    return X_sample


# Function to make predictions and convert indices to class labels
def predict_classes(model, X_sample, le2_classes):
    """
    Make predictions using the trained model and convert class indices to labels.

    Parameters:
    - model: Keras model, the trained model to use for predictions.
    - X_sample: numpy array, the preprocessed input data.
    - le2_classes: numpy array, the class labels array (from le2_classes.npy).

    Returns:
    - predicted_class_names: list of predicted class names corresponding to the input data.
    """
    # Make predictions
    predictions = model.predict(X_sample)

    # Get the predicted class indices
    predicted_class_indices = np.argmax(predictions, axis=1)

    # Convert class indices to class names
    predicted_class_names = le2_classes[predicted_class_indices]

    return predicted_class_names


# Example usage:
if __name__ == "__main__":
    # Load the label encoder classes from le2_classes.npy
    le2_classes = np.load('./label_classes/le2_classes.npy', allow_pickle=True)

    # Load and preprocess the sample input data
    X_sample = preprocess_sample_input(file_path)

    # Load the trained model
    model = load_model(MODEL_PATH)

    # Make predictions and get class names
    predicted_class_names = predict_classes(model, X_sample, le2_classes)

    # Print the predicted class names
    print("Predicted class names:", predicted_class_names)