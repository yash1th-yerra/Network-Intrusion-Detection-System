import shap
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import argparse

# Constants
MODEL_PATH = "models/modified_bilstm_attention_model.h5"
DEFAULT_INPUT = "sample_input.csv"

def load_trained_model(model_path):
    """Load the trained Keras model."""
    # compile=False avoids searching for loss functions/optimizers training configuration
    return load_model(model_path, compile=False)

def preprocess_data(file_path):
    """Load and preprocess data similarly to main.py."""
    df = pd.read_csv(file_path)
    X = df.values
    # Reshape to (samples, time_steps, features) for BiLSTM
    X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
    return X_reshaped, df.columns

def explain_predictions(model_path, input_file, output_plot="shap_summary.png"):
    print(f"Loading model from {model_path}...")
    model = load_trained_model(model_path)
    
    print(f"Loading data from {input_file}...")
    X_sample, feature_names = preprocess_data(input_file)
    
    # SHAP DeepExplainer or GradientExplainer is preferred for Deep Learning
    # But since inputs are 3D (samples, 1, features), we need to be careful.
    # We use a background set (e.g., first 100 samples) to integrate over.
    background = X_sample[:100] 
    
    print("Initializing SHAP explainer...")
    # diverse background is better, but using what we have for now
    explainer = shap.GradientExplainer(model, background)
    
    print("Calculating SHAP values...")
    shap_values = explainer.shap_values(X_sample)
    
    # shap_values is a list of arrays (one for each class)
    # or a single array. 
    # For multi-class, it's usually a list.
    
    # To visualize, we might need to flatten requirements if we use summary_plot
    # X_sample is (N, 1, F), we want (N, F) for the plot
    X_2d = X_sample.reshape(X_sample.shape[0], X_sample.shape[2])
    
    # If shap_values is a list (multi-class), we pick the class of interest or aggregate
    # For now, let's plot the summary for the first class (or all if feasible)
    
    print(f"Generating plot to {output_plot}...")
    plt.figure()
    
    # Handling list of shap values (multi-class legacy behavior)
    if isinstance(shap_values, list):
        # Flatten the time dimension for the first class's explanation
        # shap_values[0] shape: (samples, time_steps, features)
        sv_class0_2d = shap_values[0].reshape(shap_values[0].shape[0], shap_values[0].shape[2])
        shap.summary_plot(sv_class0_2d, X_2d, feature_names=feature_names, show=False)
    else:
        # If it's single output (binary)
        sv_2d = shap_values.reshape(shap_values.shape[0], shap_values.shape[2])
        shap.summary_plot(sv_2d, X_2d, feature_names=feature_names, show=False)
        
    plt.savefig(output_plot, bbox_inches='tight')
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SHAP explanations for NIDS")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to input CSV")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to model .h5 file")
    parser.add_argument("--output", default="shap_summary.png", help="Output path for plot")
    
    args = parser.parse_args()
    
    explain_predictions(args.model, args.input, args.output)
