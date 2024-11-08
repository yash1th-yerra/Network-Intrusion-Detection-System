# Intrusion Detection System - Deep Learning Models

This project demonstrates a machine learning-based Intrusion Detection System (IDS) that leverages binary and multi-class classification techniques. The aim is to identify and classify network intrusions effectively. By using various Machine and Deep learning models—including SVM, Decision Tree, and advanced LSTM networks—this IDS attempts to distinguish between normal and malicious network activity.




# Project Overview
Intrusion Detection Systems are essential in cybersecurity, helping organizations detect and respond to suspicious activities on their networks. This project develops and evaluates several Deep and Machine learning models to classify network intrusions into either:

-   **Binary classification**: Distinguishing between "normal" and "intrusive" activity.
-   **Multi-class classification**: Categorizing intrusions into multiple specific types.

This project follows a systematic approach:

1.  **Data Preparation**: The dataset is pre-processed for binary and multi-class classification.
2.  **Model Training and Evaluation**: Models are trained using both binary and multi-class datasets. Each model's performance is evaluated and saved.
3.  **Visualization**: Model performance metrics are visualized to enable easy comparison.

## Project Structure

-   **`bin_data.csv`** - Pre-processed dataset for binary classification.
-   **`multi_data.csv`** - Pre-processed dataset for multi-class classification.
-   **`le1_classes.npy`**, **`le2_classes.npy`** - Label encoding for binary and multi-class labels.
-   **Trained Models** - Saved models in `.pkl` (for SVM and Decision Tree) or `.h5` (for deep learning models).
-   **Classification Reports** - Classification reports saved for each model, detailing metrics like precision, recall, and F1-score.

## Models and Methodology

The project employs a mix of classical machine learning and deep learning models, chosen for their ability to capture patterns in complex network intrusion data.

1.  **Binary Classification Models**
    
    -   **Linear Support Vector Machine (SVM)**:
        -   Uses a linear kernel to classify network traffic as normal or intrusive.
        -   Model saved as `lsvm_binary.pkl`.
    -   **Decision Tree**:
        -   A tree-based model providing interpretable rules for binary classification.
        -   Model saved as `dt_binary.pkl`.
    -   **LSTM (Long Short-Term Memory)**:
        -   A recurrent neural network designed to capture temporal dependencies in binary data.
        -   Model saved as `lstm_binary.h5`.
2.  **Multi-Class Classification Models**
    
    -   **Quadratic Support Vector Machine (SVM)**:
        -   Uses a polynomial kernel to identify specific types of intrusions.
        -   Model saved as `qsvm_binary.pkl`.
    -   **Decision Tree**:
        -   Tree-based model adapted for multi-class classification.
        -   Model saved as `dt_multi.pkl`.
    -   **LSTM Model**:
        -   A recurrent neural network that models sequences to distinguish between multiple intrusion types.
        -   Model saved as `lstm_multi.h5`.
    -   **BiLSTM with Multi-Head Attention**:
        -   A more advanced model combining bidirectional LSTM with attention mechanisms for enhanced multi-class classification.
        -   Model saved as `bilstm_attention_model.h5`.

## Evaluation and Performance Metrics

Each model’s performance is evaluated using:

-   **Accuracy**: Proportion of correct predictions.
-   **Precision**: Correctness of positive predictions.
-   **Recall**: Ability to detect all relevant instances.
-   **F1 Score**: Balance between precision and recall.

These metrics are summarized in classification report files for each model and saved in `classification_report_*.txt` files.


## Model Performance Visualization

-   **Accuracy Comparison**: Bar plot comparing model accuracies.
-   **Training Metrics**: For deep learning models, additional plots visualize training and validation accuracy and loss across epochs.


## 
## Results and Insights

This project provides a comparative view of several machine learning techniques for network intrusion detection. The models provide a baseline for binary and multi-class classifications, showing how classical machine learning and neural network approaches perform on the tas


