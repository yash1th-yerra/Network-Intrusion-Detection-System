import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

# Function to save classification reports
def save_report(report, model_name):
    report_file = os.path.join('reports/new', f"classification_report_{model_name}.txt")
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Report saved: {report_file}")

def main():
    # Load label classes (assumes these files exist in the data folder)
    le1_classes = np.load('data/le1_classes.npy', allow_pickle=True)
    le2_classes = np.load('data/le2_classes.npy', allow_pickle=True)

    # Load binary data (bin_data.csv)
    bin_data = pd.read_csv('data/bin_data.csv')
    bin_data.drop(bin_data.columns[0], axis=1, inplace=True)

    # Load multi-class data (multi_data.csv)
    multi_data = pd.read_csv('data/multi_data.csv')
    multi_data.drop(multi_data.columns[0], axis=1, inplace=True)

    # Prepare input features and target labels for binary classification
    X_bin = bin_data.iloc[:, :-1].to_numpy()
    y_bin = bin_data['intrusion'].to_numpy()

    # Prepare input features and target labels for multi-class classification
    X_multi = multi_data.iloc[:, :-1].to_numpy()
    y_multi = multi_data['intrusion'].to_numpy()

    # Split data into train and test sets
    X_bin_train, X_bin_test, y_bin_train, y_bin_test = train_test_split(X_bin, y_bin, test_size=0.25, random_state=42)
    X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(X_multi, y_multi, test_size=0.25, random_state=42)

    # Ensure reports folder exists
    os.makedirs('reports', exist_ok=True)

    # **Linear SVM (Binary Classification)**

    # Load the model from the models folder
    with open('models/lsvm_binary.pkl', 'rb') as file:
        lsvm = pickle.load(file)

    # Predict and calculate accuracy
    y_bin_pred_lsvm = lsvm.predict(X_bin_test)
    bin_accuracy_lsvm = accuracy_score(y_bin_test, y_bin_pred_lsvm) * 100
    print("LSVM Binary Classification Accuracy: ", bin_accuracy_lsvm)

    # Print and save classification report
    report_lsvm = classification_report(y_bin_test, y_bin_pred_lsvm, target_names=le1_classes)
    print(report_lsvm)
    save_report(report_lsvm, "lsvm")

    # **Quadratic SVM (Binary Classification)**

    # Load the model from the models folder
    with open('models/qsvm_binary.pkl', 'rb') as file:
        qsvm = pickle.load(file)

    # Predict and calculate accuracy
    y_bin_pred_qsvm = qsvm.predict(X_bin_test)
    bin_accuracy_qsvm = accuracy_score(y_bin_test, y_bin_pred_qsvm) * 100
    print("QSVM Binary Classification Accuracy: ", bin_accuracy_qsvm)

    # Print and save classification report
    report_qsvm = classification_report(y_bin_test, y_bin_pred_qsvm, target_names=le1_classes)
    print(report_qsvm)
    save_report(report_qsvm, "qsvm")

    # **Decision Tree Classifier (Binary Classification)**

    # Load the model from the models folder
    with open('models/dt_binary.pkl', 'rb') as file:
        dt_bin = pickle.load(file)

    # Predict and calculate accuracy
    y_bin_pred_dt = dt_bin.predict(X_bin_test)
    bin_accuracy_dt = accuracy_score(y_bin_test, y_bin_pred_dt) * 100
    print("Decision Tree Binary Classification Accuracy: ", bin_accuracy_dt)

    # Print and save classification report
    report_dt = classification_report(y_bin_test, y_bin_pred_dt, target_names=le1_classes)
    print(report_dt)
    save_report(report_dt, "dt_binary")

    # **LSTM (Binary Classification)**

    # Load the LSTM model from the models folder
    lstm_bin = load_model('models/lstm_binary.h5')

    # Reshape the data to be 3-dimensional [samples, time steps, features] for LSTM
    X_bin_lstm = X_bin.reshape((X_bin.shape[0], 1, X_bin.shape[1]))
    X_bin_lstm_test = X_bin_lstm[X_bin_test.index]

    # Predict and calculate accuracy
    y_bin_pred_lstm = (lstm_bin.predict(X_bin_lstm_test) > 0.5).astype("float32").flatten()
    bin_accuracy_lstm = accuracy_score(y_bin_test, y_bin_pred_lstm) * 100
    print("LSTM Binary Classification Accuracy: ", bin_accuracy_lstm)

    # Print and save classification report
    report_lstm = classification_report(y_bin_test, y_bin_pred_lstm, target_names=le1_classes)
    print(report_lstm)
    save_report(report_lstm, "lstm_binary")

    # **Decision Tree Classifier (Multi-Class Classification)**

    # Load the model from the models folder
    with open('models/dt_multi.pkl', 'rb') as file:
        dt_multi = pickle.load(file)

    # Predict and calculate accuracy
    y_multi_pred_dt = dt_multi.predict(X_multi_test)
    multi_accuracy_dt = accuracy_score(y_multi_test, y_multi_pred_dt) * 100
    print("Decision Tree Multi-Class Classification Accuracy: ", multi_accuracy_dt)

    # Print and save classification report
    report_dt_multi = classification_report(y_multi_test, y_multi_pred_dt, target_names=le2_classes)
    print(report_dt_multi)
    save_report(report_dt_multi, "dt_multi")

    # **LSTM (Multi-Class Classification)**

    # Load the LSTM model from the models folder
    lstm_multi = load_model('models/lstm_multi.h5')

    # Reshape the data to be 3-dimensional [samples, time steps, features] for LSTM
    X_multi_lstm = X_multi.reshape((X_multi.shape[0], 1, X_multi.shape[1]))
    X_multi_lstm_test = X_multi_lstm[X_multi_test.index]

    # Predict and calculate accuracy
    y_multi_pred_lstm = lstm_multi.predict(X_multi_lstm_test).argmax(axis=1)
    multi_accuracy_lstm = accuracy_score(y_multi_test, y_multi_pred_lstm) * 100
    print("LSTM Multi-Class Classification Accuracy: ", multi_accuracy_lstm)

    # Print and save classification report
    report_lstm_multi = classification_report(y_multi_test, y_multi_pred_lstm, target_names=le2_classes)
    print(report_lstm_multi)
    save_report(report_lstm_multi, "lstm_multi")

    # **BiLSTM + Attention (Multi-Class Classification)**

    # Load the BiLSTM + Attention model from the models folder
    bilstm_attention_model = load_model('models/bilstm_attention_model.h5')

    # Reshape the data to be 3-dimensional [samples, time steps, features] for BiLSTM
    X_multi_blstm = X_multi.reshape((X_multi.shape[0], 1, X_multi.shape[1]))
    X_multi_blstm_test = X_multi_blstm[X_multi_test.index]

    # Predict and calculate accuracy
    y_multi_pred_blstm = bilstm_attention_model.predict(X_multi_blstm_test).argmax(axis=1)
    multi_accuracy_blstm = accuracy_score(y_multi_test, y_multi_pred_blstm) * 100
    print("BiLSTM + Attention Multi-Class Classification Accuracy: ", multi_accuracy_blstm)

    # Print and save classification report
    report_blstm = classification_report(y_multi_test, y_multi_pred_blstm, target_names=le2_classes)
    print(report_blstm)
    save_report(report_blstm, "blstm_attention")

if __name__ == "__main__":
    main()
