# A Network Intrusion Detection Model Based on BiLSTM with Attention Mechanism

A network intrusion detection tool can identify and detect potential malicious activities or attacks by monitoring network traffic and system logs. The data within intrusion detection networks possesses characteristics that include a high degree of feature dimension and an unbalanced distribution across categories. Currently, the actual detection accuracy of some detection models is relatively low. To solve these problems, we propose a network intrusion detection model based on attention and BiLSTM (Bidirectional Long Short-Term Memory), which can introduce different attention weights for each vector in the feature vector that strengthen the relationship between some vectors and the detection attack type. The model also utilizes the advantage that BiLSTM can capture long-distance dependency relationships to obtain a higher detection accuracy. This model combined the advantages of the two models, adding a dropout layer between the two models to improve the detection accuracy while preventing training overfitting.

### Types of Intrusion Detection Systems

NIDS are broadly categorized into:

-   **Host-Based Intrusion Detection Systems (HIDS):** Monitors activities on specific devices (e.g., servers) by analyzing OS logs, system calls, and file access. HIDS can detect unusual behavior directly on the host but may struggle with network-level attacks.
-   **Network-Based Intrusion Detection Systems (NIDS):** Focuses on monitoring traffic across entire network segments. Placed strategically within the network (e.g., near firewalls), NIDS are effective for identifying suspicious network traffic patterns and known attack signatures but may miss host-specific threats.
-   **Hybrid Systems:** Combine aspects of HIDS and NIDS to enhance overall detection coverage.


### Detection Techniques

-   **Signature-Based Detection (Misuse Detection):** Matches network traffic against a database of known attack patterns or signatures. It is effective for known threats but ineffective for new or polymorphic attacks.
-   **Anomaly-Based Detection:** Uses statistical models, machine learning, or behavioral analysis to define "normal" network activity. When traffic deviates from this baseline, it’s flagged as potentially malicious. While anomaly detection can catch novel threats, it often has higher false positive rates.


# Project Overview


The purpose of your project, based on the context gathered from other chats, is to develop and improve a **network intrusion detection system (IDS)** that uses **attention** and **BiLSTM (Bidirectional Long Short-Term Memory)** models. This system is designed to detect unusual patterns or malicious activities in network traffic, which is crucial for **network security** and **privacy protection**. Specifically, your IDS aims to:

![image](https://github.com/user-attachments/assets/6ad704a2-79d5-43b9-875d-e327b4a0f8eb)


1.  **Identify network intrusions** (such as unauthorized access or policy violations) that can lead to security breaches.
2.  **Enhance detection accuracy** by combining deep learning techniques (BiLSTM and multi-head attention) to handle long-distance dependencies and automatically learn important features.
3.  **Classify different types of intrusions** more precisely, not just predicting general attacks but also distinguishing among different types, improving accuracy over standard methods.
4.  **Improve performance on datasets** such as  NSLKDD with already achieved high detection accuracies.

In addition, the project also seeks to address challenges like the **training sample requirements**, **feature selection**, and **computational time**, aiming to make the detection system more effective and efficient.

## Data Overview

The **NSL-KDD dataset** is a widely used dataset for evaluating and testing Intrusion Detection Systems (IDS). It is a refined version of the original **KDD Cup 1999 dataset**, which was used in a competition to detect network intrusions. The NSL-KDD dataset was introduced to address some of the limitations of the original KDD dataset, such as redundant records and imbalanced class distribution.

### Key Features of the NSL-KDD Dataset:

1.  **Purpose**: The dataset is used to train and evaluate machine learning and deep learning models for network intrusion detection. It contains both normal network traffic and malicious activity, categorized into different types of attacks.
    
2.  **Data Format**: The dataset consists of network traffic data that is pre-processed into feature vectors. Each record in the dataset represents a connection (or session) in the network, and each record has multiple features that describe the session. These features include:
    
    -   **Basic Features**: Connection characteristics (e.g., duration, protocol type, service, etc.).
    -   **Content Features**: Information about the data payload (e.g., number of failed login attempts, number of connections).
    -   **Traffic Features**: Statistical features (e.g., number of connections from the same source to a destination).
    -   **Label**: The attack class or whether the connection is normal.
3.  **Attack Categories**: The dataset contains multiple types of network attacks, grouped into several categories, including:
    
    -   **DoS (Denial of Service)**: Attacks that overwhelm a network or server, making it unavailable.
    -   **Probe**: Scanning and probing attempts to gather information about a system or network.
    -   **R2L (Remote to Local)**: Attacks where an attacker tries to send packets to a machine to exploit vulnerabilities and gain unauthorized access.
    -   **U2R (User to Root)**: Attacks where an attacker gains root access to a machine by exploiting vulnerabilities.
    -   **Normal**: Normal network connections without any intrusion attempts.
4.  **Improved Version**: The NSL-KDD dataset addresses several issues present in the original KDD Cup dataset, such as:
    
    -   **Redundant Records**: NSL-KDD removes duplicate records, making it a cleaner and more realistic representation of network traffic.
    -   **Class Imbalance**: It reduces the disproportionate number of normal records compared to attack records, which makes it more suitable for evaluating machine learning algorithms.
5.  **Data Split**: The dataset is typically split into training and testing sets:
    
    -   **Training Set**: Contains both normal and attack records to train the model.
    -   **Test Set**: Contains unseen data used to evaluate the model's performance.
6.  **Size**: The NSL-KDD dataset consists of **125,973 training records** and **22,544 testing records**, making it a moderate-sized dataset suitable for testing machine learning algorithms.
`Dual Classification`	  	
![Pie_chart_binary](https://github.com/user-attachments/assets/9f155912-51bc-4d8a-b134-fceaf565f19a)
`Multi Classification`
![Pie_chart_multi](https://github.com/user-attachments/assets/a7d44def-8e3c-4e5c-91d1-0761d3dca7f5)

								

## **Design and Implementation of BiLSTM with Attention**

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

 **Model Selection**: After evaluating various models, we decided to implement a **BiLSTM** architecture due to its ability to process sequential data and capture both **forward** and **backward dependencies** in the dataset, which is essential for intrusion detection where attack patterns can span across time.
-   **Attention Mechanism**: We integrated an **attention mechanism** to enable the model to assign different levels of importance to different features of the input data. This mechanism allowed us to focus the model’s attention on the most critical parts of the data (e.g., specific attack types or unusual patterns), improving detection accuracy.
-   **Data Handling**: We ensured the data was preprocessed effectively, which involved **normalization**, **one-hot encoding** for labels, and addressing class imbalance with **oversampling** (SMOTE).

## Evaluation and Performance Metrics

Each model’s performance is evaluated using:

-   **Accuracy**: Proportion of correct predictions.
-   **Precision**: Correctness of positive predictions.
-   **Recall**: Ability to detect all relevant instances.
-   **F1 Score**: Balance between precision and recall.

These metrics are summarized in classification report files for each model and saved in `classification_report_*.txt` files.

After they are plotter using following:
-   **Accuracy vs Epochs**: For both training and validation accuracy.
-   **Loss vs Epochs**: For both training and validation loss.
### LSTM Binary Classification
![accuracy_loss_lstm](https://github.com/user-attachments/assets/85ec71eb-df7d-4344-a3ac-9a56316cffc7)


### LSTM Multi Classification
![accuracy_loss_lstm_multi](https://github.com/user-attachments/assets/12c3167e-a76b-4a76-8b34-b259a82f5ae5)



### BiLSTM with ATTENTION
![accuracy_loss_blstm_multi](https://github.com/user-attachments/assets/81dbd45a-a836-4171-b391-90e3aa47df08)


## **Key Achievements**:

-   The model showed significant improvements in terms of detection accuracy and F1-score compared to other models tested on the **NSL-KDD dataset**.
-   We demonstrated the model's ability to differentiate between **normal traffic** and various **intrusion types** with high precision and recall.
-   The successful integration of **BiLSTM** and the **attention mechanism** enhanced the model's performance, particularly in handling complex attack patterns.
![final_results](https://github.com/user-attachments/assets/7049dd43-f01e-405e-92ec-a72faeab16ab)

  


## Model Performance Visualization

-   **Accuracy Comparison**: Bar plot comparing model accuracies.
-   **Training Metrics**: For deep learning models, additional plots visualize training and validation accuracy and loss across epochs.


## 
## Results and Insights

This project provides a comparative view of several machine learning techniques for network intrusion detection. The models provide a baseline for binary and multi-class classifications, showing how classical machine learning and neural network approaches perform on the tas

## **Future Work**

"In the future, we plan to improve the model further by:

-   Exploring advanced techniques like **multi-head attention** and **transformers** for even better performance.
-   Enhancing the model's **generalization** to unknown attack types.
-   Investigating **online learning** techniques to adapt the model to real-time data changes.
-   Reducing **sampling randomness** in the dataset balancing process to ensure more accurate results."
