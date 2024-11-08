# Data Preprocessing for Network Intrusion Detection

This guide covers the data preprocessing steps required to prepare a dataset for a network intrusion detection system (IDS) using machine learning models. The preprocessing includes loading and cleaning the dataset, normalizing features, encoding categorical variables, and setting up binary and multi-class labels for classification tasks.


# Table of Contents
-   [Requirements](#requirements)
-   [Data Loading](#data-loading)
-   [Data Cleaning and Transformation](#data-cleaning-and-transformation)
-   [Normalization](#normalization)
-   [One-Hot Encoding](#one-hot-encoding)
-   [Binary Classification](#binary-classification)
-   [Multi-Class Classification](#multi-class-classification)
-   [Feature Selection](#feature-selection)
-   [Saving Processed Data](#saving-processed-data)

## Requirements
Install the required libraries by running:

`pip install pandas numpy scikit-learn matplotlib seaborn`

## Data Loading
**Import the required libraries**:
-   `import pandas as pd
    import numpy as np` 
    
-   **Load the dataset**: The dataset does not contain column names, so we provide them manually.
- -   `col_names = ["duration","protocol_type","service","flag","src_bytes", ... ,"label","difficulty_level"]
    data = pd.read_csv('KDDTrain+.txt', header=None, names=col_names)` 
    
-   **Drop irrelevant columns**: Remove `difficulty_level` as it is not needed for classification.
  `data.drop('difficulty_level', axis=1, inplace=True)` 
    

## Data Cleaning and Transformation
**Classify Attack Labels**: Group various attack types into four main categories (`DoS`, `R2L`, `Probe`, `U2R`).
 `def change_label(df):
        df.label.replace(['apache2', 'back', ...], 'Dos', inplace=True)
        df.label.replace(['ftp_write', ...], 'R2L', inplace=True)
        df.label.replace(['ipsweep', ...], 'Probe', inplace=True)
        df.label.replace(['buffer_overflow', ...], 'U2R', inplace=True)
    change_label(data)` 
    
-   **Verify Class Distribution**:
`data['label'].value_counts()`


## Normalization
Standardize 38 numerical columns using `StandardScaler` to improve model performance.

``` from sklearn.preprocessing import StandardScaler
numerical_col = data.select_dtypes(include='number').columns
std_scaler = StandardScaler()

def normalization(df, col):
    for i in col:
        df[i] = std_scaler.fit_transform(df[i].values.reshape(-1, 1))
    return df

data = normalization(data, numerical_col)
```

## One-Hot Encoding
Convert categorical columns (`protocol_type`, `service`, `flag`) into binary dummy variables.

`categorical_col = ['protocol_type', 'service', 'flag']
categorical = pd.get_dummies(data[categorical_col]) `


# Binary Classification
Convert labels into two categories: `normal` and `abnormal`.

```bin_label = pd.DataFrame(data.label.map(lambda x: 'normal' if x == 'normal' else 'abnormal'))
bin_data = data.copy()
bin_data['label'] = bin_label

from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
bin_data['intrusion'] = bin_label.apply(le1.fit_transform)
np.save("le1_classes.npy", le1.classes_, allow_pickle=True)
bin_data = pd.get_dummies(bin_data, columns=['label'], prefix="", prefix_sep="")
```


## Multi-Class Classification
Categorize labels into multiple classes (`Dos`, `Probe`, `U2R`, `R2L`, `normal`).
```
multi_data = data.copy()
multi_label = pd.DataFrame(multi_data.label)
le2 = LabelEncoder()
multi_data['intrusion'] = multi_label.apply(le2.fit_transform)
np.save("le2_classes.npy", le2.classes_, allow_pickle=True)
multi_data = pd.get_dummies(multi_data, columns=['label'], prefix="", prefix_sep="")
```

## Feature Selection
Select the most relevant features based on Pearson correlation coefficients.
```
corr = bin_data[numerical_col].corr()
highest_corr = corr['intrusion'][abs(corr['intrusion']) > 0.5].sort_values()
selected_features = ['count', 'srv_serror_rate', 'serror_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'logged_in', 'dst_host_same_srv_rate', 'dst_host_srv_count', 'same_srv_rate']
numeric_bin = bin_data[selected_features].join(categorical)
bin_data = numeric_bin.join(bin_data[['intrusion', 'abnormal', 'normal', 'label']])
```
Repeat similar steps for multi-class data.

## Saving Processed Data
Save the final processed datasets for binary and multi-class classification.

```
bin_data.to_csv("bin_data.csv")
multi_data.to_csv("multi_data.csv")
```

## Visualization

Plot the class distributions for binary and multi-class labels using pie charts.

```
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,8))
plt.pie(bin_data.label.value_counts(), labels=bin_data.label.unique(), autopct='%0.2f%%')
plt.savefig('Pie_chart_binary.png')
plt.show()

plt.figure(figsize=(8,8))
plt.pie(multi_data.label.value_counts(), labels=multi_data.label.unique(), autopct='%0.2f%%')
plt.savefig('Pie_chart_multi.png')
plt.show()
```
## Conclusion

This preprocessing pipeline prepares the dataset for an IDS by:

-   Cleaning and categorizing attack labels.
-   Standardizing numerical features.
-   Encoding categorical data.
-   Setting up binary and multi-class classification labels.
-   Selecting significant features for optimal model performance.




