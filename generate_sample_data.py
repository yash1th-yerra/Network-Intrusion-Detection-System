import pandas as pd
import numpy as np

# Generate random data with 93 features and 1 time step for 10 samples
num_samples = 10  # Number of samples (rows)
num_features = 93  # Number of features per sample

# Generate random values for the sample data (you can adjust this to real data)
# Random data in the range of [0, 1] for example
X_sample = np.random.rand(num_samples, num_features)

# Create a DataFrame from the random data
df_sample = pd.DataFrame(X_sample)

# Save the DataFrame to a CSV file
df_sample.to_csv('sample_input.csv', index=False)

print("sample_input.csv has been generated successfully.")

