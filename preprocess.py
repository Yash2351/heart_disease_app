import pandas as pd
import numpy as np

def load_and_clean_data(file_path='processed.cleveland.data'):
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
        'ca', 'thal', 'target'
    ]

    df = pd.read_csv(file_path, names=column_names)
    
    # Replace '?' with NaN and drop rows with missing values
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Convert all columns to numeric
    df = df.apply(pd.to_numeric)

    # Binarize the target: 0 (no heart disease), 1 (heart disease)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

    return df
