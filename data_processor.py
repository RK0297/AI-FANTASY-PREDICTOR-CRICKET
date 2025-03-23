import pandas as pd

def load_data(filepath):
    """
    Loads data from a CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return pd.DataFrame()

def preprocess_data(df):
    """
    Preprocesses the dataframe: cleans missing values,
    converts datatypes, and performs basic transformations.
    """
    if df.empty:
        return df
    # Remove rows with missing target values
    df = df.dropna(subset=['predicted_points'])
    
    # Fill missing values using forward fill
    df.fillna(method='ffill', inplace=True)
    
    # Convert numeric columns to floats
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_columns:
        df[col] = df[col].astype(float)
    
    return df
    
