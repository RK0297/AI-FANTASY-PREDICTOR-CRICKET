import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(file):
    """
    Load player data from uploaded file
    
    Args:
        file: Uploaded file object
    
    Returns:
        pandas DataFrame with player data
    """
    df = pd.read_csv(file)
    # Standardize column names by stripping whitespace
    df.columns = df.columns.str.strip()
    return df

def preprocess_data(df):
    """
    Preprocess player data for model training
    
    Args:
        df: DataFrame with player data
    
    Returns:
        Preprocessed DataFrame
    """
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Handle the new format from Book1.csv
    # Rename columns to match our model's expectations
    column_mapping = {
        'Player Name': 'Player',
        'Player Type': 'Role',
        'Credits': 'Credits',
        'Team': 'Team',
        'IsPlaying': 'IsPlaying',
        'lineupOrder': 'lineupOrder'
    }
    
    # Apply renaming for columns that exist
    for old_col, new_col in column_mapping.items():
        if old_col in data.columns:
            data.rename(columns={old_col: new_col}, inplace=True)
    
    # Fill missing values
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].fillna(0)
    
    # Create role-specific features
    if 'Role' in data.columns:
        # Create role indicators
        data['is_batsman'] = data['Role'].str.contains('Batter').astype(int)
        data['is_bowler'] = data['Role'].str.contains('Bowler').astype(int)
        data['is_allrounder'] = data['Role'].str.contains('All-rounder').astype(int)
        data['is_wicketkeeper'] = data['Role'].str.contains('Wicketkeeper').astype(int)
    
    # Add playing status feature
    if 'IsPlaying' in data.columns:
        # Convert playing status to numerical value for model
        data['is_playing'] = data['IsPlaying'].apply(
            lambda x: 1 if x == 'PLAYING' else 0.5 if x == 'X_FACTOR_SUBSTITUTE' else 0
        )
    
    # Add lineup order as feature (0 for non-playing players)
    if 'lineupOrder' in data.columns:
        # Convert to numeric and normalize
        data['lineup_position'] = data['lineupOrder'].astype(float)
        max_lineup = data['lineup_position'].max() if data['lineup_position'].max() > 0 else 1
        # Normalize so that opening batsmen (position 1,2) get higher scores
        data['lineup_position'] = (max_lineup - data['lineup_position'] + 1) / max_lineup
        # Replace 0s with minimal value (for non-playing players)
        data.loc[data['lineup_position'] <= 0, 'lineup_position'] = 0.1
    
    # Create synthetic performance metrics based on playing status and lineup
    # This is used when historical stats aren't available
    data['predicted_points'] = (
        data['Credits'] * 0.5 +  # Higher credit players likely perform better
        (data['is_playing'] if 'is_playing' in data.columns else 0) * 5 +  # Playing status boost
        (data['lineup_position'] if 'lineup_position' in data.columns else 0) * 3  # Lineup position boost
    )
    
    # Ensure we have the required columns
    required_columns = ['Player', 'Team', 'Role', 'Credits', 'predicted_points']
    for col in required_columns:
        if col not in data.columns:
            if col == 'Credits':
                # Try to use the Credits column (note the space at the end)
                if 'Credits ' in data.columns:
                    data.rename(columns={'Credits ': 'Credits'}, inplace=True)
                else:
                    data[col] = 8.0  # Default value
            else:
                data[col] = "Unknown" if col in ['Player', 'Team', 'Role'] else 0.0
    
    return data

def scale_features(data):
    """
    Scale numerical features for model training
    
    Args:
        data: DataFrame with features
    
    Returns:
        DataFrame with scaled features and feature names
    """
    # Identify numeric columns to scale (excluding target and categorical variables)
    exclude_cols = ['Player', 'Team', 'Role', 'Credits', 'predicted_points']
    numeric_cols = [col for col in data.columns if col not in exclude_cols and data[col].dtype in [np.float64, np.int64]]
    
    # Create a copy of the data
    scaled_data = data.copy()
    
    # Scale numeric features
    if numeric_cols:
        scaler = StandardScaler()
        scaled_data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    
    return scaled_data, numeric_cols
