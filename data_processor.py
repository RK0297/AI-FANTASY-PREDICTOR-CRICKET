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
    
    # Fill missing values
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].fillna(0)
    
    # Create feature for boundary percentage if not exists
    if 'Boundary_Percentage' in data.columns:
        data['Boundary_Percentage'] = data['Boundary_Percentage'].fillna(0)
    
    # Create consistency features
    if 'Dream11_Avg' in data.columns and 'TOTAL DREAM 11 POINTS' in data.columns and 'Innings' in data.columns:
        # Points per innings played
        data['Points_per_innings'] = data['TOTAL DREAM 11 POINTS'] / data['Innings'].replace(0, 1)
        
        # Consistency score - how close average is to total divided by innings
        data['consistency_score'] = data['Dream11_Avg'] / (data['TOTAL DREAM 11 POINTS'] / data['Innings'].replace(0, 1) + 0.001)
        data['consistency_score'] = data['consistency_score'].apply(lambda x: min(x, 1.0))
    
    # Create role-specific features
    if 'Role' in data.columns:
        # Create role indicators
        data['is_batsman'] = data['Role'].str.contains('Batter').astype(int)
        data['is_bowler'] = data['Role'].str.contains('Bowler').astype(int)
        data['is_allrounder'] = data['Role'].str.contains('All-rounder').astype(int)
        data['is_wicketkeeper'] = data['Role'].str.contains('Wicketkeeper').astype(int)
    
    # Create form metrics
    if 'Fantasy Performance Rating' in data.columns:
        data['form'] = data['Fantasy Performance Rating']
    
    # Performance metrics
    features_to_include = ['Player', 'Team', 'Role', 'Credits', 'Dream11_Avg', 'Overall_Impact_Score',
                         'Fantasy Performance Rating', 'is_batsman', 'is_bowler', 
                         'is_allrounder', 'is_wicketkeeper']
    
    # Add batting features if they exist
    batting_features = ['Runs', 'Strike Rate', 'Batting Average', 'Boundary_Percentage']
    for feature in batting_features:
        if feature in data.columns:
            features_to_include.append(feature)
    
    # Add bowling features if they exist
    bowling_features = ['Wickets', 'Economy Rate', 'Bowling Average', 'Bowling  Strike_rate']
    for feature in bowling_features:
        if feature in data.columns:
            features_to_include.append(feature)
    
    # Filter to features that actually exist in the dataframe
    features_to_include = [f for f in features_to_include if f in data.columns]
    
    # Select features
    processed_data = data[features_to_include].copy()
    
    # Add predicted points column (will be filled by model later)
    processed_data['predicted_points'] = 0.0
    
    return processed_data

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
