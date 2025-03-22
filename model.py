import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from data_processor import scale_features

def train_random_forest_model(processed_data, n_estimators=100, max_depth=10):
    """
    Train a RandomForest model to predict player performance
    
    Args:
        processed_data: Preprocessed player data
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
        
    Returns:
        Trained RandomForest model
    """
    # If we're using the new format with playing status, skip training
    if 'IsPlaying' in processed_data.columns:
        # Return a dummy model - we'll use our pre-calculated predictions directly
        return DummyModel()
    
    # Scale features
    scaled_data, numeric_features = scale_features(processed_data)
    
    # Define target variable based on available columns
    if 'Dream11_Avg' in processed_data.columns:
        target = processed_data['Dream11_Avg']
    elif 'TOTAL DREAM 11 POINTS' in processed_data.columns:
        target = processed_data['TOTAL DREAM 11 POINTS']
    else:
        # Fallback to Overall_Impact_Score if available
        target = processed_data['Overall_Impact_Score'] if 'Overall_Impact_Score' in processed_data.columns else None
    
    if target is None:
        # If no suitable target found, create a dummy model
        return DummyModel()
    
    # Prepare features
    features = scaled_data[numeric_features].copy()
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(features, target)
    
    return model

def predict_player_performances(model, processed_data):
    """
    Predict player performances using the trained model
    
    Args:
        model: Trained RandomForest model or DummyModel
        processed_data: Preprocessed player data
        
    Returns:
        DataFrame with player performances
    """
    result_data = processed_data.copy()
    
    # Check if we're using a DummyModel (for playing status-based predictions)
    if isinstance(model, DummyModel):
        # The predictions are already calculated in the preprocessed data
        # Sort by predicted points (descending)
        result_data = result_data.sort_values('predicted_points', ascending=False)
        return result_data
    
    # For traditional model prediction:
    # Scale features
    scaled_data, numeric_features = scale_features(processed_data)
    
    # Predict performance
    predictions = model.predict(scaled_data[numeric_features])
    
    # Add predictions to data
    result_data['predicted_points'] = predictions
    
    # Sort by predicted points (descending)
    result_data = result_data.sort_values('predicted_points', ascending=False)
    
    return result_data


class DummyModel:
    """
    A dummy model class that doesn't actually do any predictions.
    This is used when we already have predictions calculated based on 
    playing status and lineup position.
    """
    def __init__(self):
        self.feature_importances_ = np.array([0.5, 0.3, 0.2])  # Dummy feature importances
        
    def predict(self, features):
        # Returns the same values for all inputs (not used in practice)
        return np.ones(len(features)) * 50
